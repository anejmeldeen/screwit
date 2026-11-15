"""The add_part tool: put a new component into the object (SPEC 6.3 tool layer).

A new part is either a simple solid (box, cylinder or sphere) sized and placed in the scan frame, or
the model of a CAD job placed exactly where the viewer overlay showed it. The part is saved like any
other part (database, and parts.json / scene_graph.json in live storage), so it appears in the parts
list, the hierarchy and the 3D viewer, and the part tools and routes work on it.

Trust boundary: every argument comes from the model. Numbers must be finite and in range, sizes are
clamped, the quaternion must be unit, ids must exist. The part id and every storage key are made here,
never derived from model input, and the result goes through the Part contract before anything is
written. Object-agnostic: nothing here knows what kind of object it is adding to.
"""

from __future__ import annotations

import io
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import get_args

import numpy as np

from screwit.agent.context import clean_text, fmt_mm, fmt_vec
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import Part, Relation, SceneGraph
from screwit.contracts.models import RelationType

KINDS = ("box", "cylinder", "sphere", "cad", "cone", "triangular_prism", "cat_ear")
RELATION_TYPES: tuple[str, ...] = get_args(RelationType)
MIN_SIZE_MM = 0.1
MAX_SIZE_MM = 2000.0
MAX_COORD_MM = 5000.0
MAX_CAD_BYTES = 64 * 1024 * 1024
_THUMBNAIL_MAX_FACES = 20000
_LABEL_MAX = 80
_DESCRIPTION_MAX = 500
_MATERIAL_MAX = 60
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_ALLOWED_ARGS = frozenset({"label", "kind", "size_mm", "center_mm", "rotation_quat", "cad_id",
                           "description", "material", "attach_to", "relation"})
_IDENTITY = (0.0, 0.0, 0.0, 1.0)


class AddPartError(ValueError):
    """A rejected add_part request; the message tells the model what to fix."""


@dataclass
class PartAddedSignal:
    """Returned by add_part so the loop highlights the new part and gives the model `result`."""

    part_id: str
    result: dict = field(default_factory=dict)


def input_schema() -> dict:
    """The add_part input schema (Anthropic tool form)."""
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Short name for the new part."},
            "kind": {"type": "string", "enum": list(KINDS),
                     "description": "box, cylinder, sphere, cone, triangular_prism or cat_ear; cad for the model "
                                    "of a CAD result (needs cad_id)."},
            "size_mm": {**vec3, "description": (
                "Required for all primitives: the part's size in mm along the scan-frame X, "
                "Y and Z axes (before any rotation_quat). box: edge lengths [x, y, z]. cylinder: two "
                "equal values are the diameter and the third is the length along the cylinder's "
                "axis, which runs along that third axis (all three equal: along Z). For example "
                "[30, 15, 30] is a disc 30 mm across and 15 mm thick whose axis runs along Y. "
                f"sphere: [d, d, d] (unequal values give an ellipsoid). Each value "
                f"{MIN_SIZE_MM}-{MAX_SIZE_MM:g} mm.")},
            "surface": {"type": "string", "enum": list(_FACES),
                        "description": "Snap onto this scan-axis face of attach_to instead of supplying center_mm."},
            "offset_uv": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"},
                          "description": "Position across the face, -1 to 1 on each remaining scan axis "
                                         "in XYZ order. Default [0,0] is the middle."},
            "center_mm": {**vec3, "description": (
                "Without surface placement: where the part's centre goes, [x, y, z] in mm "
                "in the scan frame (right-handed, Z up, z = 0 on the support surface).")},
            "rotation_quat": {"type": "array", "items": {"type": "number"}, "minItems": 4,
                              "maxItems": 4, "description": (
                "Optional, only to tilt the part off the scan axes: a unit quaternion [x, y, z, w] "
                "applied about the part's centre, with size_mm read as the size before rotating. "
                "Leave it out for a part lined up with X, Y or Z (size_mm already sets a cylinder's "
                "axis). A rotated cylinder must be given as [diameter, diameter, length], axis "
                "along Z before the rotation.")},
            "cad_id": {"type": "string", "description": (
                "Required for kind=cad: the cad_id of a CAD result for this workspace. The part "
                "lands where that CAD preview was shown; size_mm, center_mm and rotation_quat are "
                "not used.")},
            "description": {"type": "string", "description": "Optional one-sentence description."},
            "material": {"type": "string", "description": "Optional; only when the user named the "
                                                          "material, e.g. PLA or steel."},
            "attach_to": {"type": "string", "description": (
                "Existing part id, required with surface; records a "
                "relation in the scene graph.")},
            "relation": {"type": "string", "enum": list(RELATION_TYPES), "description": (
                "Relation from the new part to attach_to; default rests_on. Needs attach_to.")},
        },
        "required": ["label", "kind"],
    }


# ---------------------------------------------------------------- validation
def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AddPartError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AddPartError(f"{name} must be a finite number")
    return number


def _vector(value: object, name: str, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise AddPartError(f"{name} must be a list of {length} numbers")
    return [_number(v, f"{name}[{i}]") for i, v in enumerate(value)]


def _text(value: object, name: str, max_len: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AddPartError(f"{name} must be a string")
    return clean_text(value, max_len) or None


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise AddPartError(f"{name} must be an existing id (letters, digits, '_', '-' or '.')")
    return value


def _unit_quat(value: object) -> tuple[float, float, float, float]:
    q = _vector(value, "rotation_quat", 4)
    norm = math.sqrt(sum(c * c for c in q))
    if not 0.98 <= norm <= 1.02:
        raise AddPartError(f"rotation_quat must be a unit quaternion [x, y, z, w]; its norm is {norm:.3f}")
    x, y, z, w = (c / norm for c in q)
    return (x, y, z, w)


def _is_identity(quat: tuple[float, float, float, float]) -> bool:
    return abs(abs(quat[3]) - 1.0) < 1e-6


@dataclass
class _Request:
    label: str
    kind: str
    size: list[float] | None
    center: list[float] | None
    quat: tuple[float, float, float, float]
    cad_id: str | None
    description: str | None
    material: str | None
    attach_to: str | None
    relation: str | None
    notes: list[str]
    axis: int | None = None  # cylinder only: index (0 X, 1 Y, 2 Z) of the axis in size


def _parse(args: dict, part_ids: set[str]) -> _Request:
    if not isinstance(args, dict):
        raise AddPartError("add_part needs an object of arguments")
    unknown = sorted(set(args) - _ALLOWED_ARGS)
    if unknown:
        raise AddPartError(f"unknown argument(s) {unknown}; allowed: {sorted(_ALLOWED_ARGS)}")
    label = _text(args.get("label"), "label", _LABEL_MAX)
    if not label:
        raise AddPartError("label is required")
    kind = args.get("kind")
    if kind not in KINDS:
        raise AddPartError(f"kind must be one of {list(KINDS)}")
    notes: list[str] = []

    size = center = None
    quat = _IDENTITY
    cad_id = axis = None
    if kind == "cad":
        cad_id = _identifier(args.get("cad_id"), "cad_id")
        ignored = [k for k in ("size_mm", "center_mm", "rotation_quat") if args.get(k) is not None]
        if ignored:
            notes.append(f"{', '.join(ignored)} not used for kind=cad: the part sits where the CAD "
                         "preview was shown")
    else:
        if args.get("cad_id") is not None:
            raise AddPartError("cad_id is only used with kind=cad")
        if args.get("size_mm") is None or args.get("center_mm") is None:
            raise AddPartError(f"kind={kind} needs size_mm and center_mm")
        size = _vector(args["size_mm"], "size_mm", 3)
        if any(v <= 0 for v in size):
            raise AddPartError("size_mm values must be positive (mm)")
        clamped = [min(max(v, MIN_SIZE_MM), MAX_SIZE_MM) for v in size]
        if clamped != size:
            notes.append(f"size_mm clamped to {fmt_vec(clamped)} (allowed {MIN_SIZE_MM}-{MAX_SIZE_MM:g} mm)")
        size = clamped
        axis = _cylinder_axis(size) if kind == "cylinder" else None
        center = _vector(args["center_mm"], "center_mm", 3)
        if any(abs(c) > MAX_COORD_MM for c in center):
            raise AddPartError(f"center_mm must be within ±{MAX_COORD_MM:g} mm of the scan origin")
        if args.get("rotation_quat") is not None:
            quat = _unit_quat(args["rotation_quat"])
        if axis not in (None, 2) and not _is_identity(quat):
            raise AddPartError(
                f"size_mm {fmt_vec(size)} already puts the cylinder's axis along {'XYZ'[axis]}, so "
                "rotation_quat would turn it a second time. Leave rotation_quat out for an axis along "
                "X, Y or Z; to tilt the cylinder, give size_mm as [diameter, diameter, length] (axis "
                "along Z) together with the rotation.")

    attach_to = relation = None
    if args.get("attach_to") is not None:
        attach_to = _identifier(args["attach_to"], "attach_to")
        if attach_to not in part_ids:
            known = ", ".join(sorted(part_ids)[:30])
            raise AddPartError(f"attach_to {attach_to!r} is not a part of this object; parts: {known}")
        relation = args.get("relation") or "rests_on"
        if relation not in RELATION_TYPES:
            raise AddPartError(f"relation must be one of {list(RELATION_TYPES)}")
    elif args.get("relation") is not None:
        raise AddPartError("relation needs attach_to (the existing part it relates to)")

    return _Request(label=label, kind=kind, size=size, center=center, quat=quat, cad_id=cad_id,
                    description=_text(args.get("description"), "description", _DESCRIPTION_MAX),
                    material=_text(args.get("material"), "material", _MATERIAL_MAX),
                    attach_to=attach_to, relation=relation, notes=notes, axis=axis)


# ---------------------------------------------------------------- geometry
def _matrix(quat_xyzw: tuple[float, float, float, float], translate) -> np.ndarray:
    """4x4 transform: rotate by a contract quaternion [x, y, z, w], then translate (mm)."""
    import trimesh

    x, y, z, w = quat_xyzw
    matrix = trimesh.transformations.quaternion_matrix([w, x, y, z])  # trimesh order is [w, x, y, z]
    matrix[:3, 3] = [float(v) for v in translate]
    return matrix


def _cylinder_axis(size: list[float]) -> int:
    """Index of the cylinder's axis: the size_mm value that differs from the other two (the equal
    pair is the diameter); Z when all three are equal."""
    x, y, z = size

    def same(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-3)

    if same(x, y):
        return 2
    if same(x, z):
        return 1
    if same(y, z):
        return 0
    raise AddPartError("for a cylinder two size_mm values must be equal (the diameter); the third is "
                       "the length along its axis, e.g. [30, 15, 30] for an axis along Y")


def _primitive(kind: str, size: list[float], axis: int | None = None):
    """A solid centred on the origin whose extents along the scan X, Y, Z axes equal size (mm)."""
    import trimesh

    if kind == "cat_ear":
        # Rounded triangular shell with a raised rim and recessed front panel.
        # Rings share vertices at every seam, including the floor of the recess.
        corners = np.array([[-.5, -.5], [.5, -.5], [0, .5]])
        outline = []
        for i, corner in enumerate(corners):
            start = corner + .08 * (corners[i-1] - corner)
            end = corner + .08 * (corners[(i+1) % 3] - corner)
            for t in np.linspace(0, 1, 9):
                outline.append((1-t)**2 * start + 2*t*(1-t)*corner + t*t*end)
        x, z = np.asarray(outline).T
        rings = [(.88, .5), (.98, .4), (1, .25), (1, -.30), (.96, -.49),
                 (.87, -.50), (.78, -.43), (.69, -.22)]
        vertices = np.concatenate([np.column_stack((x*k, np.full_like(x, y), z*k)) for k, y in rings])
        n = len(x)
        faces = []
        for k in range(len(rings)-1):
            for i in range(n):
                a, b = k*n+i, k*n+(i+1) % n
                faces.extend([(a, b, b+n), (a, b+n, a+n)])
        last = (len(rings)-1)*n
        for i in range(1, n-1):
            faces.extend([(0, i+1, i), (last, last+i, last+i+1)])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(np.asarray(size)/mesh.extents)
        return mesh
    if kind == "box":
        return trimesh.creation.box(extents=size)
    if kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
        mesh.apply_scale(size)
        return mesh
    if kind in ("cone", "triangular_prism"):
        if kind == "cone":
            mesh = trimesh.creation.cone(radius=1, height=1, sections=96)
        else:
            vertices = [[x, y, z] for y in (-0.5, 0.5) for x, z in ((-.5, 0), (.5, 0), (0, 1))]
            mesh = trimesh.convex.convex_hull(vertices)
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(np.asarray(size) / mesh.extents)
        return mesh
    axis = _cylinder_axis(size) if axis is None else axis
    diameter = size[(axis + 1) % 3]
    mesh = trimesh.creation.cylinder(radius=diameter / 2.0, height=size[axis], sections=96)  # along Z
    if axis != 2:
        direction = [0.0, 0.0, 0.0]
        direction[axis] = 1.0
        mesh.apply_transform(trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction))
    return mesh


def _cad_mesh(reg: Registry, store: ScanStore, scan_id: str, cad_id: str):
    """The CAD job's STL (mm) and the placement the viewer overlay applies to it."""
    import trimesh

    job = store.get_cad_job(scan_id, cad_id)
    if job is None:
        raise AddPartError(f"there is no CAD result {cad_id!r} for this workspace")
    _, result = job
    if result.files is None or not result.files.stl:
        raise AddPartError(f"CAD result {cad_id!r} has no model to add (status {result.status})")
    head = reg.storage.head(result.files.stl)
    if head is None:
        raise AddPartError(f"the model file of CAD result {cad_id!r} is missing")
    if head["size"] > MAX_CAD_BYTES:
        raise AddPartError("that CAD model is too large to add as a part (64 MB limit)")
    try:
        mesh = trimesh.load(io.BytesIO(reg.storage.get_bytes(result.files.stl)), file_type="stl",
                            force="mesh")
    except Exception:  # noqa: BLE001 - a corrupt or unreadable STL is a request-level failure
        raise AddPartError(f"the model file of CAD result {cad_id!r} could not be read") from None
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise AddPartError(f"the model of CAD result {cad_id!r} has no surface")
    # Same transform as web/components/viewer/CadOverlay.tsx: position = translate_mm, quaternion =
    # rotation_quat, applied to the STL as stored. If the overlay's placement rule changes, change it
    # here too, so the part keeps landing where the preview was shown.
    placement = result.placement
    quat = tuple(placement.rotation_quat) if placement else _IDENTITY
    norm = math.sqrt(sum(c * c for c in quat))
    quat = tuple(c / norm for c in quat)
    translate = tuple(placement.translate_mm) if placement else (0.0, 0.0, 0.0)
    return mesh, quat, translate, result.status


def _thumbnail(mesh) -> bytes:
    from screwit.perception.render import render_mesh_png

    shown = mesh.convex_hull if len(mesh.faces) > _THUMBNAIL_MAX_FACES else mesh
    return render_mesh_png(np.asarray(shown.vertices, dtype=float), np.asarray(shown.faces, dtype=int))


# ---------------------------------------------------------------- scene graph
def graph_with_part(graph: SceneGraph, part_ids: list[str], new_id: str,
                    relation: Relation | None) -> SceneGraph:
    """The graph with new_id added, its part list matched to the parts that exist (graph order kept),
    and relations or hypotheses that name a part which no longer exists dropped, so it stays valid."""
    known = set(part_ids) | {new_id}
    order = [pid for pid in graph.parts if pid in known]
    order += [pid for pid in dict.fromkeys([*part_ids, new_id]) if pid not in order]
    relations = [r for r in graph.relations if r.a in known and r.b in known]
    if (relation is not None and relation.a in known and relation.b in known
            and not any((r.a, r.b, r.type) == (relation.a, relation.b, relation.type) for r in relations)):
        relations.append(relation)
    hypotheses = [h for h in graph.failure_hypotheses if h.part_id in known]
    return SceneGraph.model_validate({**graph.model_dump(mode="json"), "parts": order,
                                      "relations": [r.model_dump(mode="json") for r in relations],
                                      "failure_hypotheses": [h.model_dump(mode="json") for h in hypotheses]})


def _latest_graph_revision(scan_id: str) -> int:
    """The highest revision a stored scene graph row carries, so a new row is read back as latest."""
    from sqlmodel import select

    from screwit.app.db import SceneGraphRow, db_session

    with db_session() as session:
        revisions = session.exec(select(SceneGraphRow.reconstruction_revision)
                                 .where(SceneGraphRow.scan_id == scan_id)).all()
    return max(revisions, default=0)


# ---------------------------------------------------------------- the tool
_FACES = {"top": (2, 1), "bottom": (2, -1), "right": (0, 1), "left": (0, -1),
          "front": (1, -1), "back": (1, 1)}


def _surface_point(reg, anchor, face, uv):
    """Intersect a scan-axis ray with the actual triangles, choosing the outermost hit."""
    import trimesh

    if not isinstance(face, str) or face not in _FACES:
        raise AddPartError("unknown surface")
    uv = np.asarray(_vector(uv, "offset_uv", 2))
    if np.any(np.abs(uv) > 1):
        raise AddPartError("offset_uv must lie between -1 and 1")
    try:
        scene = trimesh.load(io.BytesIO(reg.storage.get_bytes(anchor.mesh_uri)), file_type="glb", force="scene")
        mesh = scene.to_geometry()
        triangles = mesh.triangles
        lo, hi = mesh.bounds
    except Exception as exc:
        raise AddPartError("could not read the attachment mesh") from exc
    axis, sign = _FACES[face]
    plane = [i for i in range(3) if i != axis]
    target = (lo + hi) / 2
    target[plane] += uv * (hi-lo)[plane] / 2
    a, b, c = triangles.transpose(1, 0, 2)
    u, v = (b-a)[:, plane], (c-a)[:, plane]
    w = target[plane] - a[:, plane]

    def cross(x, y):
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    det = cross(u, v)
    valid = np.abs(det) > 1e-12
    det = np.where(valid, det, 1)
    i, j = cross(w, v)/det, cross(u, w)/det
    hit = valid & (i >= -1e-8) & (j >= -1e-8) & (i+j <= 1+1e-8)
    if np.any(hit):
        depths = (a[:, axis] + i*(b-a)[:, axis] + j*(c-a)[:, axis])[hit]
        target[axis] = sign * np.max(sign*depths)
    else:
        # A hole or an open outline: use the closest projected triangle edge.
        ends = np.roll(triangles, -1, axis=1)
        edges = ends - triangles
        delta = target[plane] - triangles[:, :, plane]
        flat = edges[:, :, plane]
        t = np.clip((delta*flat).sum(2) / np.maximum((flat*flat).sum(2), 1e-20), 0, 1)
        points = (triangles + t[:, :, None]*edges).reshape(-1, 3)
        distances = np.linalg.norm(points[:, plane]-target[plane], axis=1)
        nearest = points[distances <= distances.min()+1e-7]
        target = nearest[np.argmax(sign*nearest[:, axis])].copy()
    return target, axis, sign


def add_part(reg: Registry, store: ScanStore, scan_id: str, args: dict) -> PartAddedSignal | dict:
    parts = store.list_parts(scan_id)
    part_ids = [p.part_id for p in parts]
    try:
        args = dict(args) if isinstance(args, dict) else args
        surface = args.pop("surface", None) if isinstance(args, dict) else None
        uv = args.pop("offset_uv", None) if isinstance(args, dict) else None
        if uv is not None and surface is None:
            raise AddPartError("offset_uv needs a surface")
        if surface is not None:
            if not args.get("attach_to"):
                raise AddPartError("surface needs attach_to")
            if args.get("center_mm") is not None:
                raise AddPartError("choose either surface or center_mm")
            if args.get("kind") != "cad":
                args["center_mm"] = [0, 0, 0]
        req = _parse(args, set(part_ids))
        if req.kind == "cad":
            mesh, quat, translate, cad_status = _cad_mesh(reg, store, scan_id, req.cad_id)
        else:
            mesh, quat, translate, cad_status = (_primitive(req.kind, req.size, req.axis), req.quat,
                                                 req.center, None)
        if surface is not None:
            anchor = next(p for p in parts if p.part_id == req.attach_to)
            point, axis, sign = _surface_point(reg, anchor, surface, uv if uv is not None else [0, 0])
            rotated = mesh.vertices @ _matrix(quat, [0, 0, 0])[:3, :3].T
            depths = sign * rotated[:, axis]
            support = rotated[np.isclose(depths, depths.min(), rtol=0, atol=1e-7)]
            contact = (support.min(axis=0) + support.max(axis=0)) / 2
            translate = point - contact
            if np.any(np.abs(translate) > MAX_COORD_MM):
                raise AddPartError("computed placement exceeds the scan coordinate limit")
    except AddPartError as e:
        return {"error": "bad_request", "message": str(e)}

    # Box in the part's own frame, then placed: the OBB is exact for a rotated solid too.
    lo, hi = (np.asarray(b, dtype=float) for b in mesh.bounds)
    matrix = _matrix(quat, translate)
    mesh.apply_transform(matrix)
    center = (matrix[:3, :3] @ ((lo + hi) / 2.0) + matrix[:3, 3]).tolist()
    extent = np.maximum(hi - lo, 0.001).tolist()

    part_id = _new_part_id(set(part_ids))
    mesh_key = f"scans/{scan_id}/parts/{part_id}.glb"
    thumb_key = f"scans/{scan_id}/renders/{part_id}.png"
    scan = store.get_scan(scan_id)
    revision = scan.reconstruction_revision if scan else max(
        (p.reconstruction_revision for p in parts), default=0)
    shape = _shape_words(req, extent)
    part = Part.model_validate({
        "part_id": part_id, "reconstruction_revision": revision, "measurement_revision": 1,
        "label": req.label, "label_confidence": 1.0,
        "description": req.description or f"Added in chat: {shape}.",
        "mesh_uri": mesh_key, "thumbnail_uri": thumb_key,
        "centroid_mm": center,
        "obb": {"center": center, "extent": extent, "rotation_quat": list(quat)},
        "dimensions": _dimensions(req),
        "features": [],
        "material_guess": {"value": req.material or "unspecified",
                           "confidence": 1.0 if req.material else 0.0},
        "source": "user_edited",
    })
    relation = (Relation(a=part_id, b=req.attach_to, type=req.relation, confidence=1.0,
                         note="added in chat") if req.attach_to else None)
    graph = store.get_scene_graph(scan_id)
    new_graph = graph_with_part(graph, part_ids, part_id, relation) if graph is not None else None

    _put_asset(reg, mesh_key, mesh.export(file_type="glb"), "model/gltf-binary")
    _put_asset(reg, thumb_key, _thumbnail(mesh), "image/png")
    store.save_part(scan_id, part)
    if new_graph is not None:
        store.save_scene_graph(new_graph, max(revision, _latest_graph_revision(scan_id)))
    warnings = [] if reg.settings.is_mock else _update_live_storage(reg, scan_id, part, relation)

    result = {
        "status": "added", "part_id": part_id, "chip": f"[[part:{part_id}]]", "label": part.label,
        "kind": req.kind, "shape": shape,
        "center_mm": [round(v, 2) for v in center], "extent_mm": [round(v, 2) for v in extent],
        "rotation_quat": [round(v, 4) for v in quat],
        "relation": ({"a": part_id, "type": req.relation, "b": req.attach_to} if relation else None),
        "scene_graph_updated": new_graph is not None,
        "message": "Saved. It now appears in the parts list, the hierarchy and the 3D viewer.",
    }
    if req.kind == "cad":
        result["cad_id"] = req.cad_id
        result["cad_status"] = cad_status
    if req.notes or warnings:
        result["notes"] = req.notes + warnings
    return PartAddedSignal(part_id=part_id, result=result)


def _new_part_id(existing: set[str]) -> str:
    while True:
        part_id = "added_" + uuid.uuid4().hex[:12]
        if part_id not in existing:
            return part_id


def _shape_words(req: _Request, extent: list[float]) -> str:
    if req.kind == "cad":
        return f"the model of CAD result {req.cad_id}, {' x '.join(fmt_mm(v) for v in extent)} mm"
    size = req.size or extent
    if req.kind in ("cone", "triangular_prism", "cat_ear", "box"):
        return f"{req.kind} {' x '.join(fmt_mm(v) for v in size)} mm"
    if req.kind == "cylinder":
        axis = req.axis if req.axis is not None else 2
        return (f"cylinder {fmt_mm(size[(axis + 1) % 3])} mm across and {fmt_mm(size[axis])} mm long, "
                f"axis along {'XYZ'[axis]}")
    if size[0] == size[1] == size[2]:
        return f"sphere {fmt_mm(size[0])} mm across"
    return f"ellipsoid {' x '.join(fmt_mm(v) for v in size)} mm"


def _dimensions(req: _Request) -> list[dict]:
    """The sizes the user chose, as user-set values (no scan value): shown as 'your value'."""
    if req.size is None:
        return []
    x, y, z = req.size
    if req.kind in ("box", "cone", "triangular_prism", "cat_ear"):
        named = [("size_x", x), ("size_y", y), ("size_z", z)]
    elif req.kind == "cylinder":
        axis = req.axis if req.axis is not None else 2
        named = [("diameter", req.size[(axis + 1) % 3]), ("length", req.size[axis])]
    else:
        named = [("diameter", x)] if x == y == z else [("diameter_x", x), ("diameter_y", y), ("diameter_z", z)]
    return [{"name": name, "value": {"scan_mm": None, "caliper_mm": round(value, 3),
                                     "user_confirmed": True, "preferred": "caliper"}}
            for name, value in named]


def _put_asset(reg: Registry, key: str, data: bytes, content_type: str) -> None:
    """Write through the configured storage. The mock fixture store is read-only and serves generated
    assets from <data_dir>/generated (as the mock merge does), so write there in that case."""
    from screwit.storage.keys import validate_key

    validate_key(key)
    try:
        reg.storage.put_bytes(key, data, content_type)
    except NotImplementedError:
        root = (reg.settings.data_dir / "generated").resolve()
        dest = (root / key).resolve()
        if not dest.is_relative_to(root):
            raise ValueError("asset key escapes the generated directory") from None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    if reg.storage.head(key) is None:
        raise RuntimeError(f"storage did not keep {key}")


def _update_live_storage(reg: Registry, scan_id: str, part: Part, relation: Relation | None) -> list[str]:
    """Live mode: add the part to parts.json and scene_graph.json, which the perception service reads
    for PATCH, merge, measure and rescale. Each file is extended as stored rather than replaced from
    the database, so nothing another writer put there is lost. Returns warnings instead of raising:
    the part is already saved, and a failed tool would make the model add it twice."""
    warnings: list[str] = []
    parts_key, graph_key = f"scans/{scan_id}/parts.json", f"scans/{scan_id}/scene_graph.json"
    store = ScanStore(reg.data)
    try:
        if reg.storage.head(parts_key) is not None:
            stored = [Part.model_validate(p) for p in json.loads(reg.storage.get_bytes(parts_key))]
        else:
            stored = store.list_parts(scan_id)
        stored = [p for p in stored if p.part_id != part.part_id] + [part]
        payload = json.dumps([p.model_dump(mode="json") for p in stored], indent=2).encode()
        reg.storage.put_bytes(parts_key, payload, "application/json")
    except Exception as e:  # noqa: BLE001
        return [f"parts.json in storage was not updated ({type(e).__name__}); editing this part may fail"]
    try:
        if reg.storage.head(graph_key) is not None:
            graph = SceneGraph.model_validate_json(reg.storage.get_bytes(graph_key))
        else:
            graph = store.get_scene_graph(scan_id)
        if graph is not None:
            ids = [p.part_id for p in stored if p.part_id != part.part_id]
            updated = graph_with_part(graph, ids, part.part_id, relation)
            reg.storage.put_bytes(graph_key, updated.model_dump_json(indent=2).encode(), "application/json")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"scene_graph.json in storage was not updated ({type(e).__name__})")
    return warnings
