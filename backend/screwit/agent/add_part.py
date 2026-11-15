"""The add_part and probe_surface tools: put a new component into the object (SPEC 6.3 tool layer).

A new part is either a solid made from a few numbers (box, cylinder, cone, pyramid, triangular prism,
sphere, torus, capsule, cat ear, a flat profile extruded from an outline, or one of the supplied
headphone-ear meshes) or the model of a CAD job placed exactly where the
viewer overlay showed it. Each solid is built in its own frame: width along local X, depth along local
Y, height along local Z, with the centre of its base on the origin. It is placed with direction vectors
a language model can reason about: `up` (where the height points), `width_axis`, and exactly one of
on_part + direction (stand it on the outer surface of an existing part), surface (snap onto a
scan-axis face of attach_to), base_mm or center_mm.
probe_surface runs the same surface search without adding anything.

The part is saved like any other part (database, and parts.json / scene_graph.json in live storage), so
it appears in the parts list, the hierarchy and the 3D viewer, and the part tools and routes work on it.

Trust boundary: every argument comes from the model. Numbers must be finite and in range, sizes are
clamped, vectors must not be zero, outlines must be simple polygons, ids must exist. The part id and
every storage key are made here, never derived from model input, and the result goes through the Part
contract before anything is written. Object-agnostic: nothing here knows what kind of object it is
adding to.
"""

from __future__ import annotations

import io
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from importlib.resources import files
from typing import get_args

import numpy as np

from screwit.agent.context import clean_text, fmt_mm, fmt_vec
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import Part, Relation, SceneGraph
from screwit.contracts.models import RelationType

HEADPHONE_EARS = ("headphone_ear_left", "headphone_ear_right")
KINDS = ("box", "cylinder", "cone", "pyramid", "triangular_prism", "sphere", "torus", "capsule", "cat_ear",
         "profile", *HEADPHONE_EARS, "cad")
SIZED_KINDS = frozenset({"box", "cylinder", "cone", "pyramid", "triangular_prism", "sphere", "torus",
                         "capsule", "cat_ear", *HEADPHONE_EARS})
RELATION_TYPES: tuple[str, ...] = get_args(RelationType)
MIN_SIZE_MM = 0.1
MAX_SIZE_MM = 2000.0
MAX_COORD_MM = 5000.0
MAX_CAD_BYTES = 64 * 1024 * 1024
MIN_OUTLINE_POINTS = 3
MAX_OUTLINE_POINTS = 64
DEFAULT_EMBED_MM = 2.0
CONTACT_TOLERANCE_MM = 0.5
_SECTIONS = 96
_THUMBNAIL_MAX_FACES = 20000
_LABEL_MAX = 80
_DESCRIPTION_MAX = 500
_MATERIAL_MAX = 60
_LISTED_IDS = 30
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_PLACEMENTS = ("on_part", "surface", "base_mm", "center_mm")
# surface: (scan axis, sign) of the face of attach_to that a new part snaps onto.
_FACES = {"top": (2, 1), "bottom": (2, -1), "right": (0, 1), "left": (0, -1),
          "front": (1, -1), "back": (1, 1)}
_SHAPE_ARGS = {"top_scale": ("cone", "pyramid"), "tip_radius_mm": ("cone",), "sides": ("pyramid",),
               "outline_mm": ("profile",), "thickness_mm": ("profile",)}
_ALLOWED_ARGS = frozenset({"label", "kind", "size_mm", *_SHAPE_ARGS, *_PLACEMENTS, "direction",
                           "embed_mm", "offset_uv", "up", "width_axis", "cad_id", "description", "material",
                           "attach_to", "relation"})
_PROBE_ARGS = frozenset({"part_id", "direction"})
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_Z = np.array([0.0, 0.0, 1.0])
# The default width axis is the horizontal tangent cross(Z, up); within about 10 degrees of vertical
# that tangent is unstable (scan normals are noisy), so [1, 0, 0] is used instead.
_NEAR_VERTICAL = 0.17
# Added parts carry one vertex colour so render_view shows them apart from the scan. The web viewer
# sets its own part colours, so this only affects renders.
_ADDED_COLOR = (235, 170, 60, 255)


class AddPartError(ValueError):
    """A rejected add_part or probe_surface request; the message tells the model what to fix."""


@dataclass
class PartAddedSignal:
    """Returned by add_part so the loop highlights the new part and gives the model `result`."""

    part_id: str
    result: dict = field(default_factory=dict)


# ---------------------------------------------------------------- schemas
_VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def input_schema() -> dict:
    """The add_part input schema (Anthropic tool form)."""
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Short name for the new part."},
            "kind": {"type": "string", "enum": list(KINDS), "description": (
                "The solid. Each is built in its own frame: width along width_axis, depth across it, "
                "height along up, base at the bottom. box: size_mm [w, d, h]. cylinder: [dx, dy, h] "
                "(round when dx = dy). cone: base [w, d], height h; pointed unless top_scale > 0. "
                "pyramid: sides 3 or 4, base [w, d], height h; pointed unless top_scale > 0. sphere: "
                "[w, d, h], an ellipsoid when unequal. torus: a ring around up, [outer w, outer d, "
                "tube thickness]. capsule: [diameter, diameter, total length] along up. triangular_prism: "
                "a triangle [w, h] standing on its base edge, extruded through d. cat_ear: a sculpted "
                "rounded-triangle ear with a raised rim, [w, d, h]. profile: any flat silhouette, "
                "outline_mm extruded through thickness_mm along the depth. headphone_ear_left / "
                "headphone_ear_right: the supplied curved headphone-ear meshes at their own size "
                "(size_mm optional). cad: the model of a CAD result (cad_id), placed where its preview "
                "was shown, or snapped with surface.")},
            "size_mm": {**_VEC3, "description": (
                "[width, depth, height] in mm, as described for each kind; not used for profile or "
                f"cad. Each value {MIN_SIZE_MM}-{MAX_SIZE_MM:g} mm.")},
            "top_scale": {"type": "number", "minimum": 0, "maximum": 1, "description": (
                "cone and pyramid: size of the top as a fraction of the base. 0 (default) is pointed; "
                "for example 0.5 makes a frustum.")},
            "tip_radius_mm": {"type": "number", "minimum": 0, "description": (
                "Pointed cone only: round the tip with this radius in mm.")},
            "sides": {"type": "integer", "enum": [3, 4], "description": (
                "pyramid: 3 or 4 sides (default 4). A 3-sided base has one edge along the width and "
                "its corner toward +depth.")},
            "outline_mm": {"type": "array", "minItems": MIN_OUTLINE_POINTS,
                           "maxItems": MAX_OUTLINE_POINTS,
                           "items": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                     "maxItems": 2},
                           "description": (
                "profile: 3-64 [u, v] points in mm going once around the silhouette, without "
                "crossings; u runs along width_axis and v along up, v = 0 at the base. Concave "
                "outlines are fine. The outline is centred on u = 0 and its lowest point sits on the "
                "base.")},
            "thickness_mm": {"type": "number", "description": (
                "profile: thickness in mm along the depth axis (extruded symmetrically).")},
            "on_part": {"type": "string", "description": (
                "Placement, best for anything that attaches: the id of an existing part to stand the "
                "new part on. With direction it finds that part's outer surface, puts the new part's "
                "base there with up along the outward surface normal, and sinks the base embed_mm in. "
                "It also sets attach_to (relation default fastened).")},
            "direction": {**_VEC3, "description": (
                "With on_part: which way to look from on_part's centre, [x, y, z] in the scan frame, "
                "for example toward the side the new part should stand out of. The outermost surface "
                "in that direction is used.")},
            "surface": {"type": "string", "enum": list(_FACES), "description": (
                "Placement: snap onto this scan-axis face of attach_to (top = +Z, bottom = -Z, right = "
                "+X, left = -X, back = +Y, front = -Y). The outermost point of the attach_to mesh on "
                "that face is found and the base goes there, with up pointing out of the face. Needs "
                "attach_to. Also works for kind=cad (the model keeps its preview rotation).")},
            "offset_uv": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"},
                          "description": (
                "With surface: position across the face, -1 to 1 on each of the two remaining scan axes "
                "in XYZ order. Default [0, 0] is the middle.")},
            "embed_mm": {"type": "number", "minimum": 0, "description": (
                f"With on_part: how far to sink the base into the surface; default the smaller of "
                f"{DEFAULT_EMBED_MM:g} mm and 10% of the height.")},
            "base_mm": {**_VEC3, "description": (
                "Placement: the point [x, y, z] in mm (scan frame) where the centre of the part's base "
                "goes.")},
            "center_mm": {**_VEC3, "description": "Placement: the point where the part's centre goes."},
            "up": {**_VEC3, "description": (
                "Where the part's height axis points, [x, y, z]. Default: the outward surface normal "
                "with on_part, otherwise [0, 0, 1].")},
            "width_axis": {**_VEC3, "description": (
                "Where the part's width points; made perpendicular to up. Default: the horizontal "
                "direction cross([0, 0, 1], up), or [1, 0, 0] when up is vertical. The depth axis is "
                "cross(up, width_axis).")},
            "cad_id": {"type": "string", "description": (
                "Required for kind=cad: the cad_id of a CAD result for this workspace. The part lands "
                "where that CAD preview was shown; the size and placement arguments are not used.")},
            "description": {"type": "string", "description": "Optional one-sentence description."},
            "material": {"type": "string", "description": "Optional; only when the user named the "
                                                          "material, e.g. PLA or steel."},
            "attach_to": {"type": "string", "description": (
                "Optional id of an existing part the new part touches or mounts on, when not using "
                "on_part (required with surface); records a relation in the scene graph and checks "
                "the contact.")},
            "relation": {"type": "string", "enum": list(RELATION_TYPES), "description": (
                "Relation from the new part to attach_to or on_part; default fastened with on_part, "
                "rests_on with attach_to.")},
        },
        "required": ["label", "kind"],
    }


def probe_input_schema() -> dict:
    """The probe_surface input schema (Anthropic tool form)."""
    return {
        "type": "object",
        "properties": {
            "part_id": {"type": "string", "description": "The existing part to probe."},
            "direction": {**_VEC3, "description": (
                "Which way to look from the part's centre, [x, y, z] in the scan frame. The outermost "
                "surface in that direction is reported.")},
        },
        "required": ["part_id", "direction"],
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


def _direction(value: object, name: str) -> np.ndarray:
    """A unit direction from 3 finite numbers that are not all zero."""
    vec = np.asarray(_vector(value, name, 3), dtype=float)
    norm = float(np.linalg.norm(vec))
    if not norm > 1e-9 or not math.isfinite(norm):
        raise AddPartError(f"{name} must be a direction [x, y, z] that is not all zeros")
    return vec / norm


def _point(value: object, name: str) -> np.ndarray:
    point = _vector(value, name, 3)
    if any(abs(c) > MAX_COORD_MM for c in point):
        raise AddPartError(f"{name} must be within ±{MAX_COORD_MM:g} mm of the scan origin")
    return np.asarray(point, dtype=float)


def _length(value: object, name: str, notes: list[str]) -> float:
    """A positive length in mm, clamped to the allowed range (with a note when clamped)."""
    number = _number(value, name)
    if number <= 0:
        raise AddPartError(f"{name} must be positive (mm)")
    clamped = min(max(number, MIN_SIZE_MM), MAX_SIZE_MM)
    if clamped != number:
        notes.append(f"{name} clamped to {fmt_mm(clamped)} mm (allowed {MIN_SIZE_MM}-{MAX_SIZE_MM:g} mm)")
    return clamped


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


def _existing_part(value: object, name: str, part_ids: set[str]) -> str:
    part_id = _identifier(value, name)
    if part_id not in part_ids:
        known = ", ".join(sorted(part_ids)[:_LISTED_IDS])
        raise AddPartError(f"{name} {part_id!r} is not a part of this object; parts: {known}")
    return part_id


@dataclass
class _Request:
    label: str
    kind: str
    size: list[float] | None = None  # [width, depth, height]: local X, Y, Z extents of the solid
    top_scale: float = 0.0
    tip_radius: float = 0.0
    sides: int = 4
    outline: np.ndarray | None = None  # profile: (n, 2) [u, v], counter-clockwise, re-centred
    placement: str = "cad"  # on_part | base_mm | center_mm | cad
    on_part: str | None = None
    direction: np.ndarray | None = None
    surface: str | None = None
    offset_uv: list[float] | None = None
    embed: float | None = None
    point: np.ndarray | None = None
    up: np.ndarray | None = None
    width_axis: np.ndarray | None = None
    cad_id: str | None = None
    description: str | None = None
    material: str | None = None
    attach_to: str | None = None
    relation: str | None = None
    notes: list[str] = field(default_factory=list)


def _parse(args: dict, part_ids: set[str]) -> _Request:
    if not isinstance(args, dict):
        raise AddPartError("add_part needs an object of arguments")
    if "rotation_quat" in args:
        raise AddPartError("rotation_quat is no longer accepted: give up (where the part's height "
                           "points) and, when it matters, width_axis")
    unknown = sorted(set(args) - _ALLOWED_ARGS)
    if unknown:
        raise AddPartError(f"unknown argument(s) {unknown}; allowed: {sorted(_ALLOWED_ARGS)}")
    label = _text(args.get("label"), "label", _LABEL_MAX)
    if not label:
        raise AddPartError("label is required")
    kind = args.get("kind")
    if kind not in KINDS:
        raise AddPartError(f"kind must be one of {list(KINDS)}")
    for name, kinds in _SHAPE_ARGS.items():
        if args.get(name) is not None and kind not in kinds:
            raise AddPartError(f"{name} is only used with kind={' or '.join(kinds)}")
    req = _Request(label=label, kind=kind,
                   description=_text(args.get("description"), "description", _DESCRIPTION_MAX),
                   material=_text(args.get("material"), "material", _MATERIAL_MAX))
    if kind == "cad":
        _parse_cad(args, req, part_ids)
    else:
        if args.get("cad_id") is not None:
            raise AddPartError("cad_id is only used with kind=cad")
        _parse_shape(args, req)
        _parse_placement(args, req, part_ids)
    _parse_relation(args, req, part_ids)
    return req


def _parse_cad(args: dict, req: _Request, part_ids: set[str]) -> None:
    req.cad_id = _identifier(args.get("cad_id"), "cad_id")
    if args.get("surface") is not None:
        _parse_surface(args, req, part_ids)
    elif args.get("offset_uv") is not None:
        raise AddPartError("offset_uv needs surface")
    ignored = [k for k in ("size_mm", "base_mm", "center_mm", "direction", "embed_mm", "up", "width_axis")
               if args.get(k) is not None]
    if ignored:
        where = ("snapped to the surface with its preview rotation" if req.surface
                 else "where the CAD preview was shown")
        req.notes.append(f"{', '.join(ignored)} not used for kind=cad: the part sits {where}")
    if args.get("on_part") is not None:
        req.on_part = _existing_part(args["on_part"], "on_part", part_ids)
        req.notes.append("on_part only records the relation for kind=cad; the part sits where the CAD "
                         "preview was shown")


def _parse_shape(args: dict, req: _Request) -> None:
    notes, kind = req.notes, req.kind
    if kind == "profile":
        if args.get("size_mm") is not None:
            notes.append("size_mm not used for kind=profile: outline_mm and thickness_mm set its size")
        if args.get("outline_mm") is None or args.get("thickness_mm") is None:
            raise AddPartError("kind=profile needs outline_mm (the [u, v] points of the silhouette) and "
                               "thickness_mm")
        req.outline = _outline(args["outline_mm"], notes)
        thickness = _length(args["thickness_mm"], "thickness_mm", notes)
        span = req.outline.max(axis=0) - req.outline.min(axis=0)
        req.size = [float(span[0]), thickness, float(span[1])]
        return
    if kind in HEADPHONE_EARS and args.get("size_mm") is None:
        req.size = [float(v) for v in _ear_asset(kind).extents]
        return
    if args.get("size_mm") is None:
        raise AddPartError(f"kind={kind} needs size_mm [width, depth, height]")
    size = _vector(args["size_mm"], "size_mm", 3)
    if any(v <= 0 for v in size):
        raise AddPartError("size_mm values must be positive (mm)")
    clamped = [min(max(v, MIN_SIZE_MM), MAX_SIZE_MM) for v in size]
    if clamped != size:
        notes.append(f"size_mm clamped to {fmt_vec(clamped)} (allowed {MIN_SIZE_MM}-{MAX_SIZE_MM:g} mm)")
    w, d, h = clamped
    if kind in ("cone", "pyramid") and args.get("top_scale") is not None:
        req.top_scale = _number(args["top_scale"], "top_scale")
        if not 0.0 <= req.top_scale <= 1.0:
            raise AddPartError("top_scale must be between 0 (pointed) and 1")
    if kind == "cone" and args.get("tip_radius_mm") is not None:
        tip = _number(args["tip_radius_mm"], "tip_radius_mm")
        if tip < 0:
            raise AddPartError("tip_radius_mm must not be negative")
        if tip > 0 and req.top_scale > 0:
            raise AddPartError("tip_radius_mm only rounds a pointed cone (top_scale 0)")
        limit = min(0.45 * w, 0.4 * h)
        if tip > limit:
            notes.append(f"tip_radius_mm clamped to {fmt_mm(limit)} mm (at most 90% of the base radius "
                         "and 40% of the height)")
            tip = limit
        req.tip_radius = tip
    if kind == "pyramid" and args.get("sides") is not None:
        sides = _number(args["sides"], "sides")
        if sides not in (3.0, 4.0):
            raise AddPartError("sides must be 3 or 4")
        req.sides = int(sides)
    if kind == "torus":
        if not (h < w / 2 and h < d / 2):
            raise AddPartError("torus: the tube thickness (third value) must be less than half the outer "
                               "width and depth, or the ring has no hole")
        tube = h / 2
        a, b = w / 2 - tube, d / 2 - tube
        if min(a * a / b, b * b / a) <= tube:
            raise AddPartError("torus: the ring is too oval for that tube thickness; make the outer width "
                               "and depth closer or the tube thinner")
    if kind == "capsule" and h < max(w, d):
        notes.append(f"capsule length raised to {fmt_mm(max(w, d))} mm, its diameter (a shorter capsule "
                     "is a sphere)")
        h = max(w, d)
    req.size = [w, d, h]


def _parse_placement(args: dict, req: _Request, part_ids: set[str]) -> None:
    given = [k for k in _PLACEMENTS if args.get(k) is not None]
    if len(given) != 1:
        found = f" (got {', '.join(given)})" if given else ""
        raise AddPartError("give exactly one placement: on_part with direction (stand it on that part's "
                           f"surface), surface with attach_to, base_mm or center_mm{found}")
    req.placement = given[0]
    if req.placement != "surface" and args.get("offset_uv") is not None:
        raise AddPartError("offset_uv needs surface")
    if req.placement == "surface":
        _parse_surface(args, req, part_ids)
        if args.get("direction") is not None or args.get("embed_mm") is not None:
            raise AddPartError("direction and embed_mm are only used with on_part")
    elif req.placement == "on_part":
        req.on_part = _existing_part(args["on_part"], "on_part", part_ids)
        if args.get("direction") is None:
            raise AddPartError("on_part needs direction: which way from that part's centre to put the new "
                               "part, e.g. [0, 0, 1] for its top")
        req.direction = _direction(args["direction"], "direction")
        if args.get("embed_mm") is not None:
            embed = _number(args["embed_mm"], "embed_mm")
            if embed < 0:
                raise AddPartError("embed_mm must not be negative")
            limit = 0.9 * req.size[2]
            if embed > limit:
                req.notes.append(f"embed_mm clamped to {fmt_mm(limit)} mm (90% of the height)")
                embed = limit
            req.embed = embed
    else:
        for name in ("direction", "embed_mm"):
            if args.get(name) is not None:
                raise AddPartError(f"{name} is only used with on_part")
        req.point = _point(args[req.placement], req.placement)
    if args.get("up") is not None:
        req.up = _direction(args["up"], "up")
    if args.get("width_axis") is not None:
        req.width_axis = _direction(args["width_axis"], "width_axis")
    # Fail early on a width_axis parallel to an explicit up (on_part's default up is known later).
    if req.up is not None:
        _frame(req.up, req.width_axis)


def _parse_surface(args: dict, req: _Request, part_ids: set[str]) -> None:
    face = args.get("surface")
    if not isinstance(face, str) or face not in _FACES:
        raise AddPartError(f"surface must be one of {list(_FACES)}")
    if args.get("attach_to") is None:
        raise AddPartError("surface needs attach_to: the part whose face the new part snaps onto")
    _existing_part(args["attach_to"], "attach_to", part_ids)
    uv = _vector(args["offset_uv"], "offset_uv", 2) if args.get("offset_uv") is not None else [0.0, 0.0]
    if any(abs(v) > 1 for v in uv):
        raise AddPartError("offset_uv values must lie between -1 and 1")
    req.surface, req.offset_uv = face, uv


def _parse_relation(args: dict, req: _Request, part_ids: set[str]) -> None:
    attach_to = None
    if args.get("attach_to") is not None:
        attach_to = _existing_part(args["attach_to"], "attach_to", part_ids)
    if req.on_part is not None:
        if attach_to is not None and attach_to != req.on_part:
            raise AddPartError("attach_to must be the same part as on_part (on_part already attaches the "
                               "new part to it); leave attach_to out")
        attach_to = req.on_part
    if attach_to is None:
        if args.get("relation") is not None:
            raise AddPartError("relation needs attach_to or on_part (the existing part it relates to)")
        return
    relation = args.get("relation") or ("fastened" if req.on_part is not None else "rests_on")
    if relation not in RELATION_TYPES:
        raise AddPartError(f"relation must be one of {list(RELATION_TYPES)}")
    req.attach_to, req.relation = attach_to, relation


# ---------------------------------------------------------------- outlines (profile)
def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """z of (a - o) x (b - o): positive when o, a, b turn counter-clockwise."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _segments_meet(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float) -> bool:
    """Whether closed segments ab and cd share any point (crossing, touching or overlapping)."""
    d1, d2 = _cross2(c, d, a), _cross2(c, d, b)
    d3, d4 = _cross2(a, b, c), _cross2(a, b, d)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
            ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return ((abs(d1) <= eps and _on_segment(c, d, a, eps)) or (abs(d2) <= eps and _on_segment(c, d, b, eps))
            or (abs(d3) <= eps and _on_segment(a, b, c, eps)) or (abs(d4) <= eps and _on_segment(a, b, d, eps)))


def _outline(value: object, notes: list[str]) -> np.ndarray:
    """The profile outline as a simple counter-clockwise polygon, centred on u = 0 with its lowest
    point on v = 0. Repeated and straight-through points are dropped; crossings are rejected."""
    if not isinstance(value, list | tuple):
        raise AddPartError("outline_mm must be a list of [u, v] points")
    if not MIN_OUTLINE_POINTS <= len(value) <= MAX_OUTLINE_POINTS:
        raise AddPartError(f"outline_mm needs {MIN_OUTLINE_POINTS}-{MAX_OUTLINE_POINTS} points; it has "
                           f"{len(value)}")
    raw = np.asarray([_vector(p, f"outline_mm[{i}]", 2) for i, p in enumerate(value)], dtype=float)
    if np.abs(raw).max() > MAX_SIZE_MM:
        raise AddPartError(f"outline_mm points must be within ±{MAX_SIZE_MM:g} mm")
    span = float(np.ptp(raw, axis=0).max())
    if span < MIN_SIZE_MM:
        raise AddPartError("outline_mm points are all in one place")
    tol = span * 1e-6
    eps = span * span * 1e-9
    points: list[np.ndarray] = []
    for p in raw:
        if not points or np.linalg.norm(p - points[-1]) > tol:
            points.append(p)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tol:
        points.pop()  # the outline was closed by repeating its first point
    changed = True
    while changed and len(points) >= 3:
        changed = False
        for i in range(len(points)):
            a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
            if abs(_cross2(a, b, c)) <= eps:
                if float(np.dot(b - a, c - b)) < 0:
                    raise AddPartError("outline_mm doubles back on itself (three points in a line that turn "
                                       "back); give the points once around the silhouette")
                points.pop(i)  # a point on a straight edge adds nothing
                changed = True
                break
    if len(points) < 3:
        raise AddPartError("outline_mm has no area: its points are in a line")
    poly = np.asarray(points)
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbouring edges share a point
            if _segments_meet(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n], eps):
                raise AddPartError(
                    f"outline_mm crosses itself (the edge from point {i} meets the edge from point {j}); "
                    "give the points in order once around the silhouette, without crossings")
    area2 = float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(np.roll(poly[:, 0], -1), poly[:, 1]))
    if area2 < 0:
        poly = poly[::-1]
    span_u, span_v = np.ptp(poly, axis=0)
    if span_u < MIN_SIZE_MM or span_v < MIN_SIZE_MM:
        raise AddPartError(f"outline_mm must be at least {MIN_SIZE_MM} mm wide and tall")
    shift = np.array([-(poly[:, 0].min() + poly[:, 0].max()) / 2.0, -poly[:, 1].min()])
    if np.abs(shift).max() > 0.05:
        notes.append(f"outline_mm shifted by [{fmt_mm(shift[0])}, {fmt_mm(shift[1])}] mm so it is centred "
                     "on u = 0 and its lowest point is the base")
    return poly + shift


def ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple counter-clockwise polygon (convex or concave) by ear clipping. Returns
    counter-clockwise index triples; raises AddPartError if no ear can be found."""
    poly = np.asarray(poly, dtype=float)
    span = float(np.ptp(poly, axis=0).max()) or 1.0
    eps = span * span * 1e-10
    remaining = list(range(len(poly)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        count = len(remaining)
        for k in range(count):
            i, j, m = remaining[k - 1], remaining[k], remaining[(k + 1) % count]
            a, b, c = poly[i], poly[j], poly[m]
            if _cross2(a, b, c) <= eps:
                continue  # reflex (or flat) corner: not an ear
            others = poly[[r for r in remaining if r not in (i, j, m)]]
            if len(others):
                inside = ((b[0] - a[0]) * (others[:, 1] - a[1]) - (b[1] - a[1]) * (others[:, 0] - a[0]) >= -eps) \
                    & ((c[0] - b[0]) * (others[:, 1] - b[1]) - (c[1] - b[1]) * (others[:, 0] - b[0]) >= -eps) \
                    & ((a[0] - c[0]) * (others[:, 1] - c[1]) - (a[1] - c[1]) * (others[:, 0] - c[0]) >= -eps)
                if inside.any():
                    continue
            triangles.append((i, j, m))
            remaining.pop(k)
            break
        else:
            raise AddPartError("outline_mm could not be triangulated; simplify the outline")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


# ---------------------------------------------------------------- solids (local frame)
def _solid(req: _Request):
    """The solid in its own frame: width along X, depth along Y, height along Z, base centred on the
    origin (z from 0 up)."""
    import trimesh

    w, d, h = req.size
    if req.kind == "box":
        mesh = trimesh.creation.box(extents=[w, d, h])
    elif req.kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=_SECTIONS)
        mesh.apply_scale([w, d, h])
    elif req.kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
        mesh.apply_scale([w, d, h])
    elif req.kind == "capsule":
        diameter = max(w, d)
        mesh = trimesh.creation.capsule(height=max(h / diameter - 1.0, 0.0), radius=0.5)
        mesh.apply_scale([w, d, diameter])
    elif req.kind == "cone":
        return _cone(w, d, h, req.top_scale, req.tip_radius)
    elif req.kind == "pyramid":
        return _pyramid(w, d, h, req.sides, req.top_scale)
    elif req.kind == "torus":
        return _torus(w, d, h)
    elif req.kind == "triangular_prism":
        mesh = trimesh.convex.convex_hull([[x, y, z] for y in (-d / 2, d / 2)
                                           for x, z in ((-w / 2, 0.0), (w / 2, 0.0), (0.0, h))])
        return mesh
    elif req.kind == "cat_ear":
        return _on_base(_cat_ear(), [w, d, h])
    elif req.kind in HEADPHONE_EARS:
        mesh = _ear_asset(req.kind)
        return _on_base(mesh, [w, d, h])
    else:
        return _extrusion(req.outline, d)
    mesh.apply_translation([0.0, 0.0, h / 2.0])
    return mesh


def _on_base(mesh, size: list[float]):
    """Scale mesh to size along X, Y, Z (unless it is already that size) and stand it on the origin:
    x and y centred, z from 0 up."""
    extents = np.asarray(mesh.extents, dtype=float)
    if not np.allclose(extents, size, rtol=1e-6, atol=1e-6):
        mesh.apply_scale(np.asarray(size, dtype=float) / extents)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    return mesh


def _ear_asset(kind: str):
    """One of the supplied curved headphone-ear meshes (mm), as authored."""
    import trimesh

    filename = "left_ear_curved.stl" if kind == "headphone_ear_left" else "right_ear_curved.stl"
    asset = files("screwit.agent").joinpath("assets", filename)
    return trimesh.load(io.BytesIO(asset.read_bytes()), file_type="stl", force="mesh")


def _cat_ear():
    """A rounded triangular shell with a raised rim and a recessed front panel, about a unit in size:
    the triangle spans X (width) and Z (height), the shell's depth runs along Y. Rings share vertices
    at every seam, including the floor of the recess, so it is closed."""
    import trimesh

    corners = np.array([[-.5, -.5], [.5, -.5], [0, .5]])
    outline = []
    for i, corner in enumerate(corners):
        start = corner + .08 * (corners[i - 1] - corner)
        end = corner + .08 * (corners[(i + 1) % 3] - corner)
        for t in np.linspace(0, 1, 9):
            outline.append((1 - t) ** 2 * start + 2 * t * (1 - t) * corner + t * t * end)
    x, z = np.asarray(outline).T
    rings = [(.88, .5), (.98, .4), (1, .25), (1, -.30), (.96, -.49), (.87, -.50), (.78, -.43), (.69, -.22)]
    vertices = np.concatenate([np.column_stack((x * k, np.full_like(x, y), z * k)) for k, y in rings])
    n = len(x)
    faces = []
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            faces.extend([(a, b, b + n), (a, b + n, a + n)])
    last = (len(rings) - 1) * n
    for i in range(1, n - 1):
        faces.extend([(0, i + 1, i), (last, last + i, last + i + 1)])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def _cone(w: float, d: float, h: float, top_scale: float, tip_radius: float):
    import trimesh

    radius = w / 2.0
    if tip_radius > 0:
        profile = _rounded_tip_profile(radius, h, tip_radius)
    elif top_scale > 0:
        profile = [(0.0, 0.0), (radius, 0.0), (radius * top_scale, h), (0.0, h)]
    else:
        profile = [(0.0, 0.0), (radius, 0.0), (0.0, h)]
    mesh = trimesh.creation.revolve(np.asarray(profile, dtype=float), sections=_SECTIONS)
    if not math.isclose(w, d):
        mesh.apply_scale([1.0, d / w, 1.0])
    return mesh


def _rounded_tip_profile(radius: float, h: float, tip: float) -> list[tuple[float, float]]:
    """(r, z) profile of a cone whose side is tangent to a tip sphere of radius `tip` with its top at h.
    The side runs from (radius, 0) toward a virtual apex at height H, solved so the sphere's top is h."""

    def top(apex: float) -> float:
        return apex - tip * (math.hypot(apex, radius) / radius - 1.0)

    lo, hi = h, h
    while top(hi) < h:
        hi = hi * 2.0 + tip
    for _ in range(80):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if top(mid) < h else (lo, mid)
    apex = (lo + hi) / 2.0
    slant = math.hypot(apex, radius)
    centre = apex - tip * slant / radius
    start = math.atan2(radius / slant, apex / slant)  # the side's outward normal, from the r axis
    arc = [(tip * math.cos(a), centre + tip * math.sin(a)) for a in np.linspace(start, math.pi / 2, 13)]
    arc[-1] = (0.0, h)
    return [(0.0, 0.0), (radius, 0.0), *arc]


def _pyramid(w: float, d: float, h: float, sides: int, top_scale: float):
    import trimesh

    if sides == 4:
        base = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
        middle = np.array([0.0, 0.0])
    else:
        base = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [0.0, d / 2]])
        middle = base.mean(axis=0)
    n = len(base)
    vertices = [[x, y, 0.0] for x, y in base]
    faces = [[0, k + 1, k] for k in range(1, n - 1)]  # bottom, facing down
    if top_scale > 0:
        top = middle + (base - middle) * top_scale
        vertices += [[x, y, h] for x, y in top]
        faces += [[n, n + k, n + k + 1] for k in range(1, n - 1)]  # top, facing up
        for i in range(n):
            j = (i + 1) % n
            faces += [[i, j, n + j], [i, n + j, n + i]]
    else:
        vertices.append([middle[0], middle[1], h])
        faces += [[i, (i + 1) % n, n] for i in range(n)]
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)


def _torus(w: float, d: float, thickness: float, major: int = _SECTIONS, minor: int = 24):
    """A ring in the XY plane whose outer size is w x d and whose tube is `thickness` across; its
    bottom is on z = 0. The tube follows an ellipse, so a ring w != d keeps a round tube."""
    import trimesh

    tube = thickness / 2.0
    a, b = w / 2.0 - tube, d / 2.0 - tube
    phi = np.linspace(0.0, 2.0 * math.pi, major, endpoint=False)
    theta = np.linspace(0.0, 2.0 * math.pi, minor, endpoint=False)
    normal = np.column_stack([b * np.cos(phi), a * np.sin(phi)])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    ring_x = a * np.cos(phi)[:, None] + tube * np.cos(theta)[None, :] * normal[:, :1]
    ring_y = b * np.sin(phi)[:, None] + tube * np.cos(theta)[None, :] * normal[:, 1:]
    ring_z = np.broadcast_to(tube + tube * np.sin(theta)[None, :], ring_x.shape)
    vertices = np.column_stack([ring_x.ravel(), ring_y.ravel(), ring_z.ravel()])
    i, j = np.meshgrid(np.arange(major), np.arange(minor), indexing="ij")
    v00 = i * minor + j
    v10 = ((i + 1) % major) * minor + j
    v11 = ((i + 1) % major) * minor + (j + 1) % minor
    v01 = i * minor + (j + 1) % minor
    faces = np.concatenate([np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
                            np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _extrusion(outline: np.ndarray, thickness: float):
    """The profile: a counter-clockwise [u, v] outline in the XZ plane (u along X, v along Z),
    extruded from y = -thickness/2 to +thickness/2."""
    import trimesh

    n = len(outline)
    half = thickness / 2.0
    front = np.column_stack([outline[:, 0], np.full(n, -half), outline[:, 1]])
    back = np.column_stack([outline[:, 0], np.full(n, half), outline[:, 1]])
    faces: list[tuple[int, int, int]] = []
    for i, j, k in ear_clip(outline):
        faces.append((i, j, k))  # counter-clockwise in (u, v) faces -Y
        faces.append((n + i, n + k, n + j))  # reversed faces +Y
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, n + j, j))
        faces.append((i, n + i, n + j))
    return trimesh.Trimesh(vertices=np.vstack([front, back]), faces=np.asarray(faces), process=True)


# ---------------------------------------------------------------- frames and surfaces
def _frame(up: np.ndarray, width_axis: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(width, depth, up) unit axes, right-handed. width_axis is made perpendicular to up; by default
    it is the horizontal tangent cross(Z, up), or [1, 0, 0] when up is (nearly) vertical."""
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    if width_axis is None:
        tangent = np.cross(_Z, up)
        width_axis = tangent if np.linalg.norm(tangent) >= _NEAR_VERTICAL else np.array([1.0, 0.0, 0.0])
    width = np.asarray(width_axis, dtype=float) - float(np.dot(width_axis, up)) * up
    if np.linalg.norm(width) < 1e-3:
        raise AddPartError("width_axis must not be parallel to up")
    width /= np.linalg.norm(width)
    return width, np.cross(up, width), up


def _quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """A contract quaternion [x, y, z, w] (w >= 0) for a 3x3 rotation; trimesh uses [w, x, y, z]."""
    import trimesh

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    w, x, y, z = trimesh.transformations.quaternion_from_matrix(matrix)
    quat = np.array([x, y, z, w], dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[3] < 0:
        quat = -quat
    return tuple(float(c) for c in quat)


def _matrix(quat_xyzw: tuple[float, float, float, float], translate) -> np.ndarray:
    """4x4 transform: rotate by a contract quaternion [x, y, z, w], then translate (mm)."""
    import trimesh

    x, y, z, w = quat_xyzw
    matrix = trimesh.transformations.quaternion_matrix([w, x, y, z])  # trimesh order is [w, x, y, z]
    matrix[:3, 3] = [float(v) for v in translate]
    return matrix


@dataclass
class SurfaceHit:
    """Where a look from a part's centre meets its outer surface."""

    point: np.ndarray
    normal: np.ndarray  # outward
    found_by: str  # "ray": outermost ray hit; "nearest": the ray missed, nearest surface point used
    origin: np.ndarray
    direction: np.ndarray
    wall_mm: float | None  # material the ray crossed just before leaving, when it went in and out


def surface_hit(mesh, origin: np.ndarray, direction: np.ndarray, reach: float, radius: float) -> SurfaceHit:
    """Cast a ray from origin along direction and take the OUTERMOST hit (trimesh's own ray code, no
    embree). If the ray misses, use the surface point nearest to origin + direction * reach. The normal
    is averaged over the faces within `radius` and points outward (away from where the ray came from)."""
    import trimesh
    from trimesh.ray.ray_triangle import RayMeshIntersector

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    locations, _, triangles = RayMeshIntersector(mesh).intersects_location(
        origin[None, :], direction[None, :], multiple_hits=True)
    wall = None
    if len(locations):
        t = (locations - origin) @ direction
        k = int(np.argmax(t))
        point, triangle, outward, found_by = locations[k], int(triangles[k]), direction, "ray"
        distinct = np.unique(np.round(t, 3))
        if len(distinct) >= 2:
            wall = float(distinct[-1] - distinct[-2])
    else:
        target = origin + direction * reach
        closest, _, tris = trimesh.proximity.closest_point(mesh, target[None, :])
        point, triangle, found_by = closest[0], int(tris[0]), "nearest"
        away = target - point
        outward = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-9 else direction
    return SurfaceHit(point=np.asarray(point, dtype=float),
                      normal=_surface_normal(mesh, point, triangle, radius, outward),
                      found_by=found_by, origin=origin, direction=direction, wall_mm=wall)


def _surface_normal(mesh, point: np.ndarray, triangle: int, radius: float, outward: np.ndarray) -> np.ndarray:
    """Area-weighted mean normal of the faces within radius of point (scan meshes are noisy), each
    turned to agree with the hit face, flipped if it points back toward where the look came from."""
    normals = np.asarray(mesh.face_normals)
    ref = normals[triangle]
    if float(ref @ outward) < 0:
        ref = -ref
    near = np.flatnonzero(np.linalg.norm(np.asarray(mesh.triangles_center) - point, axis=1) <= radius)
    if len(near):
        picked = normals[near]
        signs = np.where(picked @ ref < 0, -1.0, 1.0)
        summed = (picked * (np.asarray(mesh.area_faces)[near] * signs)[:, None]).sum(axis=0)
        if np.linalg.norm(summed) > 1e-9:
            ref = summed / np.linalg.norm(summed)
    if float(ref @ outward) < 0:
        ref = -ref
    return ref / np.linalg.norm(ref)


def attachment_mesh(reg: Registry, anchor: Part):
    """The mesh of an existing part (scan frame, mm), from storage or, for an example, the fixtures."""
    import trimesh

    from screwit.app.registry import fixture_asset

    try:
        if reg.storage.head(anchor.mesh_uri) is None:
            fallback = fixture_asset(reg, anchor.mesh_uri)
            if fallback is None:
                raise AddPartError(f"the mesh of part {anchor.part_id!r} is missing")
            data = fallback[0]
        else:
            data = reg.storage.get_bytes(anchor.mesh_uri)
        return trimesh.load(io.BytesIO(data), file_type="glb", force="scene").to_geometry()
    except AddPartError:
        raise
    except Exception as exc:  # noqa: BLE001 - an unreadable mesh is a request-level failure
        raise AddPartError(f"the mesh of part {anchor.part_id!r} could not be read") from exc


def _surface_point(reg: Registry, anchor: Part, face: str, uv) -> tuple[np.ndarray, int, int]:
    """Where a ray along a scan axis, aimed at the face of anchor's bounding box at offset uv (-1..1 on
    the two other axes, XYZ order), meets the anchor's actual triangles: the outermost hit. Falls back
    to the nearest projected triangle edge when the ray passes through a hole. Returns (point, axis,
    sign)."""
    if not isinstance(face, str) or face not in _FACES:
        raise AddPartError(f"surface must be one of {list(_FACES)}")
    uv = np.asarray(_vector(uv, "offset_uv", 2))
    if np.any(np.abs(uv) > 1):
        raise AddPartError("offset_uv values must lie between -1 and 1")
    mesh = attachment_mesh(reg, anchor)
    triangles = mesh.triangles
    lo, hi = mesh.bounds
    axis, sign = _FACES[face]
    plane = [i for i in range(3) if i != axis]
    target = (lo + hi) / 2
    target[plane] += uv * (hi - lo)[plane] / 2
    a, b, c = triangles.transpose(1, 0, 2)
    u, v = (b - a)[:, plane], (c - a)[:, plane]
    w = target[plane] - a[:, plane]

    def cross(x, y):
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    det = cross(u, v)
    valid = np.abs(det) > 1e-12
    det = np.where(valid, det, 1)
    i, j = cross(w, v) / det, cross(u, w) / det
    hit = valid & (i >= -1e-8) & (j >= -1e-8) & (i + j <= 1 + 1e-8)
    if np.any(hit):
        depths = (a[:, axis] + i * (b - a)[:, axis] + j * (c - a)[:, axis])[hit]
        target[axis] = sign * np.max(sign * depths)
    else:
        # A hole or an open outline: use the closest projected triangle edge.
        ends = np.roll(triangles, -1, axis=1)
        edges = ends - triangles
        delta = target[plane] - triangles[:, :, plane]
        flat = edges[:, :, plane]
        t = np.clip((delta * flat).sum(2) / np.maximum((flat * flat).sum(2), 1e-20), 0, 1)
        points = (triangles + t[:, :, None] * edges).reshape(-1, 3)
        distances = np.linalg.norm(points[:, plane] - target[plane], axis=1)
        nearest = points[distances <= distances.min() + 1e-7]
        target = nearest[np.argmax(sign * nearest[:, axis])].copy()
    return target, axis, sign


def _face_normal(face: str) -> np.ndarray:
    axis, sign = _FACES[face]
    normal = np.zeros(3)
    normal[axis] = float(sign)
    return normal


def _part_mesh(reg: Registry, scan_id: str, part: Part):
    """A part's mesh (scan frame, mm), read the way render_view reads it."""
    import trimesh

    from screwit.app.fakes import FixtureStorage

    try:
        try:
            raw = reg.storage.get_bytes(part.mesh_uri)
        except FileNotFoundError:
            if part.mesh_uri not in {p.mesh_uri for p in reg.data.parts(scan_id)}:
                raise
            raw = FixtureStorage(reg.settings.fixtures_dir).get_bytes(part.mesh_uri)
        mesh = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene").to_geometry()
    except Exception:  # noqa: BLE001 - a missing or unreadable mesh is a request-level failure
        raise AddPartError(f"the mesh of part {part.part_id!r} could not be read") from None
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise AddPartError(f"part {part.part_id!r} has no surface to place on")
    return mesh


def _probe_radius(part: Part) -> float:
    return float(np.clip(0.03 * np.linalg.norm(part.obb.extent), 1.5, 8.0))


# ---------------------------------------------------------------- placement and contact
@dataclass
class _Placed:
    rotation: np.ndarray  # columns: width, depth, up
    base: np.ndarray
    embed: float | None
    hit: SurfaceHit | None
    target_mesh: object | None


def _place(req: _Request, reg: Registry, scan_id: str, parts: dict[str, Part], height: float) -> _Placed:
    hit = target = None
    embed = None
    if req.placement == "on_part":
        part = parts[req.on_part]
        target = _part_mesh(reg, scan_id, part)
        radius = float(np.clip(0.35 * min(req.size[0], req.size[1]), 1.5, 8.0))
        hit = surface_hit(target, np.asarray(part.obb.center, dtype=float), req.direction,
                          float(np.linalg.norm(part.obb.extent)), radius)
        up = req.up if req.up is not None else hit.normal
        width, depth, up = _frame(up, req.width_axis)
        embed = req.embed if req.embed is not None else min(DEFAULT_EMBED_MM, 0.1 * height)
        base = hit.point - up * embed
    elif req.placement == "surface":
        point, _, _ = _surface_point(reg, parts[req.attach_to], req.surface, req.offset_uv)
        width, depth, up = _frame(req.up if req.up is not None else _face_normal(req.surface), req.width_axis)
        base = point
    else:
        width, depth, up = _frame(req.up if req.up is not None else _Z, req.width_axis)
        base = req.point if req.placement == "base_mm" else req.point - up * (height / 2.0)
    return _Placed(rotation=np.column_stack([width, depth, up]), base=np.asarray(base, dtype=float),
                   embed=embed, hit=hit, target_mesh=target)


def contact_check(mesh, target, target_id: str, rim: np.ndarray, samples: int = 1500) -> dict:
    """How the new part meets `target`: the gap between their surfaces (0 when they cross), and how far
    the rim of the new part's base sits above (gap) or below (sunk) the target surface. Signs come from
    the target's face normals, which point outward on scan meshes."""
    import trimesh

    tol = max(CONTACT_TOLERANCE_MM, 0.01 * float(np.max(mesh.extents)))
    surface, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
    points = np.vstack([np.asarray(mesh.vertices), surface])
    closest, distance, triangles = trimesh.proximity.closest_point(target, points)
    signed = np.einsum("ij,ij->i", points - closest, np.asarray(target.face_normals)[triangles])
    inside = (signed < 0) & (distance > tol)
    outside = (signed > 0) & (distance > tol)
    crossing = bool(inside.any() and outside.any())
    gap = 0.0 if crossing else float(distance.min())
    buried = bool(inside.all())
    rim_closest, _, rim_triangles = trimesh.proximity.closest_point(target, rim)
    rim_signed = np.einsum("ij,ij->i", rim - rim_closest, np.asarray(target.face_normals)[rim_triangles])
    above, sunk = max(float(rim_signed.max()), 0.0), max(float(-rim_signed.min()), 0.0)
    touching = (crossing or gap <= tol) and not buried
    if buried:
        summary = (f"the new part is inside {target_id}, so it is hidden; put it on the surface with "
                   "on_part and direction")
    elif touching:
        summary = f"touches {target_id}"
        if sunk > tol:
            summary += f"; the base is sunk up to {fmt_mm(sunk)} mm into it"
        if above > tol:
            summary += f"; part of the base rim is up to {fmt_mm(above)} mm off its surface"
    else:
        summary = (f"does not touch {target_id}: a {fmt_mm(gap)} mm gap; move it onto the surface "
                   "(on_part and direction)")
    return {"part_id": target_id, "touching": touching, "gap_mm": round(gap, 2),
            "base_rim_gap_mm": round(above, 2), "base_sunk_mm": round(sunk, 2), "inside": buried,
            "summary": summary}


def _base_rim(local_mesh) -> np.ndarray:
    """The vertices on the bottom of a local-frame solid (z near its minimum)."""
    vertices = np.asarray(local_mesh.vertices)
    low, high = vertices[:, 2].min(), vertices[:, 2].max()
    return vertices[vertices[:, 2] <= low + max(1e-6, 0.02 * (high - low))]


def _round(values, digits: int = 2) -> list[float]:
    return [round(float(v), digits) + 0.0 for v in values]


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


# ---------------------------------------------------------------- the tools
def probe_surface(reg: Registry, store: ScanStore, scan_id: str, args: dict) -> dict:
    """Read-only: the outer surface point and outward normal of a part in a direction from its centre,
    found exactly as add_part's on_part placement finds it."""
    try:
        if not isinstance(args, dict):
            raise AddPartError("probe_surface needs an object of arguments")
        unknown = sorted(set(args) - _PROBE_ARGS)
        if unknown:
            raise AddPartError(f"unknown argument(s) {unknown}; probe_surface takes part_id and direction")
        parts = {p.part_id: p for p in store.list_parts(scan_id)}
        part = parts[_existing_part(args.get("part_id"), "part_id", set(parts))]
        if args.get("direction") is None:
            raise AddPartError("direction is required: which way to look from the part's centre")
        direction = _direction(args["direction"], "direction")
        hit = surface_hit(_part_mesh(reg, scan_id, part), np.asarray(part.obb.center, dtype=float),
                          direction, float(np.linalg.norm(part.obb.extent)), _probe_radius(part))
    except AddPartError as e:
        return {"error": "bad_request", "message": str(e)}
    width, _, _ = _frame(hit.normal, None)
    result = {
        "part_id": part.part_id, "direction": _round(direction, 3),
        "from_mm": _round(hit.origin, 1), "point_mm": _round(hit.point, 1), "normal": _round(hit.normal, 3),
        "found_by": ("outermost surface along the ray" if hit.found_by == "ray"
                     else "the ray missed; nearest surface point toward that direction"),
        "distance_mm": round(float(np.linalg.norm(hit.point - hit.origin)), 1),
        "default_width_axis": _round(width, 3),
        "message": (f"add_part with on_part={part.part_id!r} and this direction stands a part's base "
                    "here, its up along the normal."),
    }
    if hit.wall_mm is not None:
        result["wall_mm"] = round(hit.wall_mm, 1)
    return result


def add_part(reg: Registry, store: ScanStore, scan_id: str, args: dict, *,
             replace_part: Part | None = None) -> PartAddedSignal | dict:
    """Add a part, or with replace_part rebuild that existing part in place: same part id, a fresh
    mesh key (so viewers do not keep a cached copy) and the next measurement_revision."""
    parts = store.list_parts(scan_id)
    by_id = {p.part_id: p for p in parts}
    part_ids = [p.part_id for p in parts]
    placed = None
    try:
        req = _parse(args, set(part_ids))
        if req.kind == "cad":
            mesh, quat, translate, cad_status = _cad_mesh(reg, store, scan_id, req.cad_id)
            lo, hi = (np.asarray(b, dtype=float) for b in mesh.bounds)
            rim = _base_rim(mesh)
            matrix = _matrix(quat, translate)
            if req.surface is not None:
                # Keep the preview rotation; move the model so its face toward the surface touches it.
                point, axis, sign = _surface_point(reg, by_id[req.attach_to], req.surface, req.offset_uv)
                rotated = np.asarray(mesh.vertices) @ matrix[:3, :3].T
                depths = sign * rotated[:, axis]
                support = rotated[np.isclose(depths, depths.min(), rtol=0, atol=1e-7)]
                matrix[:3, 3] = point - (support.min(axis=0) + support.max(axis=0)) / 2
        else:
            mesh, cad_status = _solid(req), None
            lo, hi = (np.asarray(b, dtype=float) for b in mesh.bounds)
            rim = _base_rim(mesh)
            placed = _place(req, reg, scan_id, by_id, float(hi[2] - lo[2]))
            quat = _quat_xyzw(placed.rotation)
            matrix = np.eye(4)
            matrix[:3, :3] = placed.rotation
            matrix[:3, 3] = placed.base
    except AddPartError as e:
        return {"error": "bad_request", "message": str(e)}

    # Box in the part's own frame, then placed: the OBB is exact for a rotated solid too.
    mesh.apply_transform(matrix)
    mesh.visual.vertex_colors = np.tile(_ADDED_COLOR, (len(mesh.vertices), 1))
    center = (matrix[:3, :3] @ ((lo + hi) / 2.0) + matrix[:3, 3]).tolist()
    extent = np.maximum(hi - lo, 0.001).tolist()
    rim_world = rim @ matrix[:3, :3].T + matrix[:3, 3]
    if any(abs(c) > MAX_COORD_MM for c in center):
        return {"error": "bad_request", "message": (
            f"the part would land at {fmt_vec(center)}, more than {MAX_COORD_MM:g} mm from the scan origin")}

    part_id = replace_part.part_id if replace_part else _new_part_id(set(part_ids))
    version = "_" + uuid.uuid4().hex[:10] if replace_part else ""
    mesh_key = f"scans/{scan_id}/parts/{part_id}{version}.glb"
    thumb_key = f"scans/{scan_id}/renders/{part_id}{version}.png"
    scan = store.get_scan(scan_id)
    revision = scan.reconstruction_revision if scan else max(
        (p.reconstruction_revision for p in parts), default=0)
    shape = _shape_words(req, extent)
    part = Part.model_validate({
        "part_id": part_id, "reconstruction_revision": revision,
        "measurement_revision": replace_part.measurement_revision + 1 if replace_part else 1,
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
    contact, contact_note = None, None
    if req.attach_to is not None:
        try:
            target = (placed.target_mesh if placed is not None and placed.target_mesh is not None
                      else _part_mesh(reg, scan_id, by_id[req.attach_to]))
            contact = contact_check(mesh, target, req.attach_to, rim_world)
        except AddPartError as e:
            contact_note = f"contact with {req.attach_to} not checked: {e}"
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
        "center_mm": _round(center), "extent_mm": _round(extent),
        "placement": _placement_result(req, placed, quat, matrix, extent),
        "contact": contact,
        "relation": ({"a": part_id, "type": req.relation, "b": req.attach_to} if relation else None),
        "scene_graph_updated": new_graph is not None,
        "message": ("Saved. It now appears in the parts list, the hierarchy and the 3D viewer. Check it "
                    "with render_view on this part together with the part it attaches to."),
    }
    if req.kind == "cad":
        result["cad_id"] = req.cad_id
        result["cad_status"] = cad_status
    notes = req.notes + ([contact_note] if contact_note else []) + warnings
    if notes:
        result["notes"] = notes
    return PartAddedSignal(part_id=part_id, result=result)


def _placement_result(req: _Request, placed: _Placed | None, quat, matrix: np.ndarray,
                      extent: list[float]) -> dict:
    rotation = matrix[:3, :3]
    if placed is None:
        out = {"method": "cad", "rotation_quat": _round(quat, 4), "translate_mm": _round(matrix[:3, 3])}
        if req.surface is not None:
            out["surface"] = {"part_id": req.attach_to, "face": req.surface, "offset_uv": _round(req.offset_uv, 3)}
        return out
    up = rotation[:, 2]
    out = {"method": req.placement, "base_mm": _round(placed.base), "top_mm": _round(placed.base + up * extent[2]),
           "up": _round(up, 3), "width_axis": _round(rotation[:, 0], 3), "depth_axis": _round(rotation[:, 1], 3)}
    if placed.hit is not None:
        out["surface"] = {"part_id": req.on_part, "point_mm": _round(placed.hit.point),
                          "normal": _round(placed.hit.normal, 3),
                          "found_by": "ray" if placed.hit.found_by == "ray" else "nearest point (ray missed)"}
        out["embed_mm"] = round(float(placed.embed), 2)
    if req.surface is not None:
        out["surface"] = {"part_id": req.attach_to, "face": req.surface, "offset_uv": _round(req.offset_uv, 3),
                          "point_mm": _round(placed.base)}
    return out


def _new_part_id(existing: set[str]) -> str:
    while True:
        part_id = "added_" + uuid.uuid4().hex[:12]
        if part_id not in existing:
            return part_id


def _shape_words(req: _Request, extent: list[float]) -> str:
    if req.kind == "cad":
        return f"the model of CAD result {req.cad_id}, {' x '.join(fmt_mm(v) for v in extent)} mm"
    w, d, h = req.size
    round_base = math.isclose(w, d, rel_tol=1e-6, abs_tol=1e-3)
    base = f"{fmt_mm(w)} mm across" if round_base else f"{fmt_mm(w)} x {fmt_mm(d)} mm"
    if req.kind == "box":
        return f"box {fmt_mm(w)} x {fmt_mm(d)} x {fmt_mm(h)} mm (width x depth x height)"
    if req.kind == "cylinder":
        return f"{'cylinder' if round_base else 'elliptical cylinder'} {base}, {fmt_mm(h)} mm tall"
    if req.kind in ("cone", "pyramid"):
        name = "cone" if req.kind == "cone" else f"{req.sides}-sided pyramid"
        if req.top_scale > 0:
            tip = f", cut flat at {round(req.top_scale * 100)}% of the base"
        elif req.tip_radius > 0:
            tip = f", pointed with a {fmt_mm(req.tip_radius)} mm rounded tip"
        else:
            tip = ", pointed"
        return f"{name}, base {base}, {fmt_mm(h)} mm tall{tip}"
    if req.kind == "sphere":
        if round_base and math.isclose(w, h, rel_tol=1e-6, abs_tol=1e-3):
            return f"sphere {fmt_mm(w)} mm across"
        return f"ellipsoid {fmt_mm(w)} x {fmt_mm(d)} x {fmt_mm(h)} mm"
    if req.kind == "torus":
        return f"ring {base} outside, tube {fmt_mm(h)} mm thick"
    if req.kind in ("triangular_prism", "cat_ear", *HEADPHONE_EARS):
        name = {"triangular_prism": "triangular prism", "cat_ear": "cat ear"}.get(
            req.kind, req.kind.replace("_", " "))
        return f"{name} {fmt_mm(w)} x {fmt_mm(d)} x {fmt_mm(h)} mm (width x depth x height)"
    if req.kind == "capsule":
        return f"capsule {base}, {fmt_mm(h)} mm long"
    return (f"flat profile of {len(req.outline)} points, {fmt_mm(w)} mm wide, {fmt_mm(h)} mm tall and "
            f"{fmt_mm(d)} mm thick")


def _dimensions(req: _Request) -> list[dict]:
    """The sizes the user chose, as user-set values (no scan value): shown as 'your value'."""
    if req.size is None:
        return []
    w, d, h = req.size
    round_base = math.isclose(w, d, rel_tol=1e-6, abs_tol=1e-3)
    if req.kind == "torus":
        named = ([("outer_diameter", w)] if round_base else [("outer_width", w), ("outer_depth", d)])
        named.append(("tube_thickness", h))
    elif req.kind == "sphere" and round_base and math.isclose(w, h, rel_tol=1e-6, abs_tol=1e-3):
        named = [("diameter", w)]
    elif req.kind in ("cylinder", "cone", "capsule") and round_base:
        named = [("diameter", w), ("length" if req.kind == "capsule" else "height", h)]
    elif req.kind == "profile":
        named = [("width", w), ("height", h), ("thickness", d)]
    else:
        named = [("width", w), ("depth", d), ("height", h)]
    if req.kind == "cone" and req.tip_radius > 0:
        named.append(("tip_radius", req.tip_radius))
    return [{"name": name, "value": {"scan_mm": None, "caliper_mm": round(value, 3),
                                     "user_confirmed": True, "preferred": "caliper"}}
            for name, value in named]


# ---------------------------------------------------------------- CAD results, assets, storage
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
