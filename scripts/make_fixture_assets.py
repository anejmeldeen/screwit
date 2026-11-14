"""Deterministic synthetic binary assets for the hero fixtures (P3 workload bullet; SPEC 3.2).

Produces the binary assets that the JSON fixtures (fixtures/scans/scn_trap, scn_knob) refer to,
built so that each part GLB's bounds match the fixture's centroid / OBB extent within 2 mm. These
are not photoreal; they are synthetic stand-ins (a scan frame, millimetres) good enough for the
viewer, the agent's renders and the measurement/lifting tests.

Writes, per scan:
    mesh/object.glb            whole object, vertex-coloured
    parts/<pid>.glb            one GLB per part (vertex-coloured)
    renders/<pid>.png          per-part thumbnail (CPU z-buffer render)
    renders/thumbnail.png      whole-object thumbnail
    geometry/cameras.json      a valid Cameras doc, a ring of cameras looking at the object

Run from the perception worktree:
    python scripts/make_fixture_assets.py            # writes under fixtures/scans/*/...
Everything is committed (each output is well under 10 MB).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh

from screwit.contracts import Cameras

# Resolve the repo root from this script's location (…/scripts/make_fixture_assets.py).
ROOT = Path(__file__).resolve().parents[1]
SCANS = ROOT / "fixtures" / "scans"

# One palette for the web viewer, gallery thumbnails and generated fixtures.
COLORS = [
    (*tuple(bytes.fromhex(color.removeprefix("#"))), 255)
    for color in json.loads((ROOT / "web/lib/partPalette.json").read_text())
]


def _colored(mesh: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _tube(outer_d: float, length: float, center, axis="z", sections=48) -> trimesh.Trimesh:
    """A solid cylinder of the given outer diameter and length, centred at ``center``."""
    m = trimesh.creation.cylinder(radius=outer_d / 2.0, height=length, sections=sections)
    if axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    return m


def _ring(outer_d: float, inner_d: float, length: float, center, sections=48) -> trimesh.Trimesh:
    """An annulus (washer / nut body): outer cylinder minus inner bore, along Z."""
    outer = trimesh.creation.cylinder(radius=outer_d / 2.0, height=length, sections=sections)
    inner = trimesh.creation.cylinder(radius=inner_d / 2.0, height=length * 1.2, sections=sections)
    try:
        ring = outer.difference(inner)
        if ring is None or ring.is_empty or len(ring.vertices) == 0:
            raise ValueError
    except Exception:
        ring = outer  # boolean engine missing: a solid disc still matches bounds within 2 mm
    ring.apply_translation(center)
    return ring


def _box(size, center) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    return m


def _d_shaft(diameter: float, length: float, flat_depth: float, center,
             sections=40) -> trimesh.Trimesh:
    """A cylinder along -Y with one flat cut (flat normal +Z), centred at ``center``."""
    shaft = trimesh.creation.cylinder(radius=diameter / 2.0, height=length, sections=sections)
    shaft.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    # Cut the +Z side at z = radius - flat_depth by subtracting a slab.
    r = diameter / 2.0
    cut_z = r - flat_depth
    slab = trimesh.creation.box(extents=(diameter * 2, length * 2, diameter))
    slab.apply_translation((0, 0, cut_z + diameter / 2.0))
    try:
        shaft = shaft.difference(slab) or shaft
    except Exception:
        pass
    shaft.apply_translation(center)
    return shaft


def _trap_parts() -> dict[str, trimesh.Trimesh]:
    # Match fixtures/scans/scn_trap/parts.json centroids and OBB extents within 2 mm.
    return {
        # p1 tailpiece: centroid [0,0,250], extent [39,39,140] -> OD 38.1, z 180-320.
        "p1": _tube(38.1, 140.0, (0, 0, 250)),
        # p2 J-bend: centroid [60,0,110], extent [160,40,120]. A U-shaped bend approximated by a
        # bounding slab of the right size so its AABB matches the fixture OBB.
        "p2": _box((160, 40, 120), (60, 0, 110)),
        # p3 slip nut: centroid [0,0,168], extent [62,62,28] -> OD 62, bore ~38, height 28.
        "p3": _ring(62.0, 38.1, 28.0, (0, 0, 168)),
        # p4 slip-joint washer: centroid [0,0,176], extent [44,44,6] -> OD 44, bore ~38, 6 thick.
        "p4": _ring(44.0, 38.1, 6.0, (0, 0, 176)),
    }


def _knob_parts() -> dict[str, trimesh.Trimesh]:
    # Match fixtures/scans/scn_knob/parts.json. Panel plane normal -Y; D-shaft along -Y.
    return {
        # p1 control panel: centroid [0,0,60], extent [120,4,120], normal -Y.
        "p1": _box((120, 4, 120), (0, 0, 60)),
        # p2 D-shaft: centroid [0,-10,60], extent [7,20,7], flat +Z. The fixture OBB bounds the
        # nominal shaft (7 mm across, 20 mm long); a shallow 1.5 mm flat keeps every bound within
        # 2 mm of the fixture extent.
        "p2": _d_shaft(7.0, 20.0, 1.5, (0, -10, 60)),
    }


def _ring_of_cameras(scan_id: str, center, radius_mm: float, n: int = 16,
                     grid=(504, 280)) -> Cameras:
    """A ring of ``n`` cameras on a circle around ``center`` (scan frame), OpenCV world_to_camera."""
    w, h = grid
    fx = fy = 400.0
    K = ((fx, 0.0, w / 2.0), (0.0, fy, h / 2.0), (0.0, 0.0, 1.0))
    frames = []
    cz = center[2] + 40.0
    for i in range(n):
        ang = 2 * math.pi * i / n
        eye = np.array([center[0] + radius_mm * math.cos(ang),
                        center[1] + radius_mm * math.sin(ang), cz])
        fwd = np.asarray(center, float) - eye
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)  # OpenCV Y is down
        R = np.stack([right, down, fwd], axis=0)  # world->camera rotation
        t = -R @ eye
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        frames.append({
            "frame": f"frames/{i * 6 + 1:04d}.jpg",
            "width": w, "height": h,
            "world_to_camera": tuple(tuple(float(x) for x in row) for row in w2c),
            "K": K,
        })
    return Cameras(scan_id=scan_id, reconstruction_revision=1, frames=frames)


def _render_png(mesh: trimesh.Trimesh, angle_deg: float = 30.0) -> bytes:
    from screwit.perception.render import render_mesh_png

    return render_mesh_png(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int),
                           getattr(mesh.visual, "vertex_colors", None), angle_deg=angle_deg)


def build_scan(scan_id: str, parts: dict[str, trimesh.Trimesh], outputs: dict[Path, bytes]) -> None:
    base = SCANS / scan_id
    combined = []
    for i, (pid, mesh) in enumerate(parts.items()):
        _colored(mesh, COLORS[i % len(COLORS)])
        outputs[base / "parts" / f"{pid}.glb"] = mesh.export(file_type="glb")
        outputs[base / "renders" / f"{pid}.png"] = _render_png(mesh)
        combined.append(mesh)
    whole = trimesh.util.concatenate(combined)
    outputs[base / "mesh" / "object.glb"] = whole.export(file_type="glb")
    outputs[base / "renders" / "thumbnail.png"] = _render_png(whole)

    center = tuple(float(x) for x in whole.bounds.mean(axis=0))
    extent = float(np.linalg.norm(whole.bounds[1] - whole.bounds[0]))
    cams = _ring_of_cameras(scan_id, center, radius_mm=extent * 1.2)
    outputs[base / "geometry" / "cameras.json"] = cams.model_dump_json(indent=2).encode()


def build() -> dict[Path, bytes]:
    """Return {path: bytes} for every asset, so a caller can diff before writing (tests do)."""
    outputs: dict[Path, bytes] = {}
    build_scan("scn_trap", _trap_parts(), outputs)
    build_scan("scn_knob", _knob_parts(), outputs)
    return outputs


def write() -> list[Path]:
    written = []
    for path, data in build().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(path)
    return written


if __name__ == "__main__":
    paths = write()
    for p in sorted(paths):
        size = p.stat().st_size
        print(f"wrote {p.relative_to(ROOT)} ({size} bytes)")
    total = sum(p.stat().st_size for p in paths)
    print(f"total {total} bytes across {len(paths)} files")
