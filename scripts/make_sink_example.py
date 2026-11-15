"""Build a recognizable five-component sink assembly in scan-frame millimetres.

Synthetic editable example, not a real scan. Existing examples and user edits
are left untouched. Run with PYTHONPATH=backend backend/.venv/bin/python.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from screwit.contracts import Part, Reconstruction, Scan, SceneGraph
from screwit.contracts.models import MESH_INPUT_SKIPPED

ROOT = Path(__file__).resolve().parents[1]
SCAN_ID = "scn_sink"
BASE = ROOT / "fixtures/scans" / SCAN_ID
N = 64


def shell(loops: list[np.ndarray]) -> trimesh.Trimesh:
    vertices = np.concatenate(loops)
    n = len(loops[0])
    faces = []
    for row in range(len(loops)):
        next_row = (row + 1) % len(loops)
        for i in range(n):
            a, b = row * n + i, row * n + (i + 1) % n
            c, d = next_row * n + (i + 1) % n, next_row * n + i
            faces.extend([[a, b, c], [a, c, d]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals(multibody=True)
    return mesh


def rectangle(width: float, depth: float, z: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.sign(np.cos(t)) * abs(np.cos(t)) ** .4 * width / 2,
                            np.sign(np.sin(t)) * abs(np.sin(t)) ** .4 * depth / 2,
                            np.full(N, z)])


def circle(radius: float, z: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.full(N, z)])


def tube(path, radius=20., thickness=2., sections=24) -> trimesh.Trimesh:
    points = np.array(path, dtype=float)
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    t = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    normal = np.array([1., 0., 0.])
    outer, inner = [], []
    for point, tangent in zip(points, tangents, strict=True):
        normal = normal - tangent * np.dot(normal, tangent)
        if np.linalg.norm(normal) < .01:
            normal = np.cross(tangent, [0., 1., 0.])
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        ring = np.cos(t)[:, None] * normal + np.sin(t)[:, None] * binormal
        outer.append(point + radius * ring)
        inner.append(point + (radius - thickness) * ring)
    return shell([*outer, *reversed(inner)])


def collar(center, radius=27., height=12., bore=20.) -> trimesh.Trimesh:
    mesh = trimesh.creation.annulus(r_min=bore, r_max=radius, height=height, sections=32)
    mesh.apply_translation(center)
    return mesh


def assembly() -> list[tuple[str, str, trimesh.Trimesh]]:
    basin = shell([
        rectangle(580, 460, 300), rectangle(500, 350, 300),
        rectangle(380, 250, 120), circle(20, 120), circle(20, 117),
        rectangle(386, 256, 117), rectangle(506, 356, 297), rectangle(580, 460, 297),
    ])
    arch = [[0, 100 + 90 * np.cos(t), 420 + 90 * np.sin(t)] for t in np.linspace(0, np.pi, 24)]
    spout = tube([[0, 190, 306], [0, 190, 360], *arch, [0, 10, 395]], radius=12, thickness=2)
    base = trimesh.creation.cylinder(radius=30, height=12, sections=32)
    base.apply_translation([0, 190, 306])
    handle = trimesh.creation.box([52, 12, 9])
    handle.apply_translation([29, 190, 329])
    faucet = trimesh.util.concatenate([spout, base, handle])
    drain = trimesh.util.concatenate([
        collar([0, 0, 121], radius=43, height=6), tube([[0, 0, 119], [0, 0, 0]]),
        collar([0, 0, 14]), collar([0, 0, 106]),
    ])
    bend = [[60 + 60 * np.cos(t), 0, -40 + 60 * np.sin(t)] for t in np.linspace(np.pi, 2 * np.pi, 25)]
    trap = trimesh.util.concatenate([
        tube([[0, 0, 0], *bend, [120, 0, 20]]),
        collar([0, 0, -4]), collar([120, 0, 18]),
    ])
    elbow = [[120, 30 + 30 * np.cos(t), 65 + 30 * np.sin(t)] for t in np.linspace(np.pi, np.pi / 2, 13)]
    outlet = trimesh.util.concatenate([
        tube([[120, 0, 20], *elbow, [120, 230, 95]]), collar([120, 0, 28]),
    ])
    return [
        ("Sink basin", "Deep rounded rectangular basin, rolled rim and open drain hole.", basin),
        ("Mixer faucet", "High arched faucet with a mounting base, lever and downward spout.", faucet),
        ("Drain and tailpiece", "Drain flange and vertical tailpiece connect the basin to the trap.", drain),
        ("P-trap", "Hollow U-bend retains a water seal beneath the sink.", trap),
        ("Outlet pipe", "Curved waste arm connects the trap outlet toward the wall.", outlet),
    ]


def write() -> None:
    spec = importlib.util.spec_from_file_location("renderer", ROOT / "scripts/home_preview_renderer.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.BG_COLOR = (24, 25, 24)
    palette = json.loads((ROOT / "web/lib/partPalette.json").read_text())
    parts, meshes = [], []
    for i, (label, description, mesh) in enumerate(assembly()):
        pid = f"p{i + 1}"
        mesh.apply_translation([0, 0, 120])
        assert mesh.is_watertight, label
        assert mesh.volume > 0, label
        rgba = [*bytes.fromhex(palette[i % len(palette)][1:]), 255]
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        mesh_key = f"scans/{SCAN_ID}/parts/{pid}.glb"
        thumb_key = f"scans/{SCAN_ID}/renders/{pid}.png"
        for key, data in [
            (mesh_key, mesh.export(file_type="glb")),
            (thumb_key, renderer.render_mesh_png(mesh.vertices, mesh.faces, mesh.visual.vertex_colors,
                                                angle_deg=-45, width=512, height=512)),
        ]:
            path = ROOT / "fixtures" / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        center = mesh.bounds.mean(axis=0).tolist()
        dimensions = [{"name": name, "value": {"scan_mm": float(value), "preferred": "scan"}}
                      for name, value in zip(["width", "depth", "height"], mesh.extents, strict=True)]
        part = Part.model_validate(dict(
            part_id=pid, reconstruction_revision=1, measurement_revision=1,
            label=label, label_confidence=1, description=description,
            mesh_uri=mesh_key, thumbnail_uri=thumb_key, centroid_mm=center,
            obb={"center": center, "extent": mesh.extents.tolist(), "rotation_quat": [0, 0, 0, 1]},
            dimensions=dimensions, features=[], material_guess={"value": "Illustrative material", "confidence": 1},
            source="user_edited",
        ))
        parts.append(part.model_dump(mode="json"))
        meshes.append(mesh)
    whole = trimesh.util.concatenate(meshes)
    (BASE / "mesh").mkdir(exist_ok=True)
    (BASE / "mesh/object.glb").write_bytes(whole.export(file_type="glb"))
    preview = renderer.render_mesh_png(whole.vertices, whole.faces, whole.visual.vertex_colors,
                                       angle_deg=-45, width=1600, height=960)
    (BASE / "renders/thumbnail.png").write_bytes(preview)
    image = Image.open(io.BytesIO(preview))
    image.save(ROOT / "web/public/examples/scn_sink.webp", quality=94)
    (BASE / "parts.json").write_text(json.dumps(parts, indent=2) + "\n")
    scan = json.loads((ROOT / "fixtures/scans/scn_trap/scan.json").read_text())
    scan.update(scan_id=SCAN_ID, created_at="2026-11-14T22:00:00Z")
    scan["input"] = {"type": "mesh", "uri": f"scans/{SCAN_ID}/mesh/object.glb", "mesh_units": "millimetres"}
    for stage in scan["stages"]:
        stage["message"] = "5 editable components" if stage["name"] == "parts" else "Synthetic example geometry"
        if stage["name"] in MESH_INPUT_SKIPPED:
            stage.update(state="skipped", progress=None)
    Scan.model_validate(scan)
    (BASE / "scan.json").write_text(json.dumps(scan, indent=2) + "\n")
    recon = json.loads((ROOT / "fixtures/scans/scn_trap/reconstruction.json").read_text())
    recon.update(scan_id=SCAN_ID, mesh_uri=f"scans/{SCAN_ID}/mesh/object.glb", splat_uri=None,
                 mesh_mobile_uri=None, splat_mobile_uri=None, keyframes=[])
    recon["object_bbox_mm"] = {"min": whole.bounds[0].tolist(), "max": whole.bounds[1].tolist()}
    recon.update(scale={"method": "manual"}, frame_origin="object_bottom_center")
    recon["quality"].update(frames_used=0, total_frames=0, mean_depth_confidence=1, gaussians=0)
    # A synthetic mesh has no capture frames; the editor supplies orbit controls.
    cameras = {"scan_id": SCAN_ID, "reconstruction_revision": 1, "frames": []}
    (BASE / "geometry").mkdir(exist_ok=True)
    (BASE / "geometry/cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    recon["cameras_uri"] = f"scans/{SCAN_ID}/geometry/cameras.json"
    Reconstruction.model_validate(recon)
    (BASE / "reconstruction.json").write_text(json.dumps(recon, indent=2) + "\n")
    graph = SceneGraph.model_validate(dict(
        scan_id=SCAN_ID, object={"category": "kitchen sink", "summary": "Synthetic five-piece kitchen sink assembly"},
        parts=[p["part_id"] for p in parts],
        relations=[{"a": a, "b": b, "type": "fastened", "confidence": 1}
                   for a, b in [("p2", "p1"), ("p3", "p1"), ("p4", "p3"), ("p5", "p4")]],
        function="The faucet supplies water to the basin; the drain and P-trap carry it away and retain a water seal.",
        failure_hypotheses=[],
    ))
    (BASE / "scene_graph.json").write_text(graph.model_dump_json(indent=2) + "\n")
    demos_path = ROOT / "fixtures/demo_scans.json"
    demos = json.loads(demos_path.read_text())
    demos = [d for d in demos if d["scan_id"] != SCAN_ID]
    demos.insert(0, {"scan_id": SCAN_ID, "title": "Kitchen sink · 5-piece assembly", "scenario": "A",
                     "thumbnail_uri": f"scans/{SCAN_ID}/renders/thumbnail.png"})
    demos_path.write_text(json.dumps(demos, indent=2) + "\n")
    print(f"Built {len(parts)} watertight components, {len(whole.faces)} triangles total")


if __name__ == "__main__":
    write()
