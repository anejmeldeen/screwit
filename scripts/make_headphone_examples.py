"""Generate two synthetic headphone demos. Dimensions are illustrative, not manufacturer specs.

Run: PYTHONPATH=backend backend/.venv/bin/python scripts/make_headphone_examples.py
The cat-ear variant reuses identical base geometry and adds two removable parts.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import trimesh
from make_sink_example import shell
from PIL import Image
from screwit.contracts import Part, Reconstruction, Scan, SceneGraph
from screwit.contracts.models import MESH_INPUT_SKIPPED

ROOT = Path(__file__).resolve().parents[1]
SCAN_ID = "scn_headphones"
BASE = ROOT / "fixtures/scans" / SCAN_ID


def arch(radius, thickness, depth, lo=0., hi=np.pi):
    t = np.linspace(lo, hi, 96)
    v = []
    for a in t:
        for rad, y in [(radius, -depth/2), (radius, depth/2),
                       (radius-thickness, depth/2), (radius-thickness, -depth/2)]:
            v.append([rad*np.cos(a), y, 137 + rad*np.sin(a)])
    faces = []
    for j in range(len(t)-1):
        for k in range(4):
            a, b = 4*j+k, 4*j+(k+1)%4
            c, d = b+4, a+4
            faces.extend([[a,b,c], [a,c,d]])
    faces.extend([[0,2,1],[0,3,2]])
    a = 4*(len(t)-1)
    faces.extend([[a,a+1,a+2],[a,a+2,a+3]])
    m = trimesh.Trimesh(v, faces)
    m.fix_normals()
    return m


def ellipsoid(center, scale):
    m = trimesh.creation.icosphere(subdivisions=3)
    m.apply_scale(scale)
    m.apply_translation(center)
    return m


def cushion(side):
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    loops = []
    # Rounded annular cushion, axis X, with a clear central ear opening.
    for a in np.linspace(0, 2*np.pi, 16, endpoint=False):
        x = side*(75 + 7*np.sin(a))
        ry, rz = 25 + 7*np.cos(a), 36 + 7*np.cos(a)
        loops.append(np.column_stack([np.full(64,x), ry*np.cos(t), 88+rz*np.sin(t)]))
    return shell(loops)


def ear(side):
    # Extruded triangular accessory follows the upper headband; independently selectable.
    points = np.array([[side*35,-13,218],[side*79,-13,190],[side*82,-13,263],
                       [side*35,13,218],[side*79,13,190],[side*82,13,263]])
    return trimesh.convex.convex_hull(points)


def assembly(cat_ears=False):
    headband = trimesh.util.concatenate([arch(99, 7, 29), arch(91, 7, 25, .3, np.pi-.3)])
    out = [("Headband", "Curved over-ear headband with padded underside; illustrative Crusher Evo-inspired shape.", headband)]
    for side, name in [(-1,"Left"),(1,"Right")]:
        body = ellipsoid([side*94,0,88], [17,34,48])
        hinge = trimesh.creation.cylinder(radius=9,height=19,sections=32)
        hinge.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
        hinge.apply_translation([side*94,0,137])
        arm = trimesh.creation.box([9,18,30])
        arm.apply_translation([side*95,0,135])
        out.append((name+" earcup", "Separate oval driver housing with swivel hinge and headband slider.",
                    trimesh.util.concatenate([body,hinge,arm])))
    for side, name in [(-1,"Left"),(1,"Right")]:
        out.append((name+" ear cushion", "Removable oval padded earmuff with an open center.", cushion(side)))
    if cat_ears:
        for side,name in [(-1,"Left"),(1,"Right")]:
            out.append((name+" cat ear", "Removable triangular cat-ear accessory on the headband.", ear(side)))
    return out


def write(cat_ears: bool = False) -> None:
    global SCAN_ID, BASE
    SCAN_ID = "scn_headphones_cat" if cat_ears else "scn_headphones"
    BASE = ROOT / "fixtures/scans" / SCAN_ID
    spec = importlib.util.spec_from_file_location("renderer", ROOT / "scripts/home_preview_renderer.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.BG_COLOR = (24, 25, 24)
    palette = json.loads((ROOT / "web/lib/partPalette.json").read_text())
    parts, meshes = [], []
    for i, (label, description, mesh) in enumerate(assembly(cat_ears)):
        pid = f"p{i + 1}"
        mesh.apply_translation([0, 0, 0])
        assert mesh.is_watertight, label
        assert mesh.volume > 0, label
        rgba = [*bytes.fromhex(palette[i % len(palette)][1:]), 255]
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        mesh_key = f"scans/{SCAN_ID}/parts/{pid}.glb"
        thumb_key = f"scans/{SCAN_ID}/renders/{pid}.png"
        for key, data in [
            (mesh_key, mesh.export(file_type="glb")),
            (thumb_key, renderer.render_mesh_png(mesh.vertices, mesh.faces, mesh.visual.vertex_colors,
                                                angle_deg=-65, width=512, height=512)),
        ]:
            path = ROOT / "fixtures" / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        center = mesh.bounds.mean(axis=0).tolist()
        dimensions = [{"name": name, "value": {"scan_mm": float(value), "preferred": "scan"}}
                      for name, value in zip(["width", "depth", "height"], mesh.extents, strict=True)]
        part = Part.model_validate({
            "part_id": pid, "reconstruction_revision": 1, "measurement_revision": 1,
            "label": label, "label_confidence": 1, "description": description,
            "mesh_uri": mesh_key, "thumbnail_uri": thumb_key, "centroid_mm": center,
            "obb": {"center": center, "extent": mesh.extents.tolist(), "rotation_quat": [0, 0, 0, 1]},
            "dimensions": dimensions, "features": [], "material_guess": {"value": "Illustrative material", "confidence": 1},
            "source": "user_edited",
        })
        parts.append(part.model_dump(mode="json"))
        meshes.append(mesh)
    whole = trimesh.util.concatenate(meshes)
    (BASE / "mesh").mkdir(exist_ok=True)
    (BASE / "mesh/object.glb").write_bytes(whole.export(file_type="glb"))
    preview = renderer.render_mesh_png(whole.vertices, whole.faces, whole.visual.vertex_colors,
                                       angle_deg=-65, width=1600, height=960)
    (BASE / "renders/thumbnail.png").write_bytes(preview)
    image = Image.open(io.BytesIO(preview))
    image.save(ROOT / f"web/public/examples/{SCAN_ID}.webp", quality=94)
    (BASE / "parts.json").write_text(json.dumps(parts, indent=2) + "\n")
    scan = json.loads((ROOT / "fixtures/scans/scn_trap/scan.json").read_text())
    scan.update(scan_id=SCAN_ID, created_at="2026-11-15T12:00:00Z")
    scan["input"] = {"type": "mesh", "uri": f"scans/{SCAN_ID}/mesh/object.glb", "mesh_units": "millimetres"}
    for stage in scan["stages"]:
        stage["message"] = f"{len(parts)} editable components" if stage["name"] == "parts" else "Synthetic example geometry"
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
    graph = SceneGraph.model_validate({
        "scan_id": SCAN_ID, "object": {"category": "over-ear headphones", "summary": "Synthetic Crusher Evo-inspired headphones" + (" with cat ears" if cat_ears else "")},
        "parts": [p["part_id"] for p in parts],
        "relations": [{"a": a, "b": b, "type": "fastened", "confidence": 1}
                   for a, b in [("p2", "p1"), ("p3", "p1"), ("p4", "p2"), ("p5", "p3")] + ([("p6", "p1"), ("p7", "p1")] if cat_ears else [])],
        "function": "The headband supports two over-ear driver housings and removable cushions. Illustrative geometry, not manufacturer CAD or measured replacement dimensions.",
        "failure_hypotheses": [],
    })
    (BASE / "scene_graph.json").write_text(graph.model_dump_json(indent=2) + "\n")
    demos_path = ROOT / "fixtures/demo_scans.json"
    demos = json.loads(demos_path.read_text())
    demos = [d for d in demos if d["scan_id"] != SCAN_ID]
    demos.insert(0, {"scan_id": SCAN_ID, "title": ("Crusher Evo-inspired · Cat ears" if cat_ears else "Crusher Evo-inspired · Headphones"), "scenario": "A",
                     "thumbnail_uri": f"scans/{SCAN_ID}/renders/thumbnail.png"})
    demos_path.write_text(json.dumps(demos, indent=2) + "\n")
    print(f"Built {len(parts)} watertight components, {len(whole.faces)} triangles total")


if __name__ == "__main__":
    write()
    write(cat_ears=True)
