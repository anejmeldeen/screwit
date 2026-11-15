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


def paint(mesh, rgb):
    mesh.visual.vertex_colors = np.tile([*rgb,255], (len(mesh.vertices),1))
    return mesh


def oval(rx, ry, z, scale=1):
    t=np.linspace(0,2*np.pi,64,endpoint=False)
    return np.column_stack([scale*rx*np.sign(np.cos(t))*abs(np.cos(t))**.82,
                            scale*ry*np.sign(np.sin(t))*abs(np.sin(t))**.82, np.full(len(t),z)])


def housing():
    rings=[oval(36,46,z,k) for z,k in [(5,.80),(7,.94),(11,1),(24,1),(28,.96),(30,.88)]]
    return trimesh.convex.convex_hull(np.concatenate(rings))


def padded_ring():
    loops=[]
    for a in np.linspace(0,2*np.pi,24,endpoint=False):
        loops.append(oval(27+9*np.cos(a),36+9*np.cos(a),38+11*np.sin(a)))
    return shell(loops)


def band(radius, thickness, depth, lo=0, hi=np.pi):
    t=np.linspace(lo,hi,64)
    loops=[]
    # Rounded rectangle swept along the band, lying flat on the table.
    q=np.linspace(0,2*np.pi,16,endpoint=False)
    for a in t:
        radial=radius + thickness/2*np.sign(np.cos(q))*abs(np.cos(q))**.45
        z=25 + depth/2*np.sign(np.sin(q))*abs(np.sin(q))**.45
        loops.append(np.column_stack([radial*np.cos(a),60+radial*np.sin(a),z]))
    vertices=np.concatenate(loops); faces=[]
    n=len(q)
    for j in range(len(t)-1):
        for k in range(n):
            x=j*n+k; y=j*n+(k+1)%n
            faces.extend([[x,y,y+n],[x,y+n,x+n]])
    for j,flip in [(0,True),(len(t)-1,False)]:
        center=len(vertices);vertices=np.vstack([vertices,loops[j].mean(axis=0)])
        for k in range(n):
            f=[center,j*n+k,j*n+(k+1)%n]
            faces.append(f[::-1] if flip else f)
    m=trimesh.Trimesh(vertices,faces);m.fix_normals();return m


def capsule(center, size, color):
    m=trimesh.creation.icosphere(subdivisions=2);m.apply_scale(np.array(size)/2)
    m.apply_translation(center);return paint(m,color)


def assembly(cat_ears=False):
    dark=[40,44,49]; pad=[55,59,64]; trim=[75,80,86]
    headband=trimesh.util.concatenate([
        paint(band(89,7,31),dark), paint(band(82,9,27,.20,np.pi-.20),pad),
        paint(band(89.5,8,33,.24,.29),trim),paint(band(89.5,8,33,np.pi-.29,np.pi-.24),trim)])
    out=[("Headband","Broad curved matte headband with padded liner and sliding adjustment seams.",headband)]
    for side,name in [(-1,"Left"),(1,"Right")]:
        x=side*89
        body=paint(housing(),dark); body.apply_translation([x,18,0])
        plate=paint(trimesh.convex.convex_hull(np.concatenate([oval(33,43,28,.96),oval(33,43,31,.93)])),trim)
        plate.apply_translation([x,18,0])
        yoke=capsule([x,57,25],[18,36,14],dark)
        hinge=capsule([x,57,33],[21,16,8],trim)
        controls=[]
        for y in [-2,13,28]:
            controls.append(capsule([x+side*35,y,23],[5,9,7],trim))
        if side==-1:
            controls.append(capsule([x-35,43,22],[5,12,6],[235,115,28]))
        out.append((name+" earcup","Flat oval driver housing, swivel yoke, control buttons and housing seam.",
                    trimesh.util.concatenate([body,plate,yoke,hinge,*controls])))
    for side,name in [(-1,"Left"),(1,"Right")]:
        m=paint(padded_ring(),pad);m.apply_translation([side*89,18,0])
        # Recessed speaker cloth sits below the cushion lip, leaving a deep visible opening.
        cloth=paint(trimesh.convex.convex_hull(np.concatenate([oval(19,28,28),oval(19,28,29)])),[23,25,29])
        cloth.apply_translation([side*89,18,0])
        out.append((name+" ear cushion","Thick removable padded oval earmuff facing upward, with recessed speaker cloth.",
                    trimesh.util.concatenate([m,cloth])))
    if cat_ears:
        for side,name in [(-1,"Left"),(1,"Right")]:
            points=[[side*29,143,12],[side*70,116,12],[side*73,190,12],
                    [side*29,143,38],[side*70,116,38],[side*73,190,38]]
            m=paint(trimesh.convex.convex_hull(points),dark)
            out.append((name+" cat ear","Separate triangular headband accessory.",m))
    return out


def write(cat_ears: bool = False) -> None:
    global SCAN_ID, BASE
    SCAN_ID = "scn_headphones_cat" if cat_ears else "scn_headphones"
    BASE = ROOT / "fixtures/scans" / SCAN_ID
    spec = importlib.util.spec_from_file_location("renderer", ROOT / "scripts/home_preview_renderer.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.BG_COLOR = (100, 104, 108)
    parts, meshes = [], []
    for i, (label, description, mesh) in enumerate(assembly(cat_ears)):
        pid = f"p{i + 1}"
        mesh.apply_translation([0, 0, 0])
        assert mesh.is_watertight, label
        assert mesh.volume > 0, label
        mesh_key = f"scans/{SCAN_ID}/parts/{pid}.glb"
        thumb_key = f"scans/{SCAN_ID}/renders/{pid}.png"
        for key, data in [
            (mesh_key, mesh.export(file_type="glb")),
            (thumb_key, renderer.render_mesh_png(mesh.vertices, mesh.faces, mesh.visual.vertex_colors,
                                                angle_deg=-60, width=512, height=512)),
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
                                       angle_deg=-60, width=1600, height=960)
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
