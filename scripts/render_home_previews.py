"""Render homepage cards from the actual fixture GLBs and shared editor colors."""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("renderer", ROOT / "scripts/home_preview_renderer.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
renderer.BG_COLOR = (24, 25, 24)
palette = json.loads((ROOT / "web/lib/partPalette.json").read_text())
out = ROOT / "web/public/examples"
out.mkdir(parents=True, exist_ok=True)
for scan_id in ("scn_trap", "scn_knob"):
    parts = json.loads((ROOT / f"fixtures/scans/{scan_id}/parts.json").read_text())
    meshes = []
    for part in parts:
        mesh = trimesh.load(ROOT / "fixtures" / part["mesh_uri"], force="scene").to_geometry()
        slot = int(part["part_id"][1:]) - 1
        color = [*bytes.fromhex(palette[slot % len(palette)][1:]), 255]
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        meshes.append(mesh)
    mesh = trimesh.util.concatenate(meshes)
    data = renderer.render_mesh_png(mesh.vertices, mesh.faces, mesh.visual.vertex_colors,
                                    angle_deg=-45, width=2400, height=1440)
    image = Image.open(io.BytesIO(data))
    image = image.resize((1600, 960), Image.Resampling.LANCZOS)
    image.save(out / f"{scan_id}.webp", quality=95)
    print(out / f"{scan_id}.webp")
