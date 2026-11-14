"""Re-render example thumbnails with the editor palette, without changing geometry.

Run from the repository root with backend/.venv/bin/python. Only PNG previews
are written; no GLB, scan metadata, measurements or splats are modified.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def build_previews() -> dict[Path, bytes]:
    # Load the standalone CPU renderer without importing optional model adapters.
    spec = importlib.util.spec_from_file_location(
        "preview_renderer", ROOT / "backend/screwit/perception/render.py",
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.BG_COLOR = (37, 42, 45)  # Same charcoal background as the editor.
    palette = json.loads((ROOT / "web/lib/partPalette.json").read_text())
    outputs = {}
    for scan in json.loads((ROOT / "fixtures/demo_scans.json").read_text()):
        directory = ROOT / "fixtures/scans" / scan["scan_id"]
        meshes = []
        for i, part in enumerate(json.loads((directory / "parts.json").read_text())):
            slot = int(part["part_id"][1:]) - 1 if part["part_id"][1:].isdigit() else i
            rgba = (*bytes.fromhex(palette[slot % len(palette)][1:]), 255)
            mesh = trimesh.load(ROOT / "fixtures" / part["mesh_uri"], force="scene").to_geometry()
            mesh.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
            meshes.append(mesh)
            outputs[ROOT / "fixtures" / part["thumbnail_uri"]] = renderer.render_mesh_png(
                mesh.vertices, mesh.faces, mesh.visual.vertex_colors,
            )
        whole = trimesh.util.concatenate(meshes)
        outputs[ROOT / "fixtures" / scan["thumbnail_uri"]] = renderer.render_mesh_png(
            whole.vertices, whole.faces, whole.visual.vertex_colors,
        )
    return outputs


if __name__ == "__main__":
    for path, content in build_previews().items():
        path.write_bytes(content)
        print(f"Updated {path.relative_to(ROOT)}")
