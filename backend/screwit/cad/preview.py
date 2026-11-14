"""Headless CPU preview render of a CAD part (SPEC 6.4 outputs).

No GPU and no OpenGL: the solid is tessellated to a triangle mesh and drawn with matplotlib's Agg
backend, which renders entirely on the CPU. The result is a non-empty PNG suitable for the result card.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CPU-only, no display; must be set before pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def render_preview(mesh: trimesh.Trimesh, out_path: Path, *, size_px: int = 480) -> Path:
    """Render `mesh` to a PNG at `out_path`. Returns the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    tris = mesh.vertices[mesh.faces]
    coll = Poly3DCollection(tris, alpha=1.0, linewidths=0.1)
    # Shade by face normal z so the form reads; a plain CPU render, not a lit scene.
    normals = mesh.face_normals
    shade = 0.45 + 0.55 * np.clip((normals[:, 2] + 1) / 2, 0, 1)
    colours = np.zeros((len(tris), 4))
    colours[:, 0] = 0.25 * shade + 0.1
    colours[:, 1] = 0.55 * shade
    colours[:, 2] = 0.85 * shade
    colours[:, 3] = 1.0
    coll.set_facecolor(colours)
    coll.set_edgecolor((0.1, 0.1, 0.15, 0.25))
    ax.add_collection3d(coll)

    v = mesh.vertices
    mins, maxs = v.min(axis=0), v.max(axis=0)
    centre = (mins + maxs) / 2
    span = float((maxs - mins).max()) / 2 or 1.0
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, transparent=False, facecolor="white")
    plt.close(fig)
    return out_path
