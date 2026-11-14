"""CPU headless render (SPEC 6.3 render_view): produces a non-empty, non-blank PNG."""

from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image

from screwit.perception.render import render_mesh_png


def test_render_box_is_non_empty_png():
    box = trimesh.creation.box(extents=(30, 20, 10))
    png = render_mesh_png(np.asarray(box.vertices, float), np.asarray(box.faces, int),
                          width=128, height=128)
    assert len(png) > 0
    img = Image.open(io.BytesIO(png))
    assert img.size == (128, 128) and img.mode == "RGB"
    arr = np.asarray(img)
    # The mesh must have drawn something: more than one distinct colour (not a flat background).
    assert len(np.unique(arr.reshape(-1, 3), axis=0)) > 1
    # A meaningful fraction of pixels differ from the background colour.
    bg = np.array([24, 26, 30])
    assert (np.abs(arr - bg).sum(axis=2) > 10).mean() > 0.05


def test_render_empty_mesh_still_returns_png():
    png = render_mesh_png(np.empty((0, 3)), np.empty((0, 3), dtype=int), width=64, height=64)
    img = Image.open(io.BytesIO(png))
    assert img.size == (64, 64)


def test_render_uses_vertex_colors():
    box = trimesh.creation.box(extents=(20, 20, 20))
    colors = np.tile(np.array([220, 40, 40, 255], dtype=np.uint8), (len(box.vertices), 1))
    png = render_mesh_png(np.asarray(box.vertices, float), np.asarray(box.faces, int), colors,
                          width=96, height=96)
    arr = np.asarray(Image.open(io.BytesIO(png)))
    # Red channel should dominate somewhere on the rendered surface.
    reddish = (arr[:, :, 0] > arr[:, :, 1] + 30) & (arr[:, :, 0] > arr[:, :, 2] + 30)
    assert reddish.any()
