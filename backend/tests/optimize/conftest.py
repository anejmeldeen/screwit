"""Shared fixtures and helpers for the X5 optimize tests."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from screwit.optimize import mesh as mesh_mod


@pytest.fixture(scope="session")
def have_gltfpack() -> bool:
    try:
        mesh_mod.gltfpack_binary()
        return True
    except mesh_mod.GltfpackNotFound:
        return False


def coloured_sphere(subdivisions: int, radius_mm: float) -> trimesh.Trimesh:
    """A vertex-coloured icosphere in millimetres; colour is a function of position so a round-trip
    can check the colour survived."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius_mm)
    pos = np.asarray(m.vertices)
    span = np.ptp(pos, axis=0) + 1e-9
    rgb = ((pos - pos.min(axis=0)) / span * 255).astype(np.uint8)
    rgba = np.c_[rgb, np.full((len(rgb), 1), 255, np.uint8)]
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
    return m
