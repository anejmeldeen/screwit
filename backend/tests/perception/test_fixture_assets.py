"""Fixture assets (scripts/make_fixture_assets.py): each part GLB's bounds match the JSON fixture.

SPEC/P3 brief: part GLB bounds match fixture centroid/obb within 2 mm. We compare each generated
part GLB's world-space axis-aligned bounds against the Part's centroid_mm and obb.extent (the
OBB extents equal the AABB size for these axis-aligned fixtures, including the knob shaft whose
OBB is a pure 90-degree rotation).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh
from pydantic import TypeAdapter

from screwit.contracts import Part

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "fixtures" / "scans"
TOL_MM = 2.0


def _load_script():
    path = ROOT / "scripts" / "make_fixture_assets.py"
    spec = importlib.util.spec_from_file_location("make_fixture_assets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bounds_from_glb(data: bytes):
    import io

    mesh = trimesh.load(file_obj=io.BytesIO(data), file_type="glb", force="mesh")
    lo, hi = mesh.bounds
    return (lo + hi) / 2.0, (hi - lo)


def test_part_glb_bounds_match_fixtures():
    outputs = _load_script().build()
    for scan_id in ("scn_trap", "scn_knob"):
        parts = TypeAdapter(list[Part]).validate_json((FIX / scan_id / "parts.json").read_text())
        for p in parts:
            glb_path = FIX / scan_id / "parts" / f"{p.part_id}.glb"
            assert glb_path in outputs, f"generator did not produce {glb_path}"
            center, extent = _bounds_from_glb(outputs[glb_path])
            assert np.allclose(center, p.centroid_mm, atol=TOL_MM), \
                f"{scan_id}/{p.part_id} centroid {center} != {p.centroid_mm}"
            assert np.allclose(sorted(extent), sorted(p.obb.extent), atol=TOL_MM), \
                f"{scan_id}/{p.part_id} extent {extent} != {p.obb.extent}"


def test_cameras_json_is_valid_and_a_ring():
    from screwit.contracts import Cameras

    for scan_id in ("scn_trap", "scn_knob"):
        cams = Cameras.model_validate_json((FIX / scan_id / "geometry" / "cameras.json").read_text())
        assert len(cams.frames) >= 8  # a ring of cameras
        assert cams.reconstruction_revision == 1


def test_object_and_thumbnail_written_under_size_cap():
    outputs = _load_script().build()
    for scan_id in ("scn_trap", "scn_knob"):
        assert (FIX / scan_id / "mesh" / "object.glb") in outputs
        assert (FIX / scan_id / "renders" / "thumbnail.png") in outputs
    assert sum(len(v) for v in outputs.values()) < 10 * 1024 * 1024  # < 10 MB total
