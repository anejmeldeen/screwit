"""Checks, preview render and model-source generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from screwit.cad import checks as chk
from screwit.cad import geometry as geo
from screwit.cad.preview import render_preview
from screwit.cad.source import render_model_source
from screwit.contracts import PrinterProfile


@pytest.fixture
def profile() -> PrinterProfile:
    wt = Path(__file__).resolve().parents[3]
    return PrinterProfile.model_validate_json((wt / "profiles" / "prusa_mk4_petg_0.2.json").read_text())


def test_preview_png_is_non_empty(tmp_path):
    box = trimesh.creation.box((10, 10, 10))
    out = render_preview(box, tmp_path / "p.png")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_watertight_check_passes_on_closed_solid():
    box = trimesh.creation.box((5, 5, 5))
    assert chk.watertight_check(box).result == "pass"


def test_fits_build_volume(profile):
    small = trimesh.creation.box((10, 10, 10))
    check, _ = chk.fits_build_volume(small, profile)
    assert check.result == "pass"
    huge = trimesh.creation.box((400, 400, 400))
    check, _ = chk.fits_build_volume(huge, profile)
    assert check.result == "fail"


def test_strength_not_computed_without_profile_value(profile):
    # The shipped profile has no effective_strength_mpa, so a load-bearing part is not_computed.
    c = chk.strength_check(True, "bracket", "+Z up", profile)
    assert c.result == "not_computed"
    # A non-load part is not_computed with its own reason.
    c2 = chk.strength_check(False, "knobs carry no structural load", "+Z up", profile)
    assert c2.result == "not_computed"
    assert "knob" in c2.reason


def test_strength_computed_when_profile_has_value(profile):
    p = profile.model_copy(update={"effective_strength_mpa": {"+Z up": 18.0}})
    c = chk.strength_check(True, "bracket", "+Z up", p)
    assert c.result == "pass"
    assert c.value == 18.0


def test_interference_not_computed_for_non_watertight():
    # An open surface (single triangle strip) is not watertight.
    m = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    assert chk.interference_volume_check(m).result == "not_computed"
    assert chk.interference_volume_check(None).result == "not_computed"


def test_surface_separation_not_computed_without_mesh():
    box = trimesh.creation.box((5, 5, 5))
    assert chk.surface_separation_check(box, None).result == "not_computed"


def test_surface_separation_measures_gap():
    part = trimesh.creation.box((5, 5, 5))
    scanned = trimesh.creation.box((5, 5, 5))
    scanned.apply_translation((20, 0, 0))
    c = chk.surface_separation_check(part, scanned)
    assert c.result == "pass"
    assert c.value == pytest.approx(15.0, abs=1.0)  # 20 mm centre gap minus two 2.5 mm half-widths


def test_model_source_reproduces(tmp_path):
    src = render_model_source("knob", {"outer_diameter": 32.0, "height": 22.0, "shaft_diameter": 6.0,
                                        "shaft_flat_depth": 1.5, "bore_depth": 16.0, "wall_mm": 3.0,
                                        "knurl_count": 24.0, "knurl_depth": 0.8}, 0.12)
    f = tmp_path / "model.py"
    f.write_text(src)
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solid = mod.build()
    assert solid is not None


# -- geometry helpers -----------------------------------------------------

def test_quat_identity():
    m = geo.quat_to_matrix((0, 0, 0, 1))
    assert np.allclose(m, np.eye(3))


def test_rx90_maps_z_to_minus_y():
    # Rx(90 deg) quaternion [sin45, 0, 0, cos45].
    q = (0.7071068, 0, 0, 0.7071068)
    d = geo.transform_direction(q, (0, 0, 1))
    assert np.allclose(d, (0, -1, 0), atol=1e-5)


def test_coaxial_offset_parallel_axes():
    offset = geo.coaxial_offset_mm(np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                                   (0, 0, 0), (0, 0, 1))
    assert offset == pytest.approx(1.0, abs=1e-6)


def test_angle_between():
    assert geo.angle_between_deg((1, 0, 0), (0, 1, 0)) == pytest.approx(90.0, abs=1e-6)
