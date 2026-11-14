"""Snapping rules (SPEC 6.2 step 4), including the required truth-table rows."""

from __future__ import annotations

import pytest

from screwit.perception.snapping import snap_diameter


def _snap(feature_type: str, diameter_mm: float):
    return snap_diameter(feature_type, diameter_mm / 2.0)  # the API takes a radius


def test_truth_table_35_7_does_not_snap():
    # 35.7 mm is 2.05 mm from 38.1 (outside the 2.0 mm window) and far from 31.75 -> no snap.
    assert _snap("cylinder", 35.7) is None


def test_truth_table_37_0_snaps_to_38_1():
    res = _snap("cylinder", 37.0)
    assert res is not None
    assert res.snapped_diameter_mm == pytest.approx(38.1)
    assert res.snapped_radius_mm == pytest.approx(38.1 / 2.0)
    assert res.snap_rule == "tubular_1_1_2_in"


def test_snaps_to_1_1_4_in():
    res = _snap("cylinder", 31.0)  # within 0.75 of 31.75
    assert res is not None and res.snap_rule == "tubular_1_1_4_in"


def test_bore_uses_bore_table_rules():
    res = _snap("bore", 37.5)
    assert res is not None and res.snap_rule == "bore_for_tubular_1_1_2_in"
    res2 = _snap("bore", 32.0)
    assert res2 is not None and res2.snap_rule == "bore_for_tubular_1_1_4_in"


def test_never_snaps_metric_bolts_or_shafts():
    # M3-M10 and 6.0 / 6.35 mm shafts must never snap (these roles have no table anyway).
    for d in (3.0, 4.0, 5.0, 6.0, 6.35, 8.0, 10.0):
        assert snap_diameter("d_shaft", d / 2.0) is None
        assert snap_diameter("cylinder", d / 2.0) is None  # far from any tube entry
        assert snap_diameter("bore", d / 2.0) is None


def test_window_edges():
    # Exactly 2.0 mm away still snaps; just past it does not.
    assert _snap("cylinder", 38.1 - 2.0) is not None
    assert _snap("cylinder", 38.1 + 2.0) is not None
    assert _snap("cylinder", 38.1 + 2.0001) is None


def test_separation_floor_is_respected():
    # The two tube entries are 6.35 mm apart (>= 6), so both remain snappable; a value exactly
    # between them (34.925) is outside both windows and does not snap.
    assert _snap("cylinder", 34.925) is None


def test_non_snapping_roles_return_none():
    assert snap_diameter("plane", 19.0) is None
    assert snap_diameter("hole", 19.0) is None
    assert snap_diameter("thread_suspected", 19.0) is None
    assert snap_diameter("cylinder", None) is None
