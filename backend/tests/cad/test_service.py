"""CAD service behaviour (SPEC 4.6/4.7/6.4, P5 workload 2/3)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from screwit.cad.service import StaleRequest
from screwit.contracts import CadRequest


def _checks(result):
    return {c.name: c for c in result.checks}


def _load_part_stl(runner, req, parts, storage) -> trimesh.Trimesh:
    result = runner.run(req, parts, storage)
    assert result.files is not None
    import tempfile
    from pathlib import Path

    scratch = Path(tempfile.mkdtemp())
    local = storage.local_path(result.files.stl, scratch)
    return trimesh.load(str(local), force="mesh")


# -- fixture request: ok, all checks present, mate checks required --------

def test_fixture_request_is_ok_with_required_mate_checks(runner, knob_request, knob_parts, storage):
    result = runner.run(knob_request, knob_parts, storage)
    assert result.status == "ok"
    checks = _checks(result)

    required = [
        "watertight", "fits_build_volume",
        "mate_p2_f1_diameter_clearance_mm", "mate_p2_f1_flat_clearance_mm",
        "mate_p2_f1_flat_orientation_deg", "mate_p2_f1_engagement_mm",
        "mate_p2_f1_coaxial_offset_mm",
    ]
    advisory = [
        "min_wall_thickness_mm", "max_unsupported_overhang_deg", "min_feature_mm",
        "minimum_surface_separation_mm", "interference_volume_mm3", "strength_estimate",
    ]
    for name in required + advisory:
        assert name in checks, f"missing check {name}"

    for name in required:
        assert checks[name].kind == "required", f"{name} should be required"
        assert checks[name].result == "pass", f"{name} should pass"
    # Caliper/user_confirmed sources make the mate checks required.
    assert checks["mate_p2_f1_diameter_clearance_mm"].input_source == "caliper"
    assert checks["mate_p2_f1_flat_clearance_mm"].input_source == "user_confirmed"
    # strength is not_computed for a knob with a part-specific reason.
    assert checks["strength_estimate"].result == "not_computed"
    assert checks["strength_estimate"].reason


def test_fixture_outputs_written(runner, knob_request, knob_parts, storage):
    result = runner.run(knob_request, knob_parts, storage)
    assert result.files is not None
    for key in (result.files.stl, result.files.step, result.files.source, result.files.preview):
        assert storage.head(key) is not None, key
        assert storage.head(key)["size"] > 0
    assert result.print_settings is not None
    assert result.print_settings.estimate_method == "volume_formula"


# -- scan-sourced mating dimension -> mate checks advisory ----------------

def test_scan_sourced_shaft_makes_mate_checks_advisory(runner, knob_request, knob_parts, storage):
    req = knob_request.model_copy(deep=True)
    req = CadRequest.model_validate({**req.model_dump(), "param_sources": {
        **req.param_sources, "shaft_diameter": "scan"}})
    result = runner.run(req, knob_parts, storage)
    checks = _checks(result)
    for name in ("mate_p2_f1_diameter_clearance_mm", "mate_p2_f1_flat_clearance_mm",
                 "mate_p2_f1_flat_orientation_deg", "mate_p2_f1_engagement_mm",
                 "mate_p2_f1_coaxial_offset_mm"):
        assert checks[name].kind == "advisory", f"{name} should be advisory with a scan source"
    # Status is still ok because advisory checks never block and the required set is only watertight+fits.
    assert result.status == "ok"


def test_snapped_source_also_makes_mate_checks_advisory(runner, knob_request, knob_parts, storage):
    req = CadRequest.model_validate({**knob_request.model_dump(), "param_sources": {
        **knob_request.param_sources, "shaft_flat_depth": "snapped"}})
    result = runner.run(req, knob_parts, storage)
    checks = _checks(result)
    assert checks["mate_p2_f1_diameter_clearance_mm"].kind == "advisory"


# -- defect 1: placement derived from the mated feature, checks from placed geometry ---

def test_placement_derived_and_overrides_request(runner, knob_request, knob_parts, storage):
    # Give a deliberately wrong request placement; the service must ignore it for a mate and derive
    # a placement that seats the knob on the shaft.
    req = CadRequest.model_validate({**knob_request.model_dump(),
        "placement": {"translate_mm": [999, 999, 999], "rotation_quat": [0, 0, 0, 1]}})
    result = runner.run(req, knob_parts, storage)
    assert result.placement is not None
    # Placement was NOT the bogus request value.
    assert result.placement.translate_mm != (999.0, 999.0, 999.0)

    from screwit.cad import geometry as geo
    q = result.placement.rotation_quat
    t = result.placement.translate_mm
    bore_dir = geo.transform_direction(q, (0, 0, 1))
    bore_pt = geo.transform_point(q, t, (0, 0, 0))
    # Feature p2_f1: axis_dir [0,-1,0], axis_point [0,-2,60].
    axis_angle = geo.angle_between_deg(bore_dir, (0, -1, 0))
    axis_angle = min(axis_angle, 180.0 - axis_angle)
    assert axis_angle < 0.5, f"placed bore axis {axis_angle} deg off the feature axis"
    offset = geo.coaxial_offset_mm(bore_pt, bore_dir, (0, -2, 60), (0, -1, 0))
    assert offset < 0.3, f"placed bore axis {offset} mm off the feature axis line"


def test_placed_knob_sits_on_shaft_small_separation(runner, knob_request, knob_parts, tmp_path):
    # A copy of the fixture store so the real part mesh (parts/p2.glb) is available, while the
    # runner's outputs land in tmp and never rewrite the committed fixtures.
    import shutil
    from pathlib import Path

    from screwit.storage.local import LocalStorage

    wt = Path(__file__).resolve().parents[3]
    shutil.copytree(wt / "fixtures" / "scans" / "scn_knob", tmp_path / "scans" / "scn_knob")
    store = LocalStorage(tmp_path)
    result = runner.run(knob_request, knob_parts, store)
    checks = _checks(result)
    sep = checks["minimum_surface_separation_mm"]
    assert sep.result == "pass"
    assert sep.value is not None and sep.value < 1.0, f"separation {sep.value} mm; knob not on the shaft"
    # Placed-geometry mate checks still pass for the hero knob.
    assert checks["mate_p2_f1_coaxial_offset_mm"].result == "pass"
    assert checks["mate_p2_f1_coaxial_offset_mm"].value < 0.3
    assert checks["mate_p2_f1_engagement_mm"].result == "pass"
    assert result.status == "ok"


def test_flat_orientation_fails_for_misrotated_placement():
    # Unit test of the flat-orientation mate check: a placement whose roll leaves the bore flat 90 deg
    # from the shaft flat must fail (catches a template/placement bug).
    from screwit.cad import checks as chk
    from screwit.cad.templates.base import MateInfo

    mate = MateInfo(kind="d_shaft", bore_axis_point=(0, 0, 0), bore_axis_dir=(0, 0, 1),
                    bore_radius_mm=3.12, engagement_mm=16.0, flat_normal=(0, -1, 0), flat_clearance_mm=0.12)
    # Identity placement: bore axis stays +Z but the shaft axis is [0,-1,0]; and bore flat -Y vs
    # shaft flat +Z are 90 deg apart.
    out = chk.mate_checks(
        mate, fit_radial_clearance_mm=0.12,
        shaft_axis_point_mm=(0, -2, 60), shaft_axis_dir_mm=(0, -1, 0),
        shaft_flat_normal_mm=(0, 0, 1), shaft_flat_depth_mm=1.5, shaft_length_mm=18.0,
        placement_quat=(0, 0, 0, 1), placement_translate=(0, 0, 0),
        feature_id="p2_f1", required=True, diameter_input_source="caliper",
        flat_input_source="user_confirmed",
    )
    by = {c.name: c for c in out}
    assert by["mate_p2_f1_flat_orientation_deg"].result == "fail"


# -- 6.0 shaft + 0.12 snug clearance -> 6.24 mm bore on the solid ---------

def test_bore_diameter_matches_shaft_plus_clearance():
    # Build the knob in its LOCAL frame (bore along +Z, centred on the origin) and measure the bore.
    import tempfile
    from pathlib import Path

    from cadquery import exporters

    from screwit.cad.templates import get_template

    t = get_template("knob")
    p = dict(t.defaults())
    p["shaft_diameter"] = 6.0
    p["shaft_flat_depth"] = 1.5
    br = t.build(p, mate_radial_clearance_mm=0.12)
    f = Path(tempfile.mkdtemp()) / "m.stl"
    exporters.export(br.solid, str(f))
    mesh = trimesh.load(str(f), force="mesh")
    v = mesh.vertices
    r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    bore_wall = r[(r > 2.0) & (r < 4.5)]
    assert bore_wall.size > 0, "no bore-wall vertices found"
    bore_radius = float(np.median(bore_wall))
    # shaft_radius 3.0 + radial clearance 0.12 = 3.12 mm -> 6.24 mm bore diameter.
    assert bore_radius == pytest.approx(3.12, abs=0.05), bore_radius
    assert 2 * bore_radius == pytest.approx(6.24, abs=0.1)


# -- codegen mode ---------------------------------------------------------

def _sandbox_or_skip():
    import subprocess

    from screwit.codegen.sandbox import DEFAULT_IMAGE, docker_available

    if not docker_available():
        pytest.skip("docker unavailable; codegen runs only inside the sandbox container")
    r = subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True, timeout=20)
    if r.returncode != 0:
        pytest.skip(f"sandbox image {DEFAULT_IMAGE} not built")


def _codegen_request(knob_request, **over):
    return CadRequest.model_validate({**knob_request.model_dump(), "mode": "codegen",
                                      "template": None,
                                      "codegen_prompt": "a finned knob 40 mm across with a 6 mm bore",
                                      **over})


def test_codegen_builds_in_sandbox_and_runs_checks(runner, knob_request, knob_parts, storage,
                                                   monkeypatch):
    _sandbox_or_skip()
    monkeypatch.setenv("SCREWIT_LLM", "fake")
    req = _codegen_request(knob_request)
    result = runner.run(req, knob_parts, storage)
    assert result.status in ("ok", "failed_checks"), result.error
    assert result.files is not None
    for key in (result.files.stl, result.files.step, result.files.source, result.files.preview):
        assert storage.head(key) is not None, key
    checks = {c.name: c for c in result.checks}
    assert checks["watertight"].result == "pass"
    assert checks["fits_build_volume"].result == "pass"
    # The hero shaft's radius is a caliper value and its flat depth is user-confirmed.
    assert checks["mating_dimensions_confirmed"].result == "pass"
    assert result.print_settings is not None


def test_codegen_mate_needs_confirmed_dimensions(runner, knob_request, knob_parts, storage,
                                                 monkeypatch):
    _sandbox_or_skip()
    monkeypatch.setenv("SCREWIT_LLM", "fake")
    shaft = knob_parts["p2"]
    feat = shaft.feature("p2_f1")
    scan_only = feat.radius.model_copy(update={"caliper_mm": None, "preferred": "scan"})
    parts = {**knob_parts, "p2": shaft.model_copy(update={"features": [
        feat.model_copy(update={"radius": scan_only})]})}
    result = runner.run(_codegen_request(knob_request), parts, storage)
    gate = next(c for c in result.checks if c.name == "mating_dimensions_confirmed")
    assert gate.kind == "required" and gate.result == "fail"
    assert "radius" in (gate.reason or "")
    assert result.status == "failed_checks"


def test_codegen_failure_is_a_clean_error(runner, knob_request, knob_parts, storage, monkeypatch):
    import screwit.codegen as cg

    def boom(*a, **k):
        raise RuntimeError("docker exited 125: no such image")

    monkeypatch.setattr(cg, "design", boom)
    result = runner.run(_codegen_request(knob_request), knob_parts, storage)
    assert result.status == "error"
    assert "no such image" in (result.error or "")
    assert result.files is None


# -- stale revision -------------------------------------------------------

def test_stale_reconstruction_revision_rejected(runner, knob_request, knob_parts, storage):
    req = CadRequest.model_validate({**knob_request.model_dump(), "reconstruction_revision": 2})
    with pytest.raises(StaleRequest):
        runner.run(req, knob_parts, storage)


def test_stale_measurement_revision_rejected(runner, knob_request, knob_parts, storage):
    req = CadRequest.model_validate({**knob_request.model_dump(),
        "part_measurement_revisions": {"p2": 99}})
    with pytest.raises(StaleRequest):
        runner.run(req, knob_parts, storage)


# -- clearance must equal the profile's for the fit type ------------------

def test_clearance_must_match_profile(runner, knob_request, knob_parts, storage):
    bad = knob_request.model_dump()
    bad["mates_with"][0]["radial_clearance_mm"] = 0.25  # sliding, not the snug 0.12 in the profile
    req = CadRequest.model_validate(bad)
    result = runner.run(req, knob_parts, storage)
    assert result.status == "error"
    assert "clearance" in result.error.lower()


def test_incompatible_feature_rejected(runner, knob_request, knob_parts, storage):
    # Point the mate at a plane feature (p1_f1), which a knob cannot mate.
    bad = knob_request.model_dump()
    bad["mates_with"][0]["part_id"] = "p1"
    bad["mates_with"][0]["feature_id"] = "p1_f1"
    bad["part_measurement_revisions"] = {"p1": 1}
    req = CadRequest.model_validate(bad)
    result = runner.run(req, knob_parts, storage)
    assert result.status == "error"


# -- templates() exposes JSON schema --------------------------------------

def test_templates_exposes_schema(runner):
    schemas = runner.templates()
    assert "knob" in schemas
    assert schemas["knob"]["properties"]["shaft_diameter"]["type"] == "number"
    assert len(schemas) == 8
