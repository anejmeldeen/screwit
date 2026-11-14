"""Every template builds a valid solid at min / default / max (SPEC 6.4, P5 workload 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import trimesh
from cadquery import exporters

from screwit.cad.templates import TEMPLATES, get_template
from screwit.cad.templates.base import Template

ALL = sorted(TEMPLATES)


def _mesh(build_result) -> trimesh.Trimesh:
    f = Path(tempfile.mkdtemp()) / "m.stl"
    exporters.export(build_result.solid, str(f))
    return trimesh.load(str(f), force="mesh")


def _params_at(template: Template, which: str) -> dict[str, float]:
    if which == "default":
        return template.defaults()
    if which == "min":
        return {p.name: p.min for p in template.params}
    return {p.name: p.max for p in template.params}


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("which", ["min", "default", "max"])
def test_template_builds_valid_solid(name, which):
    t = get_template(name)
    params = _params_at(t, which)
    clr = 0.12 if t.mates else None
    result = t.build(dict(params), mate_radial_clearance_mm=clr)
    mesh = _mesh(result)
    assert len(mesh.faces) > 0, f"{name} {which} produced no geometry"
    assert mesh.volume > 0, f"{name} {which} has non-positive volume"
    assert mesh.is_watertight, f"{name} {which} is not watertight"


@pytest.mark.parametrize("name", ALL)
def test_template_rejects_unknown_param(name):
    t = get_template(name)
    with pytest.raises(ValueError):
        t.validate_params({"not_a_real_param": 1.0})


@pytest.mark.parametrize("name", ALL)
def test_template_rejects_out_of_range(name):
    t = get_template(name)
    spec = t.params[0]
    with pytest.raises(ValueError):
        t.validate_params({spec.name: spec.max + 1000.0})
    with pytest.raises(ValueError):
        t.validate_params({spec.name: spec.min - 1000.0})


@pytest.mark.parametrize("name", ALL)
def test_template_defaults_in_range(name):
    t = get_template(name)
    for p in t.params:
        assert p.min <= p.default <= p.max, f"{name}.{p.name} default out of its own range"


def test_mating_templates_expose_mate_info():
    for name in ("knob", "cap", "pipe_adapter"):
        t = get_template(name)
        assert t.mates
        clr = 0.12
        result = t.build(t.defaults(), mate_radial_clearance_mm=clr)
        assert result.mate is not None
        # The round bore radius is shaft/inlet radius + clearance.
        assert result.mate.bore_radius_mm > 0


def test_knob_d_bore_radius_follows_clearance():
    t = get_template("knob")
    p = dict(t.defaults())
    p["shaft_diameter"] = 6.0
    p["shaft_flat_depth"] = 1.5
    r0 = t.build(dict(p), mate_radial_clearance_mm=0.0).mate.bore_radius_mm
    r12 = t.build(dict(p), mate_radial_clearance_mm=0.12).mate.bore_radius_mm
    assert r0 == pytest.approx(3.0, abs=1e-6)
    assert r12 == pytest.approx(3.12, abs=1e-6)


def test_round_shaft_knob_has_cylinder_mate():
    t = get_template("knob")
    p = dict(t.defaults())
    p["shaft_flat_depth"] = 0.0
    result = t.build(dict(p), mate_radial_clearance_mm=0.12)
    assert result.mate.kind == "cylinder"
    assert result.mate.flat_normal is None


def test_unknown_template_raises():
    with pytest.raises(ValueError):
        get_template("does_not_exist")


# -- defect 2: knurl defaults are printable; unprintable knurls are rejected ---

def test_knob_auto_knurl_is_printable():
    from pathlib import Path

    from screwit.cad import checks as chk
    from screwit.contracts import PrinterProfile

    wt = Path(__file__).resolve().parents[3]
    profile = PrinterProfile.model_validate_json(
        (wt / "profiles" / "prusa_mk4_petg_0.2.json").read_text()
    )
    t = get_template("knob")
    # Auto knurl (count/depth 0) must yield a wall >= the profile minimum.
    result = t.build(t.defaults(), mate_radial_clearance_mm=0.12, profile=profile)
    mesh = _mesh(result)
    wall = chk.min_wall_check(mesh, profile)
    assert wall.result == "pass", f"auto knurl min wall {wall.value} < {wall.limit}"


def test_knob_rejects_unprintable_knurl():
    from pathlib import Path

    from screwit.contracts import PrinterProfile

    wt = Path(__file__).resolve().parents[3]
    profile = PrinterProfile.model_validate_json(
        (wt / "profiles" / "prusa_mk4_petg_0.2.json").read_text()
    )
    t = get_template("knob")
    # Too many grooves on a 32 mm knob leaves ridge lands thinner than min_wall -> rejected.
    p = dict(t.defaults())
    p["knurl_count"] = 100
    p["knurl_depth"] = 0.8
    with pytest.raises(ValueError, match="ridge land"):
        t.build(p, mate_radial_clearance_mm=0.12, profile=profile)


def test_knob_resolve_knurl_lands_meet_min_wall():
    from pathlib import Path

    from screwit.contracts import PrinterProfile

    wt = Path(__file__).resolve().parents[3]
    profile = PrinterProfile.model_validate_json(
        (wt / "profiles" / "prusa_mk4_petg_0.2.json").read_text()
    )
    t = get_template("knob")
    import math

    n, d = t.resolve_knurl(32.0, 0, 0, profile)
    assert n > 0 and d > 0
    land = math.pi * 32.0 / n - 1.5 * profile.nozzle_mm
    assert land >= profile.min_wall_mm - 1e-6


def test_load_bearing_flags():
    assert get_template("l_bracket").build(get_template("l_bracket").defaults()).carries_load
    assert get_template("hook").build(get_template("hook").defaults()).carries_load
    assert not get_template("knob").build(get_template("knob").defaults(),
                                          mate_radial_clearance_mm=0.12).carries_load
