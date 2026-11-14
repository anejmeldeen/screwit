"""Every fixture validates against its contract, and generated files are up to date."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from screwit.contracts import (
    CadRequest,
    CadResult,
    DemoScan,
    MockChatScript,
    Part,
    PrinterProfile,
    Reconstruction,
    RepairPlan,
    Scan,
    SceneGraph,
    measurement,
)
from screwit.contracts.sse import decode_chat_event, encode_chat_event

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures"

FILES = {
    "scan.json": Scan,
    "reconstruction.json": Reconstruction,
    "scene_graph.json": SceneGraph,
    "repair_plan.json": RepairPlan,
    "cad_request.json": CadRequest,
    "cad_result.json": CadResult,
    "chat_script.json": MockChatScript,
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("scan_dir", sorted(p for p in (FIX / "scans").iterdir() if p.is_dir()),
                         ids=lambda p: p.name)
def test_scan_fixtures_validate(scan_dir: Path):
    for fname, model in FILES.items():
        f = scan_dir / fname
        if f.exists():
            model.model_validate_json(f.read_text())
    parts = TypeAdapter(list[Part]).validate_json((scan_dir / "parts.json").read_text())
    graph = SceneGraph.model_validate_json((scan_dir / "scene_graph.json").read_text())
    assert sorted(graph.parts) == sorted(p.part_id for p in parts)
    scan = Scan.model_validate_json((scan_dir / "scan.json").read_text())
    for p in parts:
        assert p.reconstruction_revision == scan.reconstruction_revision


def test_cad_request_matches_parts():
    d = FIX / "scans" / "scn_knob"
    req = CadRequest.model_validate_json((d / "cad_request.json").read_text())
    parts = {p.part_id: p for p in TypeAdapter(list[Part]).validate_json((d / "parts.json").read_text())}
    for mate in req.mates_with:
        part = parts[mate.part_id]
        assert part.feature(mate.feature_id) is not None
        assert req.part_measurement_revisions[mate.part_id] == part.measurement_revision
    shaft = parts["p2"].feature("p2_f1")
    assert req.params["shaft_diameter"] == pytest.approx(2 * shaft.radius.value_mm)
    assert req.param_sources["shaft_diameter"] == shaft.radius.param_source == "caliper"
    assert req.param_sources["shaft_flat_depth"] == shaft.flat_depth.param_source == "user_confirmed"


def test_other_fixtures():
    TypeAdapter(list[DemoScan]).validate_json((FIX / "demo_scans.json").read_text())
    for f in (ROOT / "profiles").glob("*.json"):
        PrinterProfile.model_validate_json(f.read_text())


def test_generated_files_up_to_date():
    for script in ("build_mock_chat", "export_schemas"):
        for path, text in _load_script(script).build().items():
            assert path.exists() and path.read_text() == text, (
                f"{path.relative_to(ROOT)} is stale; run scripts/{script}.py")


def test_sse_round_trip():
    script = MockChatScript.model_validate_json((FIX / "scans" / "scn_trap" / "chat_script.json").read_text())
    for turn in script.turns:
        for ev in turn.events:
            frame = encode_chat_event(ev)
            name = frame.split("\n")[0].removeprefix("event: ")
            data = frame.split("\n")[1].removeprefix("data: ")
            assert decode_chat_event(name, data) == ev
            if ev.event == "card":
                assert set(json.loads(data)) == {"type", "payload"}  # SPEC 4.8 card shape


def test_measurement_rules():
    assert measurement(10.0).preferred == "scan"
    assert measurement(10.0, snapped_mm=9.5, snap_rule="r").preferred == "snapped"
    m = measurement(10.0, snapped_mm=9.5, snap_rule="r", caliper_mm=9.6)
    assert m.preferred == "caliper" and m.value_mm == 9.6 and m.param_source == "caliper"
    assert measurement(10.0, user_confirmed=True).param_source == "user_confirmed"
    with pytest.raises(ValidationError):
        measurement(10.0).model_copy(update={}).model_validate({**measurement(10.0).model_dump(),
                                                                 "preferred": "caliper"})
    with pytest.raises(ValidationError):
        Part.model_validate({**json.loads((FIX / "scans/scn_knob/parts.json").read_text())[1],
                             "dimensions": [{"name": "x", "derived_from": "nope.radius"}]})
