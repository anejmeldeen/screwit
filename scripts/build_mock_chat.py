"""Compose the recorded hero conversations for mock mode from the other fixtures.

Writes fixtures/scans/<scan>/chat_script.json. Run after editing any fixture:
    backend/.venv/bin/python scripts/build_mock_chat.py
tests/test_fixtures.py fails if the committed scripts are out of date.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from screwit.contracts import CadResult, MockChatScript, RepairPlan  # noqa: E402

FIX = ROOT / "fixtures" / "scans"


def _tokens(text: str) -> list[dict]:
    # Split into a few chunks so the UI's streaming path is exercised.
    words = text.split(" ")
    chunks = [" ".join(words[i:i + 6]) for i in range(0, len(words), 6)]
    return [{"event": "token", "text": c + (" " if i < len(chunks) - 1 else "")} for i, c in enumerate(chunks)]


def trap_script() -> dict:
    plan = RepairPlan.model_validate_json((FIX / "scn_trap" / "repair_plan.json").read_text())
    return {
        "scan_id": "scn_trap",
        "turns": [
            {"match": r"(?i)leak|drip|water|wet",
             "events": [
                 {"event": "tool_call", "name": "get_scene_graph", "args": {}},
                 {"event": "highlight", "part_ids": ["p3", "p4"]},
                 *_tokens("The water marks sit right below the slip joint at [[part:p3]], so the most likely "
                          "cause is a worn or misaligned washer [[part:p4]] inside that nut. The pipe measures "
                          "about 37 mm across, which matches a standard 1-1/2 in kitchen trap; check the size "
                          "stamped on the nut before you buy."),
                 {"event": "card", "card": {"type": "repair_plan", "payload": plan.model_dump(mode="json")}},
                 {"event": "card", "card": {"type": "parts_to_buy",
                                            "payload": [b.model_dump(mode="json") for b in plan.bom]}},
                 {"event": "done", "message_id": "mock_trap_1"},
             ]},
            {"match": r"(?i)how.*work|what (is|does)",
             "events": [
                 {"event": "highlight", "part_ids": ["p2"]},
                 *_tokens("The J-bend [[part:p2]] always holds a little water, which blocks sewer gas from "
                          "coming up the drain. The slip nuts [[part:p3]] squeeze beveled washers against the "
                          "pipes to seal each joint, so the joints can be taken apart by hand for cleaning."),
                 {"event": "done", "message_id": "mock_trap_2"},
             ]},
            {"match": ".*",
             "events": [*_tokens("This is a recorded demo conversation. Try asking why it leaks, or how "
                                 "the trap works."),
                        {"event": "done", "message_id": "mock_trap_default"}]},
        ],
    }


def knob_script() -> dict:
    result = CadResult.model_validate_json((FIX / "scn_knob" / "cad_result.json").read_text())
    return {
        "scan_id": "scn_knob",
        "turns": [
            {"match": r"(?i)\b\d+(\.\d+)?\s*(mm)?\b",
             "events": [
                 {"event": "tool_call", "name": "set_measurement",
                  "args": {"part_id": "p2", "feature_id": "p2_f1", "field": "radius", "caliper_mm": 3.0}},
                 {"event": "tool_call", "name": "set_measurement",
                  "args": {"part_id": "p2", "feature_id": "p2_f1", "field": "flat_depth", "user_confirmed": True}},
                 {"event": "tool_call", "name": "create_cad", "args": {"template": "knob"}},
                 *_tokens("Here is a 32 mm knob with a D-bore for your 6 mm shaft. The bore has 0.12 mm of "
                          "clearance per side from your printer profile, every required check passes, and it "
                          "prints grip face down with no supports."),
                 {"event": "card", "card": {"type": "cad_result", "payload": result.model_dump(mode="json")}},
                 {"event": "done", "message_id": "mock_knob_2"},
             ]},
            {"match": r"(?i)knob|replace|print|make",
             "events": [
                 {"event": "tool_call", "name": "get_part", "args": {"part_id": "p2"}},
                 {"event": "highlight", "part_ids": ["p2"]},
                 *_tokens("I can design a replacement knob for this D-shaft [[part:p2]]. The scan puts the shaft "
                          "at about 7 mm across with a flat about 1.5 mm deep, but that is too small for the scan "
                          "to measure reliably, and a knob needs a snug fit. Could you measure the shaft "
                          "diameter with calipers, and confirm the flat depth?"),
                 {"event": "done", "message_id": "mock_knob_1"},
             ]},
            {"match": ".*",
             "events": [*_tokens("This is a recorded demo conversation. Try asking for a replacement knob."),
                        {"event": "done", "message_id": "mock_knob_default"}]},
        ],
    }


def build() -> dict[Path, str]:
    out = {}
    for script in (trap_script(), knob_script()):
        model = MockChatScript.model_validate(script)
        path = FIX / model.scan_id / "chat_script.json"
        out[path] = json.dumps(model.model_dump(mode="json"), indent=2) + "\n"
    return out


if __name__ == "__main__":
    for path, text in build().items():
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")
