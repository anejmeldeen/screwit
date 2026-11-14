"""Hero conversation shape, checked against the recorded mock scripts (no server needed).

These assertions encode what the live chat SSE must also produce (test_routes checks the same
shapes over HTTP once the server is up). They decode every frame through contracts.sse so the
wire encoding and the fixtures are both exercised, and they pass today because they depend only
on the integration lead's fixtures and contracts.
"""

from __future__ import annotations

import re

import pytest
from pydantic import TypeAdapter

from screwit.contracts import MockChatScript
from screwit.contracts.sse import decode_chat_event, encode_chat_event

from .conftest import FIXTURES_DIR

CardList = TypeAdapter(list)

pytestmark = pytest.mark.skipif(
    not (FIXTURES_DIR / "scans").is_dir(),
    reason=f"fixtures dir not found at {FIXTURES_DIR}; set SCREWIT_MAIN_TREE",
)


def _script(scan_id: str) -> MockChatScript:
    path = FIXTURES_DIR / "scans" / scan_id / "chat_script.json"
    return MockChatScript.model_validate_json(path.read_text())


def _first_matching_turn(script: MockChatScript, message: str):
    for turn in script.turns:
        if re.search(turn.match, message):
            return turn
    raise AssertionError(f"no turn matched {message!r}")  # last turn is '.*', so unreachable


def _roundtrip(events: list) -> list:
    """Encode every event to the SPEC 4.8 wire format and decode it back, as the server/client will."""
    out = []
    for ev in events:
        frame = encode_chat_event(ev)
        head, body = frame.split("\n", 1)
        name = head.removeprefix("event: ")
        data = body.split("data: ", 1)[1].split("\n\n", 1)[0]
        out.append(decode_chat_event(name, data))
    return out


def _card_types(events: list) -> list[str]:
    return [ev.card.type for ev in events if ev.event == "card"]


def test_trap_leak_turn_ends_with_repair_and_parts_cards():
    script = _script("scn_trap")
    turn = _first_matching_turn(script, "it's leaking under here")
    events = _roundtrip(turn.events)

    assert events[-1].event == "done"
    assert _card_types(events) == ["repair_plan", "parts_to_buy"], (
        "the trap leak conversation must end with a repair_plan then a parts_to_buy card (SPEC 9 step 4)"
    )

    repair = next(ev.card.payload for ev in events if ev.event == "card" and ev.card.type == "repair_plan")
    assert len(repair.steps) == 4, "trap repair plan has 4 numbered steps (brief/P1)"
    assert [s.n for s in repair.steps] == [1, 2, 3, 4]
    assert repair.bom, "a repair plan must list parts to buy"

    parts = next(ev.card.payload for ev in events if ev.event == "card" and ev.card.type == "parts_to_buy")
    assert parts, "parts_to_buy card must list at least one BOM line"
    assert all(line.buy_links for line in parts), "every BOM line needs a buy link"


def test_knob_asks_for_shaft_then_emits_cad_result():
    script = _script("scn_knob")

    ask = _first_matching_turn(script, "make me a replacement knob")
    ask_events = _roundtrip(ask.events)
    assert _card_types(ask_events) == [], "the knob ask-first turn emits no card yet"
    asked = " ".join(ev.text for ev in ask_events if ev.event == "token").lower()
    assert "shaft" in asked and ("measure" in asked or "calipers" in asked), (
        "the knob conversation must ask the user for the shaft size first (SPEC 6.3 confirm-before-printing)"
    )

    reply = _first_matching_turn(script, "6 mm")
    reply_events = _roundtrip(reply.events)
    assert _card_types(reply_events) == ["cad_result"], (
        "after a numeric shaft reply the knob conversation emits a cad_result card (SPEC 9 step 5)"
    )
    cad = next(ev.card.payload for ev in reply_events if ev.event == "card").model_copy()
    assert cad.status == "ok" and not cad.stale
    required = [c for c in cad.checks if c.kind == "required"]
    assert required and all(c.result == "pass" for c in required), (
        "every required CAD check must pass on the knob result"
    )
    assert any(ev.event == "tool_call" and ev.name == "create_cad" for ev in reply_events)
