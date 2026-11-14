"""The offline agent on a synthetic NON-PLUMBING object (a wall shelf bracket).

Proves the agent is object-agnostic: it answers explain/diagnose/fix/buy/measure/make with no plumbing
vocabulary in the code path, grounded in scn_bracket's own scene graph and parts.
"""

from __future__ import annotations

from screwit.agent.offline import OfflineChatAgent, reset_state
from screwit.app.store import ScanStore
from screwit.contracts import CardEvent, DoneEvent, HighlightEvent, RepairPlanCard, TokenEvent

SCAN = "scn_bracket"


def _reply(reg, msg):
    reset_state()
    events = list(OfflineChatAgent(reg).reply(SCAN, msg, []))
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    return events, text


def test_synthetic_scan_loads(synthetic_registry):
    parts = ScanStore(synthetic_registry.data).list_parts(SCAN)
    assert {p.part_id for p in parts} == {"p1", "p2"}


def test_explain_uses_bracket_vocabulary(synthetic_registry):
    _, text = _reply(synthetic_registry, "explain how this works")
    assert "bracket" in text.lower()
    assert "trap" not in text.lower() and "washer" not in text.lower()


def test_diagnose_ranks_bracket_failure(synthetic_registry):
    events, text = _reply(synthetic_registry, "what's wrong with it")
    assert any(isinstance(e, HighlightEvent) for e in events)
    assert "crack" in text.lower() or "arm" in text.lower()


def test_fix_emits_contract_valid_repair_plan(synthetic_registry):
    events, _ = _reply(synthetic_registry, "how do I repair this bracket")
    plans = [e.card for e in events if isinstance(e, CardEvent) and isinstance(e.card, RepairPlanCard)]
    assert plans
    valid_ids = {p.part_id for p in ScanStore(synthetic_registry.data).list_parts(SCAN)}
    assert set(plans[0].payload.diagnosis.part_ids) <= valid_ids


def test_measure_reports_bracket_dimensions(synthetic_registry):
    _, text = _reply(synthetic_registry, "what are the dimensions")
    assert "mm" in text


def test_make_flow_prompts_then_builds_for_bracket(synthetic_registry):
    reset_state()
    agent = OfflineChatAgent(synthetic_registry)
    first = list(agent.reply(SCAN, "can you 3D print a replacement arm", []))
    first_text = "".join(e.text for e in first if isinstance(e, TokenEvent))
    assert "measure" in first_text.lower() or "calipers" in first_text.lower()
    assert not any(isinstance(e, CardEvent) for e in first)

    second = list(agent.reply(SCAN, "use the scan value", []))
    cards = [e.card for e in second if isinstance(e, CardEvent)]
    assert cards and cards[0].type == "cad_result"


def test_buy_without_catalogue_match_says_so(synthetic_registry):
    """A bracket has no catalogue match -> the agent says so rather than naming a wrong part."""
    events, text = _reply(synthetic_registry, "where can I buy a replacement bracket arm")
    cards = [e for e in events if isinstance(e, CardEvent)]
    assert not cards
    assert "couldn't find" in text.lower() or "did not find" in text.lower() or "measure" in text.lower()
    assert isinstance(events[-1], DoneEvent)
