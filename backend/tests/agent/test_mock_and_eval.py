"""Offline agent, factory routing, and the grounded evaluation set (SPEC 7 P4 #9, 8, AGENTS.md).

The offline agent answers ANY phrasing about ANY object from the scan's own data: no scripted keywords,
no invented dimensions, contract-valid cards. These tests drive >25 varied phrasings per hero scan plus
a synthetic non-plumbing scan, and check the factory's mode routing and offline-on-no-key fallback.
"""

from __future__ import annotations

import os

import pytest

from screwit.agent.eval import ALL_CASES, KNOB_CASES, TRAP_CASES, run_case
from screwit.agent.factory import build_chat_agent
from screwit.agent.offline import OfflineChatAgent, reset_state
from screwit.app.store import ScanStore
from screwit.contracts import (
    CardEvent,
    DoneEvent,
    HighlightEvent,
    PartsToBuyCard,
    RepairPlanCard,
    TokenEvent,
)


# ---------------------------------------------------------------- factory routing
def test_offline_agent_is_default_without_llm(registry):
    assert "SCREWIT_LLM" not in os.environ
    assert isinstance(build_chat_agent(registry), OfflineChatAgent)


def test_replay_mode_behind_flag(registry):
    from screwit.agent.mock import MockReplayChatAgent

    os.environ["SCREWIT_CHAT_REPLAY"] = "1"
    try:
        assert isinstance(build_chat_agent(registry), MockReplayChatAgent)
    finally:
        os.environ.pop("SCREWIT_CHAT_REPLAY", None)


def test_anthropic_without_key_falls_back_to_offline(make_registry):
    """SCREWIT_LLM=anthropic but no ANTHROPIC_API_KEY -> offline agent answers and says it is offline."""
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["SCREWIT_MODEL"] = "claude-3-5-sonnet"  # required by llm.config for a non-fake provider
    reg = make_registry(llm="anthropic")
    try:
        reset_state()
        agent = build_chat_agent(reg)
        events = list(agent.reply("scn_trap", "what's wrong with it", []))
    finally:
        os.environ.pop("SCREWIT_MODEL", None)
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert "offline" in text.lower()
    assert "washer" in text.lower() or "slip" in text.lower()  # still a grounded answer
    assert isinstance(events[-1], DoneEvent)


# ---------------------------------------------------------------- offline grounded behaviour
def test_offline_default_lists_capabilities(registry):
    reset_state()
    events = list(OfflineChatAgent(registry).reply("scn_trap", "xkcd zzz", []))
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert "explain" in text.lower() and "repair" in text.lower()
    assert isinstance(events[-1], DoneEvent)


def test_offline_missing_scan_does_not_crash(registry):
    reset_state()
    events = list(OfflineChatAgent(registry).reply("no_such_scan", "help", []))
    assert isinstance(events[-1], DoneEvent)


def test_offline_make_flow_confirms_then_builds(registry):
    reset_state()
    agent = OfflineChatAgent(registry)
    first = list(agent.reply("scn_knob", "can you 3D print a replacement knob", []))
    first_text = "".join(e.text for e in first if isinstance(e, TokenEvent))
    assert any(isinstance(e, HighlightEvent) for e in first)
    assert "measure" in first_text.lower() or "calipers" in first_text.lower()
    assert not any(isinstance(e, CardEvent) for e in first)  # no CAD until confirmed

    second = list(agent.reply("scn_knob", "the exposed length is 18 mm", []))
    cards = [e.card for e in second if isinstance(e, CardEvent)]
    assert cards and cards[0].type == "cad_result"
    assert cards[0].payload.status in {"ok", "failed_checks"}


def test_offline_cannot_design_new_shape_offers_closest_template(registry):
    """Offline cannot codegen a new shape: it says so and names the closest template (AGENTS.md)."""
    reset_state()
    events = list(OfflineChatAgent(registry).reply("scn_knob", "design me a custom clover-shaped grip", []))
    text = "".join(e.text for e in events if isinstance(e, TokenEvent)).lower()
    assert "custom" in text and ("offline" in text or "online" in text)
    assert "knob" in text  # the closest standard template for a shaft
    assert not any(isinstance(e, CardEvent) for e in events)  # no CAD produced offline


def test_offline_ordinary_make_is_not_treated_as_custom(registry):
    """A plain 'make a replacement knob' still runs the template confirm flow, not the custom refusal."""
    reset_state()
    events = list(OfflineChatAgent(registry).reply("scn_knob", "make me a replacement knob", []))
    text = "".join(e.text for e in events if isinstance(e, TokenEvent)).lower()
    assert "measure" in text or "calipers" in text
    assert "can't design a brand-new" not in text


def test_offline_buy_returns_contract_valid_card(registry):
    reset_state()
    events = list(OfflineChatAgent(registry).reply("scn_trap", "where can I buy a new washer", []))
    cards = [e.card for e in events if isinstance(e, CardEvent)]
    if cards:  # a catalogue match -> a PartsToBuyCard with real lines
        assert isinstance(cards[0], PartsToBuyCard)
        assert cards[0].payload and all(line.buy_links for line in cards[0].payload)


def test_offline_repair_plan_is_contract_valid_and_grounded(registry):
    reset_state()
    events = list(OfflineChatAgent(registry).reply("scn_trap", "how do I repair this leak", []))
    plans = [e.card for e in events if isinstance(e, CardEvent) and isinstance(e.card, RepairPlanCard)]
    assert plans
    plan = plans[0].payload
    valid_ids = {p.part_id for p in ScanStore(registry.data).list_parts("scn_trap")}
    for pid in plan.diagnosis.part_ids:
        assert pid in valid_ids  # no invented part ids


def test_offline_cites_only_measured_numbers(registry):
    """Every mm figure in a measurement answer must come from a real Measurement value."""
    import re

    reset_state()
    events = list(OfflineChatAgent(registry).reply("scn_trap", "what are the dimensions", []))
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    quoted = {round(float(x), 1) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mm", text)}
    real: set[float] = set()
    for part in ScanStore(registry.data).list_parts("scn_trap"):
        for feat in part.features:
            for fld in ("radius", "length", "depth", "flat_depth", "extent_u", "extent_v"):
                m = getattr(feat, fld, None)
                if m is not None:
                    real.add(round(m.value_mm, 1))
                    real.add(round(m.value_mm * 2.0, 1))  # diameters are 2 * radius
        for d in part.dimensions:
            if hasattr(d, "value"):
                real.add(round(d.value.value_mm, 1))
    assert quoted <= real, f"unmeasured figures: {quoted - real}"


# ---------------------------------------------------------------- the evaluation set (>25 per scan)
def test_eval_set_sizes():
    assert len(TRAP_CASES) >= 25
    assert len(KNOB_CASES) >= 25
    assert len(ALL_CASES) == len(TRAP_CASES) + len(KNOB_CASES)


@pytest.mark.parametrize("case", TRAP_CASES, ids=lambda c: c.question[:28])
def test_eval_trap_offline(registry, case):
    outcome = run_case(OfflineChatAgent(registry), case)
    assert outcome.passed, f"{outcome.reason} | reply={outcome.reply_text!r}"


@pytest.mark.parametrize("case", KNOB_CASES, ids=lambda c: c.question[:28])
def test_eval_knob_offline(registry, case):
    outcome = run_case(OfflineChatAgent(registry), case)
    assert outcome.passed, f"{outcome.reason} | reply={outcome.reply_text!r}"


@pytest.mark.network
def test_eval_against_real_llm(make_registry):
    """Run the eval set against a real LLM when SCREWIT_LLM is configured. Skipped by default."""
    from screwit.agent.anthropic_driver import AnthropicDriver
    from screwit.agent.loop import LlmChatAgent
    from screwit.llm import config

    if config().provider == "fake":
        pytest.skip("SCREWIT_LLM not configured for a live model")
    reg = make_registry()
    agent = LlmChatAgent(reg, AnthropicDriver())
    failures = [o for o in (run_case(agent, c) for c in ALL_CASES) if not o.passed]
    assert not failures, [f"{o.case.question}: {o.reason}" for o in failures]
