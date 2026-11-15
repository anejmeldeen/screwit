"""Agent tool layer and tool-use loop with the fake LLM driver (SPEC 6.3)."""

from __future__ import annotations

from screwit.agent.cad import ConfirmBeforePrinting, check_confirm_before_printing
from screwit.agent.fake_llm import FakeDriver, ScriptTurn
from screwit.agent.loop import LlmChatAgent
from screwit.agent.tools import CardSignal, HighlightSignal, ToolContext
from screwit.contracts import CardEvent, DoneEvent, HighlightEvent, TokenEvent, ToolCallEvent


def _collect(agent, scan_id, msg):
    return list(agent.reply(scan_id, msg, []))


def test_list_parts_tool(registry):
    ctx = ToolContext(registry, "scn_trap")
    parts = ctx.dispatch("list_parts", {})
    assert {p["part_id"] for p in parts} == {"p1", "p2", "p3", "p4"}


def test_get_scene_graph_tool(registry):
    ctx = ToolContext(registry, "scn_trap")
    sg = ctx.dispatch("get_scene_graph", {})
    assert sg["object"]["category"] == "sink drain P-trap"


def test_highlight_signal(registry):
    ctx = ToolContext(registry, "scn_trap")
    sig = ctx.dispatch("highlight", {"part_ids": ["p3"]})
    assert isinstance(sig, HighlightSignal) and sig.part_ids == ["p3"]


def test_search_parts_sizes_from_measurement(registry):
    ctx = ToolContext(registry, "scn_trap")
    hits = ctx.dispatch("search_parts", {"query": "slip washer", "measurement_mm": 38.1})
    assert hits
    assert any("1-1/2" in h["name"] for h in hits)


def test_confirm_before_printing_refuses_scan_source(registry):
    """p3 bore radius (p3_f2) is snapped -> a mate on it must be refused before create_cad."""
    ctx = ToolContext(registry, "scn_trap")
    result = ctx.dispatch("create_cad", {
        "template": "cap",
        "params": {"inner_diameter": 38.1},
        "mates_with": [{"part_id": "p3", "feature_id": "p3_f2", "fit": "snug"}],
    })
    assert result["error"] == "confirm_before_printing"
    assert any(o["source"] == "snapped" for o in result["offenders"])


def test_create_cad_succeeds_when_caliper_sourced(registry):
    """scn_knob p2_f1 radius is caliper and flat_depth user_confirmed -> create_cad is allowed."""
    ctx = ToolContext(registry, "scn_knob")
    result = ctx.dispatch("create_cad", {
        "template": "knob",
        "params": {"shaft_diameter": 6.0},
        "mates_with": [{"part_id": "p2", "feature_id": "p2_f1", "fit": "snug",
                        "radial_clearance_mm": 0.12, "fields": ["radius", "flat_depth"]}],
    })
    assert isinstance(result, CardSignal)
    assert result.card.payload.status == "ok"


def test_create_cad_codegen_builds_and_stores_prompt(registry):
    """mode='codegen' with a codegen_prompt produces a CAD card and stores a codegen CadRequest."""
    from screwit.app.store import ScanStore

    ctx = ToolContext(registry, "scn_knob")
    result = ctx.dispatch("create_cad", {
        "mode": "codegen",
        "codegen_prompt": "A finned knurled knob, 40 mm across and 25 mm tall, with a 6 mm D-bore.",
        "mates_with": [{"part_id": "p2", "feature_id": "p2_f1", "fit": "snug",
                        "fields": ["radius", "flat_depth"]}],
    })
    assert isinstance(result, CardSignal)
    assert result.card.type == "cad_result"

    job = ScanStore(registry.data).get_cad_job("scn_knob", result.card.payload.cad_id)
    assert job is not None
    req, _ = job
    assert req.mode == "codegen"
    assert req.template is None
    assert "D-bore" in req.codegen_prompt


def test_create_cad_codegen_requires_prompt(registry):
    ctx = ToolContext(registry, "scn_knob")
    result = ctx.dispatch("create_cad", {"mode": "codegen"})
    assert result["error"] == "bad_request"


def test_create_cad_codegen_still_confirms_mating_dims(registry):
    """Confirm-before-printing applies in codegen mode too: a scan/snapped mate is refused."""
    ctx = ToolContext(registry, "scn_trap")  # p3_f2 bore radius is snapped
    result = ctx.dispatch("create_cad", {
        "mode": "codegen",
        "codegen_prompt": "A custom cap that seals the slip nut bore.",
        "mates_with": [{"part_id": "p3", "feature_id": "p3_f2", "fit": "snug"}],
    })
    assert result["error"] == "confirm_before_printing"
    assert any(o["source"] == "snapped" for o in result["offenders"])


def test_create_cad_template_still_requires_template_name(registry):
    ctx = ToolContext(registry, "scn_knob")
    result = ctx.dispatch("create_cad", {"mode": "template"})
    assert result["error"] == "bad_request"


def test_check_confirm_helper_direct(registry):
    from screwit.app.store import ScanStore

    parts = {p.part_id: p for p in ScanStore(registry.data).list_parts("scn_knob")}
    # caliper + user_confirmed fields pass; a scan-only field (length) is refused.
    check_confirm_before_printing(parts, [("p2", "p2_f1", "radius")])  # no raise
    try:
        check_confirm_before_printing(parts, [("p2", "p2_f1", "length")])
        raised = False
    except ConfirmBeforePrinting:
        raised = True
    assert raised


def test_loop_streams_tokens_tool_cards_and_done(registry):
    driver = FakeDriver([
        ScriptTurn(text="", tool_calls=[("highlight", {"part_ids": ["p2"]})]),
        ScriptTurn(text="Here is your knob.", tool_calls=[
            ("create_cad", {"template": "knob", "params": {"shaft_diameter": 6.0},
                            "mates_with": [{"part_id": "p2", "feature_id": "p2_f1", "fit": "snug",
                                            "fields": ["radius", "flat_depth"]}]})]),
        ScriptTurn(text="Every required check passes."),
    ])
    agent = LlmChatAgent(registry, driver)
    events = _collect(agent, "scn_knob", "make me a knob, shaft is 6 mm")
    kinds = [type(e) for e in events]
    assert HighlightEvent in kinds
    assert ToolCallEvent in kinds
    assert CardEvent in kinds
    assert TokenEvent in kinds
    assert isinstance(events[-1], DoneEvent)


def test_loop_turn_cap_still_terminates(registry):
    # A driver that always wants a tool call must still end (turn cap) with a DoneEvent.
    class Loop:
        def run_turn(self, system, messages, tools):
            from screwit.agent.driver import ToolUseStep
            yield ToolUseStep(id="t", name="list_parts", input={})

    agent = LlmChatAgent(registry, Loop())
    events = _collect(agent, "scn_trap", "hi")
    assert isinstance(events[-1], DoneEvent)


def test_loop_codegen_for_new_shape(registry):
    """Full fake-LLM loop: the model chooses mode='codegen' for a shape no template covers."""
    driver = FakeDriver([
        ScriptTurn(text="", tool_calls=[("create_cad", {
            "mode": "codegen",
            "codegen_prompt": "A clover-shaped finger grip, 45 mm across, 20 mm tall, 6 mm D-bore.",
            "mates_with": [{"part_id": "p2", "feature_id": "p2_f1", "fields": ["radius", "flat_depth"]}],
        })]),
        ScriptTurn(text="Designed a custom grip; checks pass."),
    ])
    agent = LlmChatAgent(registry, driver)
    events = _collect(agent, "scn_knob", "design me a clover-shaped grip for this shaft")
    cards = [e.card for e in events if isinstance(e, CardEvent)]
    assert cards and cards[0].type == "cad_result"


def test_tool_schema_exposes_codegen():
    from screwit.agent.tools import anthropic_tool_specs

    spec = next(t for t in anthropic_tool_specs() if t["name"] == "create_cad")
    props = spec["input_schema"]["properties"]
    assert props["mode"]["enum"] == ["template", "codegen"]
    assert "codegen_prompt" in props
    assert "codegen" in spec["description"]
    # template is no longer a required field (codegen omits it)
    assert "required" not in spec["input_schema"] or "template" not in spec["input_schema"]["required"]


def test_sink_manual_targets_named_part(registry):
    from screwit.agent.offline import OfflineChatAgent

    events = _collect(OfflineChatAgent(registry), "scn_sink", "Give me a repair manual for the P-trap")
    plan = next(e.card.payload for e in events if isinstance(e, CardEvent) and e.card.type == "repair_plan")
    assert plan.diagnosis.part_ids == ["p4"]
    assert any(step.part_ids == ["p4"] for step in plan.steps)
    assert {"p3", "p4", "p5"} == set(plan.steps[1].part_ids)
    assert len(plan.bom) == 1
    assert plan.tools and plan.safety


def test_text_after_a_tool_call_starts_a_new_paragraph(registry):
    # Streamed as is, "...the parts." and "Here's what you have" ran together after a tool call.
    driver = FakeDriver([
        ScriptTurn(text="I'll get you a detailed look at the parts.", tool_calls=[("list_parts", {})]),
        ScriptTurn(text="Here's what you have.", tool_calls=[("list_parts", {})]),
        ScriptTurn(text="That is all."),
    ])
    events = _collect(LlmChatAgent(registry, driver), "scn_knob", "what are the parts?")
    text = "".join(e.text for e in events if isinstance(e, TokenEvent))
    assert text == "I'll get you a detailed look at the parts.\n\nHere's what you have.\n\nThat is all."


def test_no_break_before_the_first_text_of_a_reply(registry):
    driver = FakeDriver([
        ScriptTurn(text="", tool_calls=[("list_parts", {})]),
        ScriptTurn(text="Two parts."),
    ])
    events = _collect(LlmChatAgent(registry, driver), "scn_knob", "what are the parts?")
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "Two parts."
