"""Agent evaluation set (SPEC 7 P4 #9): varied phrasings per scan with expected grounded facts.

Each case asks a question and lists `expect_any` regexes (at least one must appear in the assembled
reply text) and optional `expect_events` (event names that must be emitted, e.g. a highlight or a card).
run_case drives any ChatAgent and reports pass/fail. The sets run against the deterministic offline
agent in tests, and the same cases run against a real LLM under @pytest.mark.network.

The expectations are object-grounded facts (a part label, a measured size, a card type), never a
scripted sentence, so they hold for both the offline router and a real model answering from the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from screwit.contracts import (
    CardEvent,
    DoneEvent,
    HighlightEvent,
    TokenEvent,
    ToolCallEvent,
)


@dataclass
class EvalCase:
    scan_id: str
    question: str
    expect_any: list[str] = field(default_factory=list)  # regexes; at least one must match the reply
    expect_events: list[str] = field(default_factory=list)  # event names that must appear


@dataclass
class EvalOutcome:
    case: EvalCase
    passed: bool
    reply_text: str
    events_seen: list[str]
    reason: str = ""


# Scenario A: leaking kitchen P-trap. Expectations are grounded facts from scn_trap's own data.
TRAP_CASES: list[EvalCase] = [
    EvalCase("scn_trap", "explain this to me", [r"P-trap|sewer gas|seal"]),
    EvalCase("scn_trap", "how does this work?", [r"P-trap|sewer gas|water"]),
    EvalCase("scn_trap", "what is this object?", [r"P-trap|trap"]),
    EvalCase("scn_trap", "tell me about this", [r"trap|part"]),
    EvalCase("scn_trap", "describe the parts", [r"tailpiece|J-bend|slip"]),
    EvalCase("scn_trap", "what is the J-bend for", [r"J-bend|water|sewer gas"], ["highlight"]),
    EvalCase("scn_trap", "what does the slip nut do", [r"slip nut|nut"], ["highlight"]),
    EvalCase("scn_trap", "what's wrong with it", [r"washer|slip|likely"], ["highlight"]),
    EvalCase("scn_trap", "why is it broken", [r"washer|slip|likely"], ["highlight"]),
    EvalCase("scn_trap", "it won't stop dripping", [r"washer|slip|repair|likely"]),
    EvalCase("scn_trap", "there is water under the sink", [r"washer|slip|repair|likely"]),
    EvalCase("scn_trap", "diagnose the problem", [r"washer|slip|likely"], ["highlight"]),
    EvalCase("scn_trap", "how do I repair this", [r"repair plan|repair|replace"], ["card"]),
    EvalCase("scn_trap", "how do I fix the leak", [r"repair|replace|plan"], ["card"]),
    EvalCase("scn_trap", "give me a repair plan", [r"repair|plan|replace"], ["card"]),
    EvalCase("scn_trap", "what do I do to mend it", [r"repair|replace|plan"], ["card"]),
    EvalCase("scn_trap", "where can I buy a new washer", [r"washer|match|part|catalogue|measure"]),
    EvalCase("scn_trap", "where do I get a replacement part", [r"part|washer|match|measure|catalogue"]),
    EvalCase("scn_trap", "I need to purchase a slip washer", [r"washer|part|match|measure"]),
    EvalCase("scn_trap", "how big is the slip nut", [r"mm|caliper|scan|snapped"], ["highlight"]),
    EvalCase("scn_trap", "what size is the tailpiece", [r"mm|diameter|scan|snapped|caliper"]),
    EvalCase("scn_trap", "what are the dimensions", [r"mm"]),
    EvalCase("scn_trap", "measure the pipe", [r"mm"]),
    EvalCase("scn_trap", "what is [[part:p4]]", [r"washer"], ["highlight"]),
    EvalCase("scn_trap", "tell me about the tailpiece", [r"tailpiece"], ["highlight"]),
    EvalCase("scn_trap", "can you 3D print a replacement", [r"print|measure|calipers|dimension|replace"]),
    EvalCase("scn_trap", "qwerty asdf zxcv", [r"help|explain|diagnose|repair|part"]),
]

# Scenario B: missing appliance knob -- a NON-PLUMBING hero object.
KNOB_CASES: list[EvalCase] = [
    EvalCase("scn_knob", "explain this to me", [r"panel|shaft|control"]),
    EvalCase("scn_knob", "how does this work?", [r"shaft|control|turn"]),
    EvalCase("scn_knob", "what is this object?", [r"control panel|panel|shaft"]),
    EvalCase("scn_knob", "describe the parts", [r"shaft|panel"]),
    EvalCase("scn_knob", "what is the D-shaft", [r"shaft"], ["highlight"]),
    EvalCase("scn_knob", "what's wrong with it", [r"knob|missing|shaft|likely"], ["highlight"]),
    EvalCase("scn_knob", "diagnose the problem", [r"knob|missing|likely"], ["highlight"]),
    EvalCase("scn_knob", "the knob fell off", [r"knob|missing|shaft|repair|likely|print"]),
    EvalCase("scn_knob", "how do I repair this", [r"repair|replace|plan|print"]),
    EvalCase("scn_knob", "give me a repair plan", [r"repair|plan|replace|print"]),
    EvalCase("scn_knob", "can you 3D print a replacement knob", [r"print|measure|calipers|shaft|knob"], ["highlight"]),
    EvalCase("scn_knob", "make me a new knob", [r"print|measure|calipers|shaft|knob"], ["highlight"]),
    EvalCase("scn_knob", "design a replacement", [r"print|measure|calipers|shaft|knob"], ["highlight"]),
    EvalCase("scn_knob", "I want to recreate the knob", [r"print|measure|calipers|shaft|knob"], ["highlight"]),
    EvalCase("scn_knob", "fabricate a knob for this shaft", [r"print|measure|calipers|shaft|knob"], ["highlight"]),
    EvalCase("scn_knob", "where can I buy a knob", [r"knob|part|match|measure|catalogue"]),
    EvalCase("scn_knob", "where do I get a replacement knob", [r"knob|part|match|measure|catalogue"]),
    EvalCase("scn_knob", "how big is the shaft", [r"mm|scan|caliper|snapped"]),
    EvalCase("scn_knob", "what size is the knob", [r"mm|scan|caliper|snapped"]),
    EvalCase("scn_knob", "what are the dimensions", [r"mm"]),
    EvalCase("scn_knob", "measure the shaft", [r"mm"]),
    EvalCase("scn_knob", "what is [[part:p2]]", [r"shaft|D-shaft"], ["highlight"]),
    EvalCase("scn_knob", "tell me about the control panel", [r"panel"], ["highlight"]),
    EvalCase("scn_knob", "is it safe to print a knob?", [r"knob|print|shaft|measure|help"]),
    EvalCase("scn_knob", "explain the shaft fitting", [r"shaft|panel|control"]),
    EvalCase("scn_knob", "zzzzz nonsense", [r"help|explain|diagnose|repair|part"]),
]

ALL_CASES: list[EvalCase] = TRAP_CASES + KNOB_CASES


def run_case(agent, case: EvalCase) -> EvalOutcome:
    from screwit.agent.offline import reset_state

    reset_state()  # each case is an independent conversation
    text_parts: list[str] = []
    events_seen: list[str] = []
    for ev in agent.reply(case.scan_id, case.question, []):
        if isinstance(ev, TokenEvent):
            text_parts.append(ev.text)
        name = {TokenEvent: "token", ToolCallEvent: "tool_call", HighlightEvent: "highlight",
                CardEvent: "card", DoneEvent: "done"}.get(type(ev), getattr(ev, "event", "unknown"))
        events_seen.append(name)

    reply = "".join(text_parts)
    for pattern in case.expect_any:
        if not re.search(pattern, reply, re.IGNORECASE):
            return EvalOutcome(case, False, reply, events_seen, f"no reply text matched {pattern!r}")
    for required in case.expect_events:
        if required not in events_seen:
            return EvalOutcome(case, False, reply, events_seen, f"missing event {required!r}")
    return EvalOutcome(case, True, reply, events_seen)
