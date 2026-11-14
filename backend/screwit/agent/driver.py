"""LLM driver abstraction behind the agent loop (SPEC 6.3, AGENTS.md decision 4, llm.py).

The loop (loop.py) talks to a Driver, not to the anthropic SDK directly, so the same loop runs against
the real streaming Claude client and against a scripted fake in tests. One driver turn yields a sequence
of Step objects: text deltas, then any tool calls the model wants, then a stop reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class TextStep:
    kind: Literal["text"] = "text"
    text: str = ""


@dataclass
class ToolUseStep:
    id: str
    name: str
    input: dict
    kind: Literal["tool_use"] = "tool_use"


@dataclass
class TurnResult:
    """One assistant turn: the ordered steps plus the stop reason the driver reported."""

    steps: list[Any] = field(default_factory=list)
    stop_reason: Literal["end_turn", "tool_use"] = "end_turn"


class Driver(Protocol):
    def run_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Iterator[Any]:
        """Yield TextStep / ToolUseStep objects for one assistant turn, in order. The loop decides
        when the turn is over by whether any ToolUseStep appeared."""


def steps_to_turn(steps: Iterator[Any]) -> TurnResult:
    collected = list(steps)
    stop = "tool_use" if any(isinstance(s, ToolUseStep) for s in collected) else "end_turn"
    return TurnResult(steps=collected, stop_reason=stop)
