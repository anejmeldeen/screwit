"""Deterministic scripted LLM driver for tests and SCREWIT_LLM=fake (AGENTS.md decision 4).

A FakeDriver is given a list of scripted turns. Each turn is either text (streamed as deltas) or a
tool call; after a tool call the loop feeds the result back and the driver advances to the next turn.
This lets a test assert the whole tool-use flow (confirm-before-printing refusal, cards, highlights)
with no network.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from screwit.agent.driver import TextStep, ToolUseStep


@dataclass
class ScriptTurn:
    """One assistant turn. `text` is streamed in small chunks; `tool_calls` are emitted after it."""

    text: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)


class FakeDriver:
    def __init__(self, turns: list[ScriptTurn]) -> None:
        self._turns = turns
        self._i = 0

    def run_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Iterator[Any]:
        if self._i >= len(self._turns):
            return
        turn = self._turns[self._i]
        self._i += 1
        for chunk in _chunks(turn.text):
            yield TextStep(text=chunk)
        for n, (name, args) in enumerate(turn.tool_calls):
            yield ToolUseStep(id=f"tool_{self._i}_{n}", name=name, input=dict(args))


def _chunks(text: str, size: int = 24) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []
