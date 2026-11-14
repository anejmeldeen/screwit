"""Real streaming Claude driver (SPEC 6.3, llm.py). Used only when SCREWIT_LLM is bedrock or anthropic.

Kept thin and import-light: the anthropic SDK and the client are obtained from screwit.llm, so the
provider switch and credential handling live in one place (AGENTS.md decision 4).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from screwit.agent.driver import TextStep, ToolUseStep
from screwit.llm import client, config

_MAX_TOKENS = 2048


class AnthropicDriver:
    def __init__(self) -> None:
        self._client = client()
        self._model = config().model

    def run_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Iterator[Any]:
        # Stream text deltas live; tool-use blocks (with their assembled JSON input) are read from the
        # final message the SDK accumulates, so partial-JSON reassembly is the SDK's job, not ours.
        with self._client.messages.stream(
            model=self._model, max_tokens=_MAX_TOKENS, system=system, messages=messages, tools=tools
        ) as stream:
            for event in stream:
                if getattr(event, "type", None) == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", None) == "text_delta":
                        yield TextStep(text=delta.text)
            final = stream.get_final_message()
        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                yield ToolUseStep(id=block.id, name=block.name, input=dict(block.input or {}))
