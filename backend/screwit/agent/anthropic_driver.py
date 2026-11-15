"""Real streaming Claude driver (SPEC 6.3, llm.py). Used for every real provider (openrouter, bedrock,
anthropic).

Kept thin and import-light: the anthropic SDK and the client are obtained from screwit.llm, so the
provider switch and credential handling live in one place (AGENTS.md decision 4).

Model: SCREWIT_CHAT_MODEL when set, else SCREWIT_MODEL. Only Screwy's chat reads SCREWIT_CHAT_MODEL;
perception and codegen keep SCREWIT_MODEL. render_view returns images, so the chat model must accept
image input.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from screwit.agent.driver import TextStep, ToolUseStep
from screwit.llm import client, config

_MAX_TOKENS = 2048
CHAT_MODEL_ENV = "SCREWIT_CHAT_MODEL"


def chat_model() -> str | None:
    """The model id for Screwy's chat replies."""
    return os.environ.get(CHAT_MODEL_ENV, "").strip() or config().model


class AnthropicDriver:
    def __init__(self) -> None:
        self._client = client()
        self._model = chat_model()

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
