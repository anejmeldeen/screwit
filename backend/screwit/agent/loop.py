"""The tool-use chat loop: a Driver plus the tool layer, yielding contract ChatEvents (SPEC 6.3, 4.8).

One reply() call runs assistant turns until the model stops without a tool call (or a turn cap is hit).
Text deltas become TokenEvents; a highlight tool becomes a HighlightEvent; a card-producing tool becomes
a CardEvent; every tool call also surfaces as a ToolCallEvent so the UI can show the agent's actions.
Ends with a DoneEvent. Any exception becomes an ErrorEvent so a stream never dies silently.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from screwit.agent.driver import Driver, TextStep, ToolUseStep, steps_to_turn
from screwit.agent.prompt import SYSTEM_PROMPT
from screwit.agent.tools import CardSignal, HighlightSignal, ToolContext, anthropic_tool_specs
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import (
    CardEvent,
    DoneEvent,
    ErrorEvent,
    HighlightEvent,
    TokenEvent,
    ToolCallEvent,
)
from screwit.contracts.sse import encode_chat_event  # noqa: F401  (re-exported convenience for callers)

_MAX_TURNS = 8


class LlmChatAgent:
    """ChatAgent (interfaces.py) driven by any Driver (real Claude or the fake)."""

    def __init__(self, reg: Registry, driver: Driver) -> None:
        self._reg = reg
        self._driver = driver

    def reply(self, scan_id: str, message: str, image_uris: list[str]) -> Iterator:
        ctx = ToolContext(self._reg, scan_id)
        tools = anthropic_tool_specs()
        try:
            history = ScanStore(self._reg.data).list_chat(scan_id)
            # The HTTP handler persists the incoming turn before starting the agent.
            if history and history[-1].role == "user" and history[-1].text == message:
                history.pop()
            messages = [{"role": turn.role, "content": turn.text[-4000:]}
                        for turn in history[-12:] if turn.text]
            messages.append({"role": "user", "content": _user_content(message, image_uris)})
            yield from self._run(ctx, tools, messages)
        except Exception as e:  # a tool or driver fault must not kill the stream silently.
            yield ErrorEvent(message=f"agent error: {e}")
        yield DoneEvent(message_id="msg_" + uuid.uuid4().hex[:12])

    def _run(self, ctx: ToolContext, tools: list[dict], messages: list[dict]) -> Iterator:
        for _ in range(_MAX_TURNS):
            turn = steps_to_turn(self._driver.run_turn(SYSTEM_PROMPT, messages, tools))

            assistant_content: list[dict] = []
            tool_results: list[dict] = []
            for step in turn.steps:
                if isinstance(step, TextStep):
                    if step.text:
                        yield TokenEvent(text=step.text)
                        assistant_content.append({"type": "text", "text": step.text})
                elif isinstance(step, ToolUseStep):
                    yield ToolCallEvent(name=step.name, args=step.input)
                    assistant_content.append(
                        {"type": "tool_use", "id": step.id, "name": step.name, "input": step.input})
                    result = ctx.dispatch(step.name, step.input)
                    event, payload = _handle_result(result)
                    if event is not None:
                        yield event
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": step.id, "content": payload})

            if turn.stop_reason != "tool_use":
                return
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})


def _handle_result(result):
    """Map a tool result to (ChatEvent-or-None, JSON string for the model)."""
    import json

    if isinstance(result, HighlightSignal):
        return HighlightEvent(part_ids=result.part_ids), json.dumps({"highlighted": result.part_ids})
    if isinstance(result, CardSignal):
        card = result.card
        return CardEvent(card=card), json.dumps(card.model_dump(mode="json"))
    return None, json.dumps(result)


def _user_content(message: str, image_uris: list[str]) -> list[dict] | str:
    if not image_uris:
        return message
    content: list[dict] = [{"type": "text", "text": message}]
    for uri in image_uris:
        content.append({"type": "text", "text": f"[attached image: {uri}]"})
    return content
