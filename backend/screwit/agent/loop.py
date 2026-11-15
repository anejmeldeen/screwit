"""The tool-use chat loop: a Driver plus the tool layer, yielding contract ChatEvents (SPEC 6.3, 4.8).

One reply() call runs assistant turns until the model stops without a tool call (or a turn cap is hit).
Text deltas become TokenEvents; a highlight tool (or add_part, for the new part) becomes a
HighlightEvent; a card-producing tool becomes a CardEvent; every tool call also surfaces as a
ToolCallEvent so the UI can show the agent's actions. Ends with a DoneEvent. Any exception becomes an
ErrorEvent so a stream never dies silently.

The model sees the system prompt plus the object context (agent/context.py, rebuilt every turn so a
tool's change shows up on the next turn), the recent stored conversation as text-only turns, then the
current message.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

from screwit.agent.add_part import PartAddedSignal
from screwit.agent.context import history_turns, object_context
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

log = logging.getLogger("screwit.agent")

_MAX_TURNS = 8


class LlmChatAgent:
    """ChatAgent (interfaces.py) driven by any Driver (real Claude or the fake)."""

    def __init__(self, reg: Registry, driver: Driver) -> None:
        self._reg = reg
        self._driver = driver

    def reply(self, scan_id: str, message: str, image_uris: list[str]) -> Iterator:
        ctx = ToolContext(self._reg, scan_id)
        store = ScanStore(self._reg.data)
        tools = anthropic_tool_specs()
        messages = self._conversation(store, scan_id, message, image_uris)
        try:
            yield from self._run(ctx, tools, messages, lambda: self._system(store, scan_id))
        except Exception as e:  # a tool or driver fault must not kill the stream silently.
            yield ErrorEvent(message=f"agent error: {e}")
        yield DoneEvent(message_id="msg_" + uuid.uuid4().hex[:12])

    def _system(self, store: ScanStore, scan_id: str) -> str:
        try:
            context = object_context(store, scan_id)
        except Exception:  # noqa: BLE001 - missing context must not stop the reply
            log.exception("object context failed for %s", scan_id)
            context = "OBJECT CONTEXT\nUnavailable for this turn; use the tools.\nEND OBJECT CONTEXT"
        return f"{SYSTEM_PROMPT}\n{context}\n"

    def _conversation(self, store: ScanStore, scan_id: str, message: str,
                      image_uris: list[str]) -> list[dict]:
        try:
            turns = history_turns(store.list_chat(scan_id), message)
        except Exception:  # noqa: BLE001 - history is a help, not a requirement
            log.exception("chat history failed for %s", scan_id)
            turns = []
        content = _user_content(message, image_uris)
        if turns and turns[-1]["role"] == "user":
            # The previous message has no stored reply (it failed); fold it in so roles alternate.
            content = _prepend_text(turns.pop()["content"], content)
        return [*turns, {"role": "user", "content": content}]

    def _run(self, ctx: ToolContext, tools: list[dict], messages: list[dict], system) -> Iterator:
        said = False      # text already streamed in an earlier turn of this reply
        for _ in range(_MAX_TURNS):
            turn = steps_to_turn(self._driver.run_turn(system(), messages, tools))

            assistant_content: list[dict] = []
            tool_results: list[dict] = []
            first_text = True
            for step in turn.steps:
                if isinstance(step, TextStep):
                    if step.text:
                        # Text after a tool call starts a new paragraph; streamed as is, it ran on
                        # from the text before the call ("the parts.Here's what you have").
                        if said and first_text and not step.text[:1].isspace():
                            yield TokenEvent(text="\n\n")
                        first_text = False
                        yield TokenEvent(text=step.text)
                        assistant_content.append({"type": "text", "text": step.text})
                elif isinstance(step, ToolUseStep):
                    yield ToolCallEvent(name=step.name, args=step.input)
                    assistant_content.append(
                        {"type": "tool_use", "id": step.id, "name": step.name, "input": step.input})
                    try:
                        result = ctx.dispatch(step.name, step.input)
                        event, payload = _handle_result(result)
                        failed = False
                    except Exception:
                        log.exception("tool %s failed", step.name)
                        event = None
                        payload = ("This tool failed. Explain the limitation and use other available data "
                                   "or ask a question.")
                        failed = True
                    if event is not None:
                        yield event
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": step.id, "content": payload, "is_error": failed})

            said = said or not first_text
            if turn.stop_reason != "tool_use":
                return
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        yield ErrorEvent(message="Screwy reached the tool-call limit. Please narrow down the request.")

def _handle_result(result):
    """Map a tool result to (ChatEvent-or-None, JSON string for the model)."""
    import json

    if isinstance(result, HighlightSignal):
        return HighlightEvent(part_ids=result.part_ids), json.dumps({"highlighted": result.part_ids})
    if isinstance(result, PartAddedSignal):
        return HighlightEvent(part_ids=[result.part_id]), json.dumps(result.result)
    if isinstance(result, CardSignal):
        card = result.card
        return CardEvent(card=card), json.dumps(card.model_dump(mode="json"))
    if isinstance(result, dict) and "image_base64" in result:
        return None, [{"type": "image", "source": {"type": "base64",
                      "media_type": result["media_type"], "data": result["image_base64"]}}]
    return None, json.dumps(result)


def _user_content(message: str, image_uris: list[str]) -> list[dict] | str:
    if not image_uris:
        return message
    content: list[dict] = [{"type": "text", "text": message}]
    for uri in image_uris:
        content.append({"type": "text", "text": f"[attached image: {uri}]"})
    return content


def _prepend_text(earlier: str, content: list[dict] | str) -> list[dict] | str:
    if isinstance(content, str):
        return f"{earlier}\n\n{content}"
    return [{"type": "text", "text": earlier}, *content]
