"""Pick the ChatAgent for the current configuration (SPEC 8, AGENTS.md decision 4).

Routing (the brief):
- SCREWIT_CHAT_REPLAY=1  -> the recorded hero script, exact-demo mode only (MockReplayChatAgent).
- SCREWIT_LLM bedrock|anthropic -> the REAL Claude tool-use loop over the fixture-backed tools, even
  in mock mode (mock = cached hero data, not canned replies). If the provider call fails (no
  credentials, network down), it falls back to the offline agent, which says it is offline.
- SCREWIT_LLM unset/fake -> the deterministic offline agent.

Live replies use Screwy's own voice. Offline replies disclose their limited mode.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

from screwit.agent.loop import LlmChatAgent
from screwit.agent.offline import OfflineChatAgent
from screwit.agent.shape_commands import headphone_ear_command, shape_command
from screwit.app.registry import Registry
from screwit.contracts import DoneEvent
from screwit.interfaces import ChatAgent
from screwit.llm import REAL_PROVIDERS, config

_OFFLINE_NOTE = "[offline agent]"
_OFFLINE_AFTER_FAILURE_NOTE = "[offline agent: the language model was unavailable, answering from the scan data]"


def _replay_enabled() -> bool:
    return os.environ.get("SCREWIT_CHAT_REPLAY") == "1"


def build_chat_agent(reg: Registry) -> ChatAgent:
    if _replay_enabled():
        from screwit.agent.mock import MockReplayChatAgent

        return MockReplayChatAgent(reg.data)

    provider = config().provider
    if provider in REAL_PROVIDERS:
        return _LlmWithOfflineFallback(reg)

    # Unset / fake: deterministic offline agent, grounded in the cached scan data.
    return OfflineChatAgent(reg, mode_note=_OFFLINE_NOTE)


class _LlmWithOfflineFallback:
    """Runs the real Claude loop; on any provider/driver failure before the first real event, falls
    back to the offline agent so chat always answers and discloses the fallback."""

    def __init__(self, reg: Registry) -> None:
        self._reg = reg

    def reply(self, scan_id: str, message: str, image_uris: list[str]) -> Iterator:
        route = headphone_ear_command if image_uris else shape_command
        command = route(self._reg, scan_id, message)
        if command is not None:
            yield from command
            yield DoneEvent(message_id="msg_" + uuid.uuid4().hex[:12])
            return
        # Try the real model. Buffer until the first substantive event so that, if the provider fails
        # before producing anything, we emit ONLY the offline note (no misleading online note).
        try:
            from screwit.agent.anthropic_driver import AnthropicDriver

            agent = LlmChatAgent(self._reg, AnthropicDriver())
            stream = agent.reply(scan_id, message, image_uris)
            events = self._online_events(stream)
            first = next(events, None)  # may raise _ProviderFailed if nothing usable came back
            if first is not None:
                yield first
                yield from events
                return
        except Exception:
            pass  # fall through to the offline agent; chat must answer.
        yield from OfflineChatAgent(self._reg, mode_note=_OFFLINE_AFTER_FAILURE_NOTE).reply(
            scan_id, message, image_uris)

    def _online_events(self, stream: Iterator) -> Iterator:
        """Pass through model events, raising _ProviderFailed if the turn yields no usable content."""
        produced = False
        for event in stream:
            kind = getattr(event, "event", None)
            if kind == "error" and not produced:
                raise _ProviderFailed(event.message)
            if kind in {"token", "tool_call", "highlight", "card"}:
                produced = True
            if kind == "done" and not produced:
                raise _ProviderFailed("model returned no content")
            yield event


class _ProviderFailed(RuntimeError):
    pass


__all__ = ["build_chat_agent", "DoneEvent"]
