"""Exact-demo replay agent: replays the fixture chat_script.json (SPEC 8, behind SCREWIT_CHAT_REPLAY=1).

This is NO LONGER the default fallback -- the deterministic offline agent (screwit.agent.offline) is.
It exists only as an optional exact-demo mode: the first turn whose regex matches the user message is
replayed verbatim as contract ChatEvents, with a small delay so the UI shows streaming. The script's
events are validated against the contracts on load, so a drifted fixture fails loudly.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from screwit.app.fakes import FixtureData, mock_match
from screwit.contracts import DoneEvent, ErrorEvent
from screwit.contracts.sse import decode_chat_event

_EVENT_DELAY_S = 0.02  # small, so replay feels live without slowing tests meaningfully.


class MockReplayChatAgent:
    def __init__(self, data: FixtureData, delay_s: float = _EVENT_DELAY_S) -> None:
        self._data = data
        self._delay = delay_s

    def reply(self, scan_id: str, message: str, image_uris: list[str]) -> Iterator:
        script = self._data.chat_script_raw(scan_id)
        if script is None:
            yield ErrorEvent(message="no replay script for this scan (SCREWIT_CHAT_REPLAY exact-demo mode)")
            yield DoneEvent(message_id="replay_no_script")
            return

        turn = mock_match(script["turns"], message)
        for raw in turn["events"]:
            yield _decode_event(raw)
            if self._delay:
                time.sleep(self._delay)


# Backwards-compatible alias (older imports referenced MockChatAgent).
MockChatAgent = MockReplayChatAgent


def _decode_event(raw: dict):
    """Rebuild a contract ChatEvent from the fixture's stored event dict via the contract decoder."""
    name = raw["event"]
    if name == "card":
        return decode_chat_event("card", _json(raw["card"]))
    payload = {k: v for k, v in raw.items() if k != "event"}
    return decode_chat_event(name, _json(payload))


def _json(obj) -> str:
    import json

    return json.dumps(obj)
