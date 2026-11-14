"""SSE wire format for chat and progress streams (SPEC 4.8).

Each event is framed as `event: <name>` plus one `data: <json>` line. The JSON is the event's payload
without its `event` discriminator; for a card event it is the card itself ({"type", "payload"}).
"""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from .models import CardEvent, ChatEvent, Scan

_chat_adapter: TypeAdapter = TypeAdapter(ChatEvent)

KEEPALIVE = ": keep-alive\n\n"


def encode_chat_event(ev) -> str:
    if isinstance(ev, CardEvent):
        data = ev.card.model_dump(mode="json")
    else:
        data = ev.model_dump(mode="json", exclude={"event"})
    return f"event: {ev.event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def decode_chat_event(name: str, data: str):
    """Inverse of encode_chat_event; used by tests and the mock-script checker."""
    payload = json.loads(data)
    if name == "card":
        return _chat_adapter.validate_python({"event": "card", "card": payload})
    return _chat_adapter.validate_python({"event": name, **payload})


def encode_progress(scan: Scan) -> str:
    return f"event: scan\ndata: {scan.model_dump_json()}\n\n"
