"""Minimal SSE reader for the SPEC 4.8 wire format, over an httpx streaming response.

The wire format is `event: <name>` + one `data: <json>` line per event, blocks separated by a
blank line, with `: keep-alive` comment lines interleaved (SPEC 3.3, 4.8). We parse exactly that
and hand each (name, data) pair to contracts.sse.decode_chat_event so the test asserts against the
real contract objects, never ad-hoc dicts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from screwit.contracts.sse import decode_chat_event


@dataclass
class RawEvent:
    name: str
    data: str


def iter_sse(resp: httpx.Response) -> Iterator[RawEvent]:
    """Yield SSE events from a streaming response; keep-alive comment lines are ignored."""
    name: str | None = None
    data_lines: list[str] = []
    for line in resp.iter_lines():
        if line == "":
            if name is not None:
                yield RawEvent(name=name, data="\n".join(data_lines))
            name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue  # keep-alive / comment
        if line.startswith("event:"):
            name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if name is not None:  # final block with no trailing blank line
        yield RawEvent(name=name, data="\n".join(data_lines))


def decode_chat(resp: httpx.Response) -> list:
    """Decode a chat stream into contracts ChatEvent objects (SPEC 4.8)."""
    return [decode_chat_event(ev.name, ev.data) for ev in iter_sse(resp)]
