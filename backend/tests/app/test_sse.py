"""SSE framing for chat and progress (SPEC 4.8). Decode with contracts.sse.decode_chat_event."""

from __future__ import annotations

from screwit.contracts import DoneEvent, Scan
from screwit.contracts.sse import decode_chat_event


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Split an SSE body into (event_name, data_json) pairs, ignoring comment keep-alive lines."""
    events: list[tuple[str, str]] = []
    name: str | None = None
    for block in body.split("\n\n"):
        name = None
        data = None
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if name is not None and data is not None:
            events.append((name, data))
    return events


def test_chat_sse_frames_decode(client):
    # The default (offline) agent plans a repair: highlight + tokens + a repair_plan card.
    r = client.post("/api/scans/scn_trap/chat", json={"message": "how do I repair this leak?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    decoded = [decode_chat_event(name, data) for name, data in events]
    assert decoded[-1].event == "done"
    assert any(e.event == "token" for e in decoded)
    assert any(e.event == "highlight" for e in decoded)
    assert any(e.event == "card" for e in decoded)


def test_chat_sse_card_roundtrips(client):
    # Two-step make flow: request a replacement, then supply the caliper value -> a cad_result card.
    client.post("/api/scans/scn_knob/chat", json={"message": "3D print me a replacement knob"})
    r = client.post("/api/scans/scn_knob/chat", json={"message": "the exposed length is 18 mm"})
    decoded = [decode_chat_event(n, d) for n, d in _parse_sse(r.text)]
    cards = [e for e in decoded if e.event == "card"]
    assert cards and cards[0].card.type == "cad_result"
    assert cards[0].card.payload.status in {"ok", "failed_checks"}


def test_progress_sse_frame_is_a_scan(client):
    r = client.get("/api/scans/scn_trap/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert events[0][0] == "scan"
    scan = Scan.model_validate_json(events[0][1])
    assert scan.scan_id == "scn_trap"


def test_chat_history_persisted_after_stream(client):
    client.post("/api/scans/scn_trap/chat", json={"message": "why does it leak?"})
    history = client.get("/api/scans/scn_trap/chat").json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "[[part:p3]]" in history[1]["text"]


def test_done_event_contract():
    ev = decode_chat_event("done", '{"message_id":"m1"}')
    assert isinstance(ev, DoneEvent)
