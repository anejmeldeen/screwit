"""Chat routes: POST streams the agent's reply as SSE, GET returns history (SPEC 5, 4.8, 6.3)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException

from screwit.agent.factory import build_chat_agent
from screwit.app import signing
from screwit.app.deps import RegistryDep
from screwit.app.registry import Registry
from screwit.app.sse import sse_response_with_keepalive
from screwit.app.store import ScanStore, utcnow
from screwit.contracts import (
    CardEvent,
    ChatMessage,
    ChatRequest,
    DoneEvent,
    TokenEvent,
)
from screwit.contracts.sse import encode_chat_event

router = APIRouter()


def _store(reg: Registry) -> ScanStore:
    return ScanStore(reg.data)


@router.get("/scans/{scan_id}/chat", response_model=list[ChatMessage])
def chat_history(scan_id: str, reg: RegistryDep) -> list[ChatMessage]:
    return _store(reg).list_chat(scan_id)


@router.post("/scans/{scan_id}/chat")
def chat(scan_id: str, req: ChatRequest, reg: RegistryDep):
    store = _store(reg)
    if store.get_scan(scan_id) is None:
        raise HTTPException(status_code=404, detail="scan not found")

    # Persist the user message before streaming the reply (SPEC 2.3 history).
    store.append_chat(scan_id, ChatMessage(message_id="usr_" + uuid.uuid4().hex[:12], role="user",
                                           text=req.message, created_at=utcnow()))
    agent = build_chat_agent(reg)

    def frames() -> Iterator[str]:
        text_parts: list[str] = []
        cards = []
        message_id = "msg_" + uuid.uuid4().hex[:12]
        for event in agent.reply(scan_id, req.message, req.image_uris):
            if isinstance(event, TokenEvent):
                text_parts.append(event.text)
            elif isinstance(event, CardEvent):
                if event.card.type == "cad_result":
                    event = CardEvent(card=event.card.model_copy(
                        update={"payload": signing.cad_result(reg.storage, event.card.payload)}))
                cards.append(event.card)
            elif isinstance(event, DoneEvent):
                message_id = event.message_id
            yield encode_chat_event(event)
        # Persist the assembled assistant message once the stream is done.
        store.append_chat(scan_id, ChatMessage(message_id=message_id, role="assistant",
                                               text="".join(text_parts), cards=cards, created_at=utcnow()))

    return sse_response_with_keepalive(frames())
