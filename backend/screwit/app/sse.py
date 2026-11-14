"""SSE streaming helpers over the contract wire format (SPEC 4.8, 3.3 keep-alive).

The wire framing itself is owned by contracts.sse (encode_chat_event / encode_progress / KEEPALIVE);
this module only turns an event source into a StreamingResponse and injects the 20 s keep-alive so
proxies do not close the minutes-long parts stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

from starlette.responses import StreamingResponse

from screwit.contracts.sse import KEEPALIVE

SSE_MEDIA_TYPE = "text/event-stream"
KEEPALIVE_INTERVAL_S = 20.0

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def sse_response(frames: Iterator[str]) -> StreamingResponse:
    """Stream pre-encoded SSE frames (each already ends with a blank line)."""

    async def gen() -> AsyncIterator[str]:
        for frame in frames:
            yield frame

    return StreamingResponse(gen(), media_type=SSE_MEDIA_TYPE, headers=_SSE_HEADERS)


def sse_response_with_keepalive(frames: Iterator[str]) -> StreamingResponse:
    """Stream frames and send KEEPALIVE every 20 s of idleness (SPEC 3.3).

    Runs the (possibly blocking) source iterator in a thread so one slow stage cannot wedge the loop.
    """

    async def gen() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        it = iter(frames)
        sentinel = object()
        while True:
            fut = loop.run_in_executor(None, lambda: next(it, sentinel))
            while True:
                try:
                    item = await asyncio.wait_for(asyncio.shield(fut), timeout=KEEPALIVE_INTERVAL_S)
                    break
                except TimeoutError:
                    yield KEEPALIVE
            if item is sentinel:
                return
            yield item

    return StreamingResponse(gen(), media_type=SSE_MEDIA_TYPE, headers=_SSE_HEADERS)
