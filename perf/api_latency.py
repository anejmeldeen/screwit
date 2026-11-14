"""API latency suite (brief step 5).

p50/p95 for every GET route in mock mode over 50 requests, and SSE first-event latency
(the time from opening the chat/progress stream to the first `event:` line arriving).

The API is reached only on loopback (SPEC 3.3 / AGENTS.md decision 1). This module is given a
base URL like `http://127.0.0.1:8000`. When the server is not reachable it returns a skip reason
and produces no numbers, so the suite never errors before the platform branch merges.

Routes measured are the mock-mode GETs from SPEC 5 that serve fixtures with no external call. The
hero scan id is `scn_trap` (the P-trap). Chat first-token is measured separately in the Playwright
suite against the real browser; here we measure the *server-side* SSE first-event latency on the
chat POST and on the progress GET, which is the API's own contribution to that budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# GET routes that mock mode can serve from fixtures (SPEC 5), parameterised for scn_trap.
HERO = "scn_trap"
GET_ROUTES: list[str] = [
    "/api/demo-scans",
    f"/api/scans/{HERO}",
    f"/api/scans/{HERO}/reconstruction",
    f"/api/scans/{HERO}/parts",
    f"/api/scans/{HERO}/scene-graph",
    f"/api/scans/{HERO}/chat",  # GET = chat history
]

N_REQUESTS = 50


@dataclass
class RouteLatency:
    route: str
    n: int
    ok: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    errors: list[str] = field(default_factory=list)


@dataclass
class SseLatency:
    stream: str
    first_event_ms: float | None
    first_event_name: str | None
    detail: str = ""


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a non-empty list (pct in 0..100)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    # nearest-rank
    k = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[k - 1]


def measure_route(client: httpx.Client, route: str, n: int = N_REQUESTS) -> RouteLatency:
    samples: list[float] = []
    errors: list[str] = []
    ok = 0
    for _ in range(n):
        start = time.perf_counter()
        try:
            resp = client.get(route)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 200:
                ok += 1
                samples.append(elapsed_ms)
            else:
                errors.append(f"HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    return RouteLatency(
        route=route,
        n=n,
        ok=ok,
        p50_ms=_percentile(samples, 50) if samples else None,
        p95_ms=_percentile(samples, 95) if samples else None,
        max_ms=max(samples) if samples else None,
        errors=errors[:5],
    )


def measure_sse_first_event(
    client: httpx.Client, method: str, route: str, *, json_body: dict | None = None,
    timeout_s: float = 10.0,
) -> SseLatency:
    """Time to the first `event:` line on an SSE stream, ignoring `: keep-alive` comments."""
    start = time.perf_counter()
    try:
        with client.stream(method, route, json=json_body) as resp:
            if resp.status_code != 200:
                return SseLatency(route, None, None, detail=f"HTTP {resp.status_code}")
            for line in resp.iter_lines():
                if not isinstance(line, str):
                    line = line.decode("utf-8", "replace")
                if line.startswith(":"):  # keep-alive comment
                    continue
                if line.startswith("event:"):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    return SseLatency(route, elapsed_ms, line.split(":", 1)[1].strip())
                if (time.perf_counter() - start) > timeout_s:
                    return SseLatency(route, None, None, detail="timed out before first event")
        return SseLatency(route, None, None, detail="stream closed with no event")
    except Exception as exc:  # noqa: BLE001
        return SseLatency(route, None, None, detail=str(exc))


def run_api_suite(base_url: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Run the full API latency suite. Returns a dict for the report, or a skip."""
    try:
        with httpx.Client(base_url=base_url, timeout=timeout_s) as client:
            # Readiness probe.
            try:
                probe = client.get("/api/demo-scans")
            except Exception as exc:  # noqa: BLE001
                return {"skipped": True, "reason": f"API not reachable at {base_url}: {exc}"}
            if probe.status_code != 200:
                return {"skipped": True,
                        "reason": f"API at {base_url} returned HTTP {probe.status_code} for /api/demo-scans"}

            routes = [measure_route(client, r) for r in GET_ROUTES]
            chat_sse = measure_sse_first_event(
                client, "POST", f"/api/scans/{HERO}/chat",
                json_body={"message": "why is it leaking?"},
            )
            progress_sse = measure_sse_first_event(client, "GET", f"/api/scans/{HERO}/events")
            return {
                "skipped": False,
                "base_url": base_url,
                "n_requests": N_REQUESTS,
                "routes": [vars(r) for r in routes],
                "sse": [vars(chat_sse), vars(progress_sse)],
            }
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"API suite error: {exc}"}
