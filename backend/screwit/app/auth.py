"""Access gate (AGENTS.md decision 1, overrides SPEC 3.3 demo token).

No bearer token anywhere. `loopback` mode admits only requests that arrived on a loopback socket;
`midway` mode is a fail-closed stub that rejects everyone with 401 until a real integration exists.
There is deliberately no bypass environment variable.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from screwit.app.settings import is_loopback, settings


def client_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def require_access(request: Request) -> None:
    """FastAPI dependency placed on every API route. Raises 401/403; returns None on success."""
    cfg = settings()
    if cfg.auth == "midway":
        raise HTTPException(status_code=401, detail="Midway integration not configured")

    host = client_host(request)
    if host is None or not is_loopback(host):
        raise HTTPException(status_code=403, detail="Only loopback clients are allowed")
