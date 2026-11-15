"""FastAPI application factory (SPEC 5, 7 P4).

Every API route lives under /api and depends on require_access (loopback gate). The Next.js server
proxies /api/* to this app so the browser talks to one origin (AGENTS.md decision 1).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request

from screwit.app.accounts import initialize_accounts, require_build_access
from screwit.app.auth import require_access
from screwit.app.db import init_db
from screwit.app.registry import build_registry
from screwit.app.routes import accounts, builds, cad, chat, parts, recon, scans, storage
from screwit.app.settings import Settings, settings


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings()
    app = FastAPI(title="Screw It API", version="0.1.0")

    @app.middleware("http")
    async def private_responses(request: Request, call_next):
        # Nothing is cached unless its route says otherwise (the workspace preview revalidates by ETag).
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "private, no-store")
        return response

    app.state.settings = cfg
    app.state.registry = build_registry(cfg)

    init_db()
    initialize_accounts(cfg.fixtures_dir)

    app.include_router(accounts.router, prefix="/api", dependencies=[Depends(require_access)])
    gate = [Depends(require_access), Depends(require_build_access)]
    app.include_router(builds.router, prefix="/api", dependencies=gate)
    app.include_router(scans.router, prefix="/api", dependencies=gate)
    app.include_router(recon.router, prefix="/api", dependencies=gate)
    app.include_router(parts.router, prefix="/api", dependencies=gate)
    app.include_router(chat.router, prefix="/api", dependencies=gate)
    app.include_router(cad.router, prefix="/api", dependencies=gate)
    app.include_router(storage.router, prefix="/api", dependencies=gate)
    return app


app = create_app()
