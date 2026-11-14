"""FastAPI application factory (SPEC 5, 7 P4).

Every API route lives under /api and depends on require_access (loopback gate). The Next.js server
proxies /api/* to this app so the browser talks to one origin (AGENTS.md decision 1).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI

from screwit.app.auth import require_access
from screwit.app.db import init_db
from screwit.app.registry import build_registry
from screwit.app.routes import cad, chat, parts, recon, scans, storage
from screwit.app.settings import Settings, settings


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings()
    app = FastAPI(title="Screw It API", version="0.1.0")
    app.state.settings = cfg
    app.state.registry = build_registry(cfg)

    init_db()

    gate = [Depends(require_access)]
    app.include_router(scans.router, prefix="/api", dependencies=gate)
    app.include_router(recon.router, prefix="/api", dependencies=gate)
    app.include_router(parts.router, prefix="/api", dependencies=gate)
    app.include_router(chat.router, prefix="/api", dependencies=gate)
    app.include_router(cad.router, prefix="/api", dependencies=gate)
    app.include_router(storage.router, prefix="/api", dependencies=gate)
    return app


app = create_app()
