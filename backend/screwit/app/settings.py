"""Runtime configuration from the environment (SPEC 3.3, AGENTS.md decisions).

No value here is a secret; credentials come from the environment and are never written to a file.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Repo root is three levels up from this file: screwit/app/settings.py -> backend/screwit/app.
_REPO_ROOT = Path(__file__).resolve().parents[3]

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # SPEC 2.2/3.3: 500 MB hard cap.


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def is_loopback(host: str) -> bool:
    """True when host is a loopback literal (127.0.0.0/8, ::1). Names are never treated as loopback."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    mode: str  # "mock" | "live"
    bind: str  # host the server binds to
    port: int
    data_dir: Path
    fixtures_dir: Path
    database_url: str
    auth: str  # "loopback" | "midway"
    queue: str  # "inline" | "rq"
    redis_url: str

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"


@lru_cache(maxsize=1)
def settings() -> Settings:
    mode = _env("SCREWIT_MODE", "live")
    if mode not in {"mock", "live"}:
        raise ValueError(f"SCREWIT_MODE must be mock or live, got {mode!r}")

    bind = _env("SCREWIT_BIND", "127.0.0.1")
    port = int(_env("SCREWIT_PORT", "8000"))

    data_dir = Path(_env("SCREWIT_DATA_DIR", str(_REPO_ROOT / "data"))).resolve()
    fixtures_dir = Path(_env("SCREWIT_FIXTURES_DIR", str(_REPO_ROOT / "fixtures"))).resolve()

    database_url = _env("DATABASE_URL", f"sqlite:///{data_dir / 'screwit.db'}")

    auth = _env("SCREWIT_AUTH", "loopback")
    if auth not in {"loopback", "midway"}:
        raise ValueError(f"SCREWIT_AUTH must be loopback or midway, got {auth!r}")

    queue = _env("SCREWIT_QUEUE", "inline")
    if queue not in {"inline", "rq"}:
        raise ValueError(f"SCREWIT_QUEUE must be inline or rq, got {queue!r}")

    redis_url = _env("SCREWIT_REDIS_URL", "redis://127.0.0.1:6379/0")

    return Settings(
        mode=mode,
        bind=bind,
        port=port,
        data_dir=data_dir,
        fixtures_dir=fixtures_dir,
        database_url=database_url,
        auth=auth,
        queue=queue,
        redis_url=redis_url,
    )


def reset_settings_cache() -> None:
    """Tests mutate the environment then call this so the next settings() re-reads it."""
    settings.cache_clear()
