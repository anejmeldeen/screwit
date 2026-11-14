"""Server entrypoint: `python -m screwit.app` (SPEC 7 P4, AGENTS.md decision 1).

Binds 127.0.0.1 by default and REFUSES to start on a non-loopback address. The only way to reach the
demo from elsewhere is an SSH local port forward or an Amazon Tunnels share, never a wide bind.
"""

from __future__ import annotations

from screwit.app.settings import is_loopback, settings


def refuse_non_loopback(bind: str) -> None:
    if not is_loopback(bind):
        raise SystemExit(
            f"refusing to bind {bind!r}: the server binds 127.0.0.1 only. "
            "Reach the demo over an SSH port forward or an Amazon Tunnels share instead."
        )


def main() -> None:
    import os

    cfg = settings()
    refuse_non_loopback(cfg.bind)

    import uvicorn

    # SCREWIT_RELOAD=1 restarts the server on backend code changes (dynamic dev edits). Off by
    # default: autoreload adds overhead and re-imports the heavy model modules on every change.
    reload = os.environ.get("SCREWIT_RELOAD", "0").strip().lower() in ("1", "true", "on", "yes")
    uvicorn.run("screwit.app.main:app", host=cfg.bind, port=cfg.port, log_level="info",
                reload=reload)


if __name__ == "__main__":
    main()  # a non-loopback bind raises SystemExit, whose message the interpreter prints to stderr.
