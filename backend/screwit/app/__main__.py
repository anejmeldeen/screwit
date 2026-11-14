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
    cfg = settings()
    refuse_non_loopback(cfg.bind)

    import uvicorn

    uvicorn.run("screwit.app.main:app", host=cfg.bind, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()  # a non-loopback bind raises SystemExit, whose message the interpreter prints to stderr.
