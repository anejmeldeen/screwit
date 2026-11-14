"""Shared stack config and process helpers for the Playwright e2e suite.

Kept in a plain module (not conftest.py) so both conftest.py and the test modules can import it,
whether the suite runs from the repo (pytest testpaths) or standalone from e2e/ (its own pytest.ini).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

E2E_DIR = Path(__file__).resolve().parent
BACKEND = E2E_DIR.parent / "backend"
MAIN_TREE = Path(
    os.environ.get("SCREWIT_MAIN_TREE") or "/local/home/brancli/.kiro/crew/workspace/screw-it"
)
FIXTURES_DIR = MAIN_TREE / "fixtures"
WEB_DIR = Path(os.environ.get("WEB_DIR") or (MAIN_TREE / "web"))
ARTIFACTS = E2E_DIR / "artifacts"


@dataclass(frozen=True)
class Stack:
    web_url: str
    api_url: str


def api_module_available() -> bool:
    added = str(BACKEND) not in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if added:
        sys.path.insert(0, str(BACKEND))
    try:
        return importlib.util.find_spec("screwit.app") is not None
    except (ImportError, ValueError):
        return False
    finally:
        if added and sys.path and sys.path[0] == str(BACKEND):
            sys.path.pop(0)


def web_app_available() -> bool:
    pkg = WEB_DIR / "package.json"
    return pkg.is_file() and (WEB_DIR / "node_modules").is_dir()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(proc: subprocess.Popen, url: str, timeout_s: float) -> None:
    import httpx

    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=2.0) as c:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"process exited early ({proc.returncode})")
            with contextlib.suppress(httpx.HTTPError):
                c.get(url)
                return
            time.sleep(0.25)
    raise TimeoutError(f"not ready within {timeout_s}s: {url}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=8)
    if proc.poll() is None:
        proc.kill()
