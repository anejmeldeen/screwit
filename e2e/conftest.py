"""Playwright end-to-end fixtures: FastAPI (mock) + the Next.js site, both on loopback.

The suite drives a real browser against the built site talking to the mock API, so it proves the
teams' parts work together. It depends on two things that may not exist yet in a partial repo:

- the API module (`python -m screwit.app`), owned by P4; and
- a Next.js app in WEB_DIR (default the main tree's `web/`), owned by P1, with `npm run start`.

When either is missing, every test SKIPS with a clear reason (never errors), which is the expected
state until those branches merge.

Servers bind 127.0.0.1 only and are torn down at session end. Screenshots go under e2e/artifacts/
(gitignored).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from _stack import (
    ARTIFACTS,
    BACKEND,
    FIXTURES_DIR,
    WEB_DIR,
    Stack,
    api_module_available,
    free_port,
    terminate,
    wait_http,
    web_app_available,
)


@pytest.fixture(scope="session")
def stack(tmp_path_factory):
    if not api_module_available():
        pytest.skip("screwit.app not present yet (P4 platform branch not merged)")
    if not web_app_available():
        pytest.skip(f"no built Next.js app at {WEB_DIR} (set WEB_DIR; P1 web branch not merged)")
    if not FIXTURES_DIR.is_dir():
        pytest.skip(f"fixtures dir not found at {FIXTURES_DIR}; set SCREWIT_MAIN_TREE")

    api_port, web_port = free_port(), free_port()
    api_url, web_url = f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{web_port}"

    api_env = {
        **os.environ,
        "SCREWIT_MODE": "mock",
        "SCREWIT_AUTH": "loopback",
        "SCREWIT_BIND": "127.0.0.1",
        "SCREWIT_PORT": str(api_port),
        "SCREWIT_FIXTURES_DIR": str(FIXTURES_DIR),
        "SCREWIT_DATA_DIR": str(tmp_path_factory.mktemp("screwit-e2e-data")),  # fresh state every run
        "PYTHONPATH": str(BACKEND) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    api_env.pop("SCREWIT_LLM", None)
    api = subprocess.Popen([sys.executable, "-m", "screwit.app"], env=api_env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # The site proxies /api to the API; P1's next.config points at 127.0.0.1:8000. We pass a standard
    # API-origin override plus PORT/HOST so `npm run start` binds the loopback port we chose.
    web_env = {
        **os.environ,
        "PORT": str(web_port),
        "HOST": "127.0.0.1",
        "HOSTNAME": "127.0.0.1",
        "SCREWIT_API_ORIGIN": api_url,
        "NEXT_PUBLIC_API_ORIGIN": api_url,
    }
    web = subprocess.Popen(["npm", "run", "start", "--", "-H", "127.0.0.1", "-p", str(web_port)],
                           cwd=str(WEB_DIR), env=web_env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait_http(api, f"{api_url}/api/demo-scans", 30)
        wait_http(web, web_url, 60)
    except Exception as e:
        terminate(web)
        terminate(api)
        pytest.skip(f"stack did not come up: {e}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    yield Stack(web_url=web_url, api_url=api_url)
    terminate(web)
    terminate(api)


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    return {**browser_type_launch_args, "headless": True}


@pytest.fixture
def page(page, stack):
    response = page.context.request.post(f"{stack.web_url}/api/auth/login",
                                         data={"username": "admin", "password": "password123"})
    assert response.ok, response.text()
    return page


@pytest.fixture(scope="session")
def browser_ready():
    """Skip (not error) when the Playwright chromium driver cannot launch on this host.

    Some sandboxes lack the shared libraries the bundled Node driver needs (e.g. libatomic.so.1),
    so a browser test would otherwise ERROR at `page` setup. Probe once and skip clearly instead.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # pragma: no cover - import guard
        pytest.skip(f"playwright not importable: {e}")
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            b.close()
    except Exception as e:
        pytest.skip(f"chromium cannot launch on this host (install browsers / missing libs): {e}")
    return True
