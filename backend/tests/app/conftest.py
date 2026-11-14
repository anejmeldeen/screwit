"""Shared fixtures for the app tests: a mock-mode app with a fresh temp database and a loopback client."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]  # worktree/backend/tests/app -> worktree
FIXTURES_DIR = REPO_ROOT.parent / "screw-it" / "fixtures"  # fixtures live in the main tree (P4 owns catalogue only)

# The worktree does not contain the fixtures tree (it is the integration lead's). Point at the main tree.
if not FIXTURES_DIR.exists():  # fall back to a sibling checkout layout
    FIXTURES_DIR = REPO_ROOT / "fixtures"


def _make_client(tmp_path: Path, mode: str = "mock", auth: str = "loopback",
                 client_host: tuple[str, int] = ("127.0.0.1", 50000), login: bool = True) -> TestClient:
    os.environ["SCREWIT_MODE"] = mode
    os.environ["SCREWIT_AUTH"] = auth
    os.environ["SCREWIT_QUEUE"] = "inline"
    os.environ["SCREWIT_DATA_DIR"] = str(tmp_path / "data")
    os.environ["SCREWIT_FIXTURES_DIR"] = str(FIXTURES_DIR)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'test.db'}"
    os.environ.pop("SCREWIT_LLM", None)
    os.environ.pop("SCREWIT_CHAT_REPLAY", None)

    from screwit.agent.offline import reset_state
    from screwit.app.db import reset_engine
    from screwit.app.main import create_app
    from screwit.app.settings import reset_settings_cache

    reset_state()  # offline conversation state is process-global; clear between apps
    reset_settings_cache()
    reset_engine()
    app = create_app()
    client = TestClient(app, client=client_host)
    if login:
        client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
    return client


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return _make_client(tmp_path)


@pytest.fixture
def make_client(tmp_path: Path):
    def factory(**kwargs) -> TestClient:
        return _make_client(tmp_path, **kwargs)

    return factory


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
