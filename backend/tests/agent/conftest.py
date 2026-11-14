"""Agent-test fixtures: a mock-mode Registry wired to the fixtures tree (no network, no LLM)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT.parent / "screw-it" / "fixtures"
if not FIXTURES_DIR.exists():
    FIXTURES_DIR = REPO_ROOT / "fixtures"

# Synthetic non-plumbing scan (FixtureData reads <dir>/scans/<id>). Not under a `data/` directory:
# the repo .gitignore drops every `data/` (runtime storage), which silently left this out of git.
SYNTHETIC_FIXTURES_DIR = Path(__file__).resolve().parent / "synthetic"


def _build_registry(tmp_path: Path, fixtures_dir: Path, llm: str | None = None):
    os.environ["SCREWIT_MODE"] = "mock"
    os.environ["SCREWIT_QUEUE"] = "inline"
    os.environ["SCREWIT_DATA_DIR"] = str(tmp_path / "data")
    os.environ["SCREWIT_FIXTURES_DIR"] = str(fixtures_dir)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'test.db'}"
    os.environ.pop("SCREWIT_CHAT_REPLAY", None)
    if llm is None:
        os.environ.pop("SCREWIT_LLM", None)
    else:
        os.environ["SCREWIT_LLM"] = llm

    from screwit.app.db import init_db, reset_engine
    from screwit.app.registry import build_registry
    from screwit.app.settings import reset_settings_cache, settings

    reset_settings_cache()
    reset_engine()
    cfg = settings()
    init_db()
    return build_registry(cfg)


@pytest.fixture
def registry(tmp_path: Path):
    return _build_registry(tmp_path, FIXTURES_DIR)


@pytest.fixture
def synthetic_registry(tmp_path: Path):
    """A mock-mode registry over the synthetic non-plumbing scan (scn_bracket)."""
    return _build_registry(tmp_path, SYNTHETIC_FIXTURES_DIR)


@pytest.fixture
def make_registry(tmp_path: Path):
    def factory(fixtures_dir: Path | None = None, llm: str | None = None):
        return _build_registry(tmp_path, fixtures_dir or FIXTURES_DIR, llm=llm)

    return factory
