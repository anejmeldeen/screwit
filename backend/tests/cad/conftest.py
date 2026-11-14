"""Shared fixtures for the CAD tests (scan-frame millimetres throughout)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from screwit.cad.service import CadRunner
from screwit.contracts import CadRequest, Part
from screwit.storage.local import LocalStorage

# Fixtures and the printer profile live in this worktree's checkout.
WT_ROOT = Path(__file__).resolve().parents[3]
FIX = WT_ROOT / "fixtures" / "scans" / "scn_knob"
PROFILES = WT_ROOT / "profiles"


@pytest.fixture
def profiles_dir() -> Path:
    return PROFILES


@pytest.fixture
def knob_request() -> CadRequest:
    return CadRequest.model_validate_json((FIX / "cad_request.json").read_text())


@pytest.fixture
def knob_parts() -> dict[str, Part]:
    data = json.loads((FIX / "parts.json").read_text())
    return {p["part_id"]: Part.model_validate(p) for p in data}


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path / "data")


@pytest.fixture
def runner(profiles_dir) -> CadRunner:
    return CadRunner(profiles_dir)


@pytest.fixture
def tmp_store() -> LocalStorage:
    return LocalStorage(Path(tempfile.mkdtemp(prefix="cadstore_")))
