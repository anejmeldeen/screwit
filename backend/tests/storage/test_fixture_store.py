"""FixtureStorage: read-only access over fixtures/ for mock mode (SPEC 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from screwit.storage.fixture import FixtureStorage

WT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = WT_ROOT / "fixtures"


@pytest.fixture
def store() -> FixtureStorage:
    return FixtureStorage(FIXTURES, signing_key=b"k" * 32)


def test_reads_fixture_json(store):
    key = "scans/scn_knob/cad_request.json"
    assert store.head(key) is not None
    data = store.get_bytes(key)
    assert b"cad_07" in data


def test_missing_key_head_none(store):
    assert store.head("scans/scn_knob/nope.json") is None


def test_writes_refused(store, tmp_path):
    with pytest.raises(PermissionError):
        store.put_bytes("scans/scn_knob/x.json", b"{}", "application/json")
    with pytest.raises(PermissionError):
        store.put_file("scans/scn_knob/x.json", tmp_path / "f", "application/json")
    with pytest.raises(PermissionError):
        store.presign_upload("scans/scn_knob/x.json", ["application/json"], 100)


def test_signed_url_shape(store):
    url = store.signed_url("scans/scn_knob/cad_request.json")
    assert url.startswith("/api/assets/scans/scn_knob/cad_request.json?exp=")


def test_local_path_copies_to_scratch(store, tmp_path):
    p = store.local_path("scans/scn_knob/cad_request.json", tmp_path / "scratch")
    assert p.exists()
    assert p.read_bytes()[:1] == b"{"


def test_traversal_rejected(store):
    from screwit.storage.keys import KeyError as StorageKeyError

    with pytest.raises(StorageKeyError):
        store.get_bytes("scans/../../etc/passwd")
