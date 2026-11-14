"""MediaValidator over every file in fixtures/media, asserting the manifest's verdict (SPEC 3.3,
defect 4). This runs the realistic phone-media fixtures X2 committed to main, including the
truncated Android clip that the header probe alone wrongly accepted."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from screwit.contracts import Scan, ScanInput, Stage
from screwit.contracts.models import STAGE_ORDER
from screwit.storage.local import LocalStorage
from screwit.storage.validation import MediaValidator

WT_ROOT = Path(__file__).resolve().parents[3]
MEDIA = WT_ROOT / "fixtures" / "media"
MANIFEST = MEDIA / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="fixtures/media not present")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _scan(kind: str, key: str) -> Scan:
    return Scan(scan_id="media", created_at=datetime.now(UTC),
                input=ScanInput(type=kind, uri=key), status="validating",
                reconstruction_revision=0,
                stages=[Stage(name=n, state="pending") for n in STAGE_ORDER])


def _video_cases():
    if not MANIFEST.exists():
        return []
    return [(f["name"], f["expected_validator"]) for f in _manifest()["files"]]


@pytest.fixture
def store(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path / "data", signing_key=b"k" * 32)


@pytest.fixture
def validator() -> MediaValidator:
    return MediaValidator()


@pytest.mark.parametrize("name,expected", _video_cases())
def test_video_fixture_verdict(name, expected, store, validator):
    key = f"scans/media/input/{name}"
    dst = store.data_dir / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MEDIA / name, dst)

    try:
        info = validator.validate(_scan("video", key), store)
        verdict = "accept"
    except ValueError:
        verdict = "reject"
        info = None

    assert verdict == expected["verdict"], f"{name}: expected {expected['verdict']}, got {verdict}"

    if verdict == "accept":
        # Reason class cross-checks for the accepted clips (HDR flag, codec).
        entry = next(f for f in _manifest()["files"] if f["name"] == name)
        assert info is not None
        assert info.codec == entry["codec"]
        if entry.get("hdr"):
            assert info.hdr is True, f"{name}: HDR not detected"


def test_every_media_file_is_covered():
    # Guards against a new fixture slipping in without a verdict.
    for f in _manifest()["files"]:
        assert "expected_validator" in f and "verdict" in f["expected_validator"], f["name"]


def _photo_cases():
    if not MANIFEST.exists():
        return []
    return [p["name"] for p in _manifest().get("photos", [])]


@pytest.mark.parametrize("name", _photo_cases())
def test_photo_fixture_accepts(name, store, validator):
    # Every photo in the manifest is a valid JPEG and must validate (accept).
    key = f"scans/media/input/{Path(name).name}"
    dst = store.data_dir / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MEDIA / name, dst)
    info = validator.validate(_scan("photos", key), store)
    assert info.kind == "photos"
    assert info.width > 0 and info.height > 0
