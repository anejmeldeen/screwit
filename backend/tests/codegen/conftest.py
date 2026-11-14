"""Shared fixtures for the codegen tests.

`docker_or_skip` skips cleanly when the daemon or the sandbox image is unavailable, so the suite
behaves on a machine without Docker. A tiny in-memory Storage fake stands in for P5's Storage (not
committed to main), implementing only the methods the codegen `design()` entry point calls.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from screwit.codegen.sandbox import DEFAULT_IMAGE, docker_available

SANDBOX_IMAGE = os.environ.get("SCREWIT_CODEGEN_IMAGE", DEFAULT_IMAGE)


def _image_present(image: str) -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def docker_or_skip() -> str:
    if not docker_available():
        pytest.skip("docker info failed; skipping container-dependent codegen tests")
    if not _image_present(SANDBOX_IMAGE):
        pytest.skip(f"sandbox image {SANDBOX_IMAGE!r} not built; run deploy/codegen-sandbox build")
    return SANDBOX_IMAGE


class FakeStorage:
    """Minimal Storage: keeps bytes in a dict keyed by storage key; records content types."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        self.objects[key] = Path(path).read_bytes()
        self.content_types[key] = content_type

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data
        self.content_types[key] = content_type

    def head(self, key: str) -> dict | None:
        if key not in self.objects:
            return None
        return {"size": len(self.objects[key]), "content_type": self.content_types.get(key, "")}


class FakeProfile:
    """Stand-in for PrinterProfile with only the fields codegen reads."""

    build_volume_mm = (250.0, 210.0, 220.0)
    min_wall_mm = 1.2
    max_overhang_deg = 50.0


@pytest.fixture
def fake_storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def fake_profile() -> FakeProfile:
    return FakeProfile()
