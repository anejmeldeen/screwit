"""Shared test helpers for perception: a tmpdir-backed fake Storage (SPEC interfaces.Storage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from screwit.contracts import PresignedPost


class FakeStorage:
    """Local-disk Storage fake, enough for perception tests (writes parts/renders/graph)."""

    def __init__(self, root: Path):
        self.root = root

    def _p(self, key: str) -> Path:
        return self.root / key

    def presign_upload(self, key, content_types, max_bytes, expires_s=600) -> PresignedPost:
        return PresignedPost(url="http://localhost/upload", fields={"key": key}, key=key)

    def signed_url(self, key, expires_s=900) -> str:
        return f"file://{self._p(key)}"

    def head(self, key):
        p = self._p(key)
        if not p.exists():
            return None
        return {"size": p.stat().st_size, "content_type": "application/octet-stream"}

    def get_bytes(self, key) -> bytes:
        return self._p(key).read_bytes()

    def put_bytes(self, key, data, content_type) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def local_path(self, key, scratch) -> Path:
        return self._p(key)

    def put_file(self, key, path, content_type) -> None:
        self.put_bytes(key, Path(path).read_bytes(), content_type)


@pytest.fixture
def storage(tmp_path) -> FakeStorage:
    return FakeStorage(tmp_path)
