"""Read-only storage over the repo's fixtures/ tree, for mock mode (SPEC 8).

Keys 'scans/<id>/...' map to fixtures/scans/<id>/...; writes are refused. signed_url returns a local
asset URL so the mock API can serve fixture files through the same route shape as the dev store.
"""

from __future__ import annotations

import mimetypes
import shutil
import time
from pathlib import Path

from screwit.contracts import PresignedPost
from screwit.storage import signing
from screwit.storage.keys import validate_key

_ASSET_PATH = "/api/assets"


class FixtureStorage:
    """Read-only Storage over fixtures/."""

    def __init__(self, fixtures_dir: Path, signing_key: bytes | None = None) -> None:
        self.fixtures_dir = Path(fixtures_dir).resolve()
        self._signing_key = signing_key if signing_key is not None else signing.load_signing_key()

    def _path(self, key: str) -> Path:
        validate_key(key)
        p = (self.fixtures_dir / key).resolve()
        if not str(p).startswith(str(self.fixtures_dir) + "/"):
            raise ValueError(f"resolved path escapes fixtures dir: {key!r}")
        return p

    def presign_upload(self, key: str, content_types: list[str], max_bytes: int,
                       expires_s: int = 600) -> PresignedPost:
        raise PermissionError("fixture storage is read-only; uploads are not allowed in mock mode")

    def signed_url(self, key: str, expires_s: int = 900) -> str:
        validate_key(key)
        exp = str(int(time.time()) + int(expires_s))
        sig = signing.sign(self._signing_key, key, exp)
        return f"{_ASSET_PATH}/{key}?exp={exp}&sig={sig}"

    def head(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.is_file():
            return None
        ct = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return {"size": p.stat().st_size, "content_type": ct}

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        raise PermissionError("fixture storage is read-only")

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        raise PermissionError("fixture storage is read-only")

    def local_path(self, key: str, scratch: Path) -> Path:
        src = self._path(key)
        if not src.is_file():
            raise FileNotFoundError(key)
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        dest = scratch / Path(key).name
        shutil.copyfile(src, dest)
        return dest
