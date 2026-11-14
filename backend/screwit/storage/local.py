"""Local-disk dev object store with S3 pre-signed POST semantics emulated (SPEC 3.3, AGENTS.md 3).

Files live under a data directory, one path per key. Uploads go through presign_upload -> the P4 route
POST /api/storage/upload -> accept_upload, which enforces expiry, HMAC signature, content type and size
while streaming. signed_url + verify_signed back the GET route /api/assets/<key>?exp=..&sig=..
"""

from __future__ import annotations

import mimetypes
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from screwit.contracts import PresignedPost
from screwit.storage import signing
from screwit.storage.keys import validate_key

_UPLOAD_PATH = "/api/storage/upload"
_ASSET_PATH = "/api/assets"
_CHUNK = 1024 * 1024


class LocalStorage:
    """Storage protocol over local disk; key -> `<data_dir>/<key>`."""

    def __init__(self, data_dir: Path, signing_key: bytes | None = None) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._signing_key = signing_key if signing_key is not None else signing.load_signing_key()

    # -- key -> path, guarded ----------------------------------------------

    def _path(self, key: str) -> Path:
        validate_key(key)
        p = (self.data_dir / key).resolve()
        if not str(p).startswith(str(self.data_dir) + "/") and p != self.data_dir:
            raise ValueError(f"resolved path escapes data dir: {key!r}")
        return p

    # -- pre-signed upload (POST emulation) --------------------------------

    def presign_upload(self, key: str, content_types: list[str], max_bytes: int,
                       expires_s: int = 600) -> PresignedPost:
        validate_key(key)
        expires = str(int(time.time()) + int(expires_s))
        allow = ",".join(sorted(content_types))
        max_s = str(int(max_bytes))
        signature = signing.sign(self._signing_key, key, allow, max_s, expires)
        fields = {
            "key": key,
            "content_type_allow": allow,
            "max_bytes": max_s,
            "expires": expires,
            "signature": signature,
        }
        return PresignedPost(url=_UPLOAD_PATH, fields=fields, key=key)

    def accept_upload(self, fields: dict[str, str], filename: str, stream: BinaryIO) -> None:
        """Validate the pre-signed fields and persist the streamed body (SPEC 3.3).

        Raises ValueError on a bad signature, expiry, disallowed content type or oversize body. The
        body is streamed to disk and truncated with an error if it exceeds max_bytes.
        """
        required = {"key", "content_type_allow", "max_bytes", "expires", "signature"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"upload fields missing: {sorted(missing)}")
        key = fields["key"]
        validate_key(key)
        allow = fields["content_type_allow"]
        max_s = fields["max_bytes"]
        expires = fields["expires"]
        signature = fields["signature"]

        if not signing.verify(self._signing_key, signature, key, allow, max_s, expires):
            raise ValueError("upload signature does not match")
        try:
            if int(time.time()) > int(expires):
                raise ValueError("upload URL has expired")
            max_bytes = int(max_s)
        except ValueError as e:
            raise ValueError(f"bad upload fields: {e}") from None

        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        allowed = set(allow.split(",")) if allow else set()
        if allowed and content_type not in allowed:
            raise ValueError(f"content type {content_type!r} not in allowed {sorted(allowed)}")

        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        written = 0
        try:
            with tmp.open("wb") as out:
                while True:
                    chunk = stream.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"upload exceeds max_bytes {max_bytes}")
                    out.write(chunk)
            if written == 0:
                raise ValueError("empty upload")
            tmp.replace(dest)
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- signed GET --------------------------------------------------------

    def signed_url(self, key: str, expires_s: int = 900) -> str:
        validate_key(key)
        exp = str(int(time.time()) + int(expires_s))
        sig = signing.sign(self._signing_key, key, exp)
        return f"{_ASSET_PATH}/{key}?exp={exp}&sig={sig}"

    def verify_signed(self, key: str, exp: str, sig: str) -> bool:
        """True if (key, exp, sig) is a valid, unexpired signed GET (used by the P4 asset route)."""
        try:
            validate_key(key)
            if int(time.time()) > int(exp):
                return False
        except (ValueError, Exception):  # noqa: BLE001
            return False
        return signing.verify(self._signing_key, sig, key, exp)

    # -- direct object access ----------------------------------------------

    def head(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.is_file():
            return None
        ct = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return {"size": p.stat().st_size, "content_type": ct}

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, p)

    def local_path(self, key: str, scratch: Path) -> Path:
        # Already a local file; return it directly (no download needed).
        p = self._path(key)
        if not p.is_file():
            raise FileNotFoundError(key)
        return p
