"""Storage key validation (SPEC 3.2/3.3).

Keys must be 'scans/<id>/...': no absolute paths, no '..' traversal, no backslashes, no leading slash,
no empty segments. A store never touches the filesystem or S3 with an unvalidated key.
"""

from __future__ import annotations

import posixpath
import re

_SCAN_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class KeyError(ValueError):  # noqa: A001 - a storage-specific key error; imported qualified
    """An object key that is not an acceptable 'scans/<id>/...' key."""


def validate_key(key: str) -> str:
    """Return the key unchanged if valid, else raise KeyError."""
    if not key or not isinstance(key, str):
        raise KeyError("empty key")
    if key.startswith("/"):
        raise KeyError(f"absolute key not allowed: {key!r}")
    if "\\" in key:
        raise KeyError(f"backslash not allowed in key: {key!r}")
    if "\x00" in key:
        raise KeyError("null byte in key")
    parts = key.split("/")
    if parts[0] != "scans":
        raise KeyError(f"key must start with 'scans/': {key!r}")
    if len(parts) < 3:
        raise KeyError(f"key must be 'scans/<id>/...': {key!r}")
    if not _SCAN_ID.match(parts[1]):
        raise KeyError(f"bad scan id segment in key: {key!r}")
    for seg in parts:
        if seg in ("", ".", ".."):
            raise KeyError(f"empty or traversal segment in key: {key!r}")
    # Normalise and confirm nothing escaped.
    norm = posixpath.normpath(key)
    if norm != key or norm.startswith(("/", "..")):
        raise KeyError(f"non-normalised key rejected: {key!r}")
    return key
