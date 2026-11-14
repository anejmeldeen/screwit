"""Object storage: local-disk dev store, read-only fixture store, and private S3 (SPEC 3.2/3.3).

All stores key objects by 'scans/<scan_id>/...'. Keys are validated against traversal and absolute
paths before any filesystem or S3 access. The local store emulates S3 pre-signed POST semantics so the
same two-client rule (API client vs storage client, SPEC 3.3) holds in dev.
"""

from __future__ import annotations

from screwit.storage.fixture import FixtureStorage
from screwit.storage.keys import KeyError as StorageKeyError
from screwit.storage.keys import validate_key
from screwit.storage.local import LocalStorage
from screwit.storage.s3 import S3Storage

__all__ = [
    "LocalStorage",
    "FixtureStorage",
    "S3Storage",
    "validate_key",
    "StorageKeyError",
]
