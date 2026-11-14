"""HMAC signing for the local dev store's pre-signed POST and signed GET (SPEC 3.3).

This provides URL/form integrity for the local S3 emulation behind the loopback gate (AGENTS.md
decision 1 and SPEC 8 offline build). It is NOT user authentication: the loopback gate and the demo
network setup are the access control. The signing key comes from SCREWIT_STORAGE_SIGNING_KEY, or is
generated once at process start (dev only); a generated key is never written to disk.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

_ENV_KEY = "SCREWIT_STORAGE_SIGNING_KEY"


def load_signing_key() -> bytes:
    """Return the signing key: from the environment if set, else a fresh random key for this process.

    A generated key lives only in memory, so restarting the dev server invalidates outstanding signed
    URLs, which is acceptable for dev emulation. No credential is ever written to a file (AGENTS.md).
    """
    env = os.environ.get(_ENV_KEY)
    if env:
        return env.encode("utf-8")
    return secrets.token_bytes(32)


def sign(key: bytes, *parts: str) -> str:
    """HMAC-SHA256 over the ordered parts, joined with a separator that cannot appear in a value."""
    msg = "\x1f".join(parts).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify(key: bytes, signature: str, *parts: str) -> bool:
    return hmac.compare_digest(signature, sign(key, *parts))
