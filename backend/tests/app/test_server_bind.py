"""The server refuses a non-loopback bind (AGENTS.md decision 1, SPEC 7 P4)."""

from __future__ import annotations

import pytest

from screwit.app.__main__ import refuse_non_loopback
from screwit.app.settings import is_loopback


def test_refuse_wildcard_bind():
    with pytest.raises(SystemExit) as exc:
        refuse_non_loopback("0.0.0.0")
    assert "127.0.0.1" in str(exc.value)


def test_refuse_public_bind():
    with pytest.raises(SystemExit):
        refuse_non_loopback("203.0.113.9")


def test_accept_loopback_binds():
    refuse_non_loopback("127.0.0.1")  # no raise
    refuse_non_loopback("::1")


def test_is_loopback():
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.5.6.7")
    assert is_loopback("::1")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("10.0.0.1")
    assert not is_loopback("example.com")
