"""Percentile helper and the API suite's clean-skip behaviour when no server is up."""

import pytest

from api_latency import _percentile, run_api_suite


def test_percentile_nearest_rank():
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _percentile(vals, 50) == 50
    assert _percentile(vals, 95) == 100
    assert _percentile(vals, 100) == 100


def test_percentile_single_value():
    assert _percentile([7.0], 50) == 7.0
    assert _percentile([7.0], 95) == 7.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        _percentile([], 50)


def test_run_api_suite_skips_when_unreachable():
    # Nothing is listening on this loopback port; the suite must SKIP, not raise.
    out = run_api_suite("http://127.0.0.1:59999", timeout_s=1.0)
    assert out["skipped"] is True
    assert "not reachable" in out["reason"] or "error" in out["reason"]
