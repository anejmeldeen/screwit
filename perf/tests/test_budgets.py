"""Budget comparison logic with fake numbers — the part that does not depend on other teams."""

import budgets
from budgets import Direction, Status, grade_metric, grade_metrics, summarize


def _b(key):
    return budgets.BUDGETS_BY_KEY[key]


def test_max_budget_pass_and_fail():
    lcp = _b("landing.lcp_s")
    assert grade_metric(lcp, 2.0, strict=False).status is Status.PASS
    assert grade_metric(lcp, 2.5, strict=False).status is Status.PASS  # boundary is inclusive
    assert grade_metric(lcp, 2.6, strict=False).status is Status.FAIL


def test_min_budget_frame_rate():
    fps = _b("viewer.fps_p50")
    assert fps.direction is Direction.MIN
    assert grade_metric(fps, 30.0, strict=False).status is Status.PASS
    assert grade_metric(fps, 30.1, strict=False).status is Status.PASS
    assert grade_metric(fps, 29.9, strict=False).status is Status.FAIL


def test_flag_false_budget():
    three = _b("landing.loads_three")
    assert three.direction is Direction.FLAG_FALSE
    assert grade_metric(three, False, strict=False).status is Status.PASS
    assert grade_metric(three, True, strict=False).status is Status.FAIL


def test_missing_measurement_skips_in_lenient_mode():
    r = grade_metric(_b("landing.lcp_s"), None, strict=False)
    assert r.status is Status.SKIP
    assert r.measured is None
    assert "no measurement" in r.detail


def test_missing_required_fails_in_strict_mode():
    # landing.lcp_s is required -> FAIL when strict and missing.
    assert grade_metric(_b("landing.lcp_s"), None, strict=True).status is Status.FAIL
    # workspace.first_paint_s is NOT required -> SKIP even in strict mode.
    assert grade_metric(_b("workspace.first_paint_s"), None, strict=True).status is Status.SKIP


def test_grade_metrics_covers_every_budget():
    results = grade_metrics({}, strict=False)
    assert len(results) == len(budgets.BUDGETS)
    assert {r.key for r in results} == set(budgets.BUDGETS_BY_KEY)
    # All SKIP in lenient mode with no measurements.
    assert all(r.status is Status.SKIP for r in results)


def test_summarize_counts_and_overall_ok():
    measured = {
        "landing.lcp_s": 1.0,            # pass
        "landing.js_gzip_kb": 500.0,     # fail (> 200)
        "viewer.fps_p50": 60.0,          # pass
    }
    results = grade_metrics(measured, strict=False)
    summary = summarize(results)
    assert summary.total == len(budgets.BUDGETS)
    assert summary.passed == 2
    assert summary.failed == 1
    assert summary.overall_ok is False
    assert summary.skipped == summary.total - 3


def test_summarize_all_pass_or_skip_is_ok():
    measured = {b.key: (0.1 if b.direction is Direction.MAX else (999.0 if b.direction is Direction.MIN else False))
                for b in budgets.BUDGETS}
    results = grade_metrics(measured, strict=False)
    summary = summarize(results)
    assert summary.failed == 0
    assert summary.overall_ok is True


def test_skip_details_override_is_used():
    # A missing metric with a provided skip_detail reports that exact cause, not the generic one.
    results = grade_metrics(
        {"viewer.fps_p50": None},
        strict=False,
        skip_details={"viewer.fps_p50": "software renderer (SwiftShader), not representative"},
    )
    fps = next(r for r in results if r.key == "viewer.fps_p50")
    assert fps.status is Status.SKIP
    assert "software renderer" in fps.detail


def test_skip_detail_ignored_when_value_present():
    # When a value is present the metric is graded; the skip_detail is irrelevant.
    results = grade_metrics(
        {"viewer.fps_p50": 45.0},
        strict=False,
        skip_details={"viewer.fps_p50": "should not appear"},
    )
    fps = next(r for r in results if r.key == "viewer.fps_p50")
    assert fps.status is Status.PASS
    assert "should not appear" not in fps.detail
