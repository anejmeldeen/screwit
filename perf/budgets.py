"""Performance budgets for Screw It, and the comparison logic that grades a run.

Source of truth: AGENTS.md "Performance budgets" (phones are a first-class target).
This module is pure Python with no I/O so it can be unit-tested now, before the web and
API branches merge. The runners (Lighthouse, Playwright, API latency, asset sizes) feed
measured numbers into `grade_metrics` and the report writer renders the result.

Budget direction:
- "max": the measured value must be <= limit (latency, size, counts).
- "min": the measured value must be >= limit (frame rate).
- "flag_false": a boolean that must be False to pass (e.g. "three.js loaded on landing").

A metric with no measured value is reported as SKIP (not a failure) unless the run is in
strict mode and the metric is marked `required`, in which case a missing value is a FAIL.
This lets the suite run and report cleanly before other teams' branches land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Direction(StrEnum):
    MAX = "max"  # measured <= limit
    MIN = "min"  # measured >= limit
    FLAG_FALSE = "flag_false"  # measured boolean must be False


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # no measurement available yet


@dataclass(frozen=True)
class Budget:
    key: str
    label: str  # human description, matches AGENTS.md wording
    direction: Direction
    limit: float | bool
    unit: str
    # If True, a missing measurement FAILs in strict mode (otherwise it SKIPs).
    required: bool = False
    # Which suite is expected to produce the number (for the report).
    source: str = ""


# ---------------------------------------------------------------------------
# The budget table. Keep the wording aligned with AGENTS.md "Performance budgets".
# ---------------------------------------------------------------------------
BUDGETS: list[Budget] = [
    # Landing page, Lighthouse mobile preset.
    Budget("landing.lcp_s", "Landing page LCP (mobile preset)", Direction.MAX, 2.5, "s",
           required=True, source="lighthouse"),
    Budget("landing.js_gzip_kb", "Landing page JS transferred, gzipped", Direction.MAX, 200.0, "KB",
           required=True, source="lighthouse"),
    Budget("landing.loads_three", "Landing page loads three.js", Direction.FLAG_FALSE, False, "bool",
           required=True, source="lighthouse"),
    Budget("landing.loads_spark", "Landing page loads Spark", Direction.FLAG_FALSE, False, "bool",
           required=True, source="lighthouse"),
    # Workspace shell first paint.
    Budget("workspace.first_paint_s", "Workspace first paint (shell, panels, status)", Direction.MAX, 2.0, "s",
           source="playwright"),
    # First 3D view of a hero scan on slow 4G.
    Budget("viewer.first_mesh_s", "First 3D mesh visible on slow 4G", Direction.MAX, 5.0, "s",
           source="playwright"),
    # Viewer frame rate on the reference phone while orbiting.
    Budget("viewer.fps_p50", "Viewer frame rate while orbiting (p50)", Direction.MIN, 30.0, "fps",
           source="playwright"),
    # Phone assets.
    Budget("assets.splat_gaussians", "Mobile splat gaussian count (splat_mobile_uri)", Direction.MAX,
           400_000, "gaussians", source="assets"),
    Budget("assets.mesh_triangles", "Mobile mesh triangle count (mesh_mobile_uri)", Direction.MAX, 50_000, "triangles",
           source="assets"),
    # Chat first token (mock mode: immediate, i.e. well under 2 s).
    Budget("chat.first_token_s", "Chat first token on screen after send", Direction.MAX, 2.0, "s",
           source="playwright"),
]

BUDGETS_BY_KEY: dict[str, Budget] = {b.key: b for b in BUDGETS}


@dataclass
class MetricResult:
    key: str
    label: str
    status: Status
    measured: float | bool | None
    limit: float | bool
    unit: str
    direction: Direction
    source: str
    detail: str = ""


def _compare(direction: Direction, measured: float | bool, limit: float | bool) -> bool:
    """Return True if the measured value satisfies the budget."""
    if direction is Direction.MAX:
        return float(measured) <= float(limit)
    if direction is Direction.MIN:
        return float(measured) >= float(limit)
    if direction is Direction.FLAG_FALSE:
        return bool(measured) is False
    raise ValueError(f"unknown direction {direction!r}")


def grade_metric(budget: Budget, measured: float | bool | None, *, strict: bool,
                 detail: str = "") -> MetricResult:
    """Grade a single measured value against its budget.

    `measured is None` means the suite could not produce the number (dependency not merged,
    asset missing, server not up). That is SKIP, unless strict mode and the budget is required,
    which is FAIL.
    """
    if measured is None:
        status = Status.FAIL if (strict and budget.required) else Status.SKIP
        if not detail:
            detail = "no measurement available"
        return MetricResult(budget.key, budget.label, status, None, budget.limit,
                            budget.unit, budget.direction, budget.source, detail)
    ok = _compare(budget.direction, measured, budget.limit)
    return MetricResult(budget.key, budget.label, Status.PASS if ok else Status.FAIL,
                        measured, budget.limit, budget.unit, budget.direction,
                        budget.source, detail)


def grade_metrics(measured: dict[str, float | bool | None], *, strict: bool,
                  skip_details: dict[str, str] | None = None) -> list[MetricResult]:
    """Grade every budget against a dict of measured values keyed by budget key.

    Keys absent from `measured` are treated as None (SKIP, or FAIL if strict+required).
    `skip_details[key]` overrides the detail string for a metric with no measurement, so a SKIP can
    state its precise cause (e.g. "software renderer, not graded") instead of a generic message.
    """
    skip_details = skip_details or {}
    results: list[MetricResult] = []
    for budget in BUDGETS:
        value = measured.get(budget.key)
        detail = skip_details.get(budget.key, "") if value is None else ""
        results.append(grade_metric(budget, value, strict=strict, detail=detail))
    return results


@dataclass
class RunSummary:
    total: int
    passed: int
    failed: int
    skipped: int
    overall_ok: bool  # True when there are zero FAILs
    results: list[MetricResult] = field(default_factory=list)


def summarize(results: list[MetricResult]) -> RunSummary:
    passed = sum(1 for r in results if r.status is Status.PASS)
    failed = sum(1 for r in results if r.status is Status.FAIL)
    skipped = sum(1 for r in results if r.status is Status.SKIP)
    return RunSummary(
        total=len(results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_ok=failed == 0,
        results=results,
    )
