"""Report writer: turns graded metrics plus raw suite output into report.json / report.md.

Pure data-in, string/dict-out (plus two file writes), so it is unit-tested now with fake numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from budgets import MetricResult, RunSummary, Status, summarize

_STATUS_MARK = {Status.PASS: "PASS", Status.FAIL: "FAIL", Status.SKIP: "SKIP"}
_STATUS_EMOJI = {Status.PASS: "✅", Status.FAIL: "❌", Status.SKIP: "⚪"}


def _fmt_value(measured: Any, unit: str) -> str:
    if measured is None:
        return "—"
    if isinstance(measured, bool):
        return "yes" if measured else "no"
    if unit in ("gaussians", "triangles"):
        return f"{int(measured):,}"
    if isinstance(measured, float):
        return f"{measured:.3g}"
    return str(measured)


def _fmt_limit(limit: Any, unit: str, direction: str) -> str:
    if isinstance(limit, bool):
        return "must be no"
    arrow = {"max": "<=", "min": ">=", "flag_false": "=="}.get(direction, "")
    if unit in ("gaussians", "triangles"):
        return f"{arrow} {int(limit):,}"
    return f"{arrow} {limit:g}"


def build_report(
    results: list[MetricResult],
    *,
    strict: bool,
    suites: dict[str, Any],
    skips: list[dict[str, str]],
    notes: list[str],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the machine-readable report dictionary.

    - results: graded budget metrics.
    - suites: raw per-suite output (lighthouse, playwright, api, assets) — whatever each produced.
    - skips: list of {suite, reason} for suites that did not run.
    - notes: free-text caveats and instrumentation requests.
    - environment: toolchain versions, bind addresses, hero scan ids, etc.
    """
    summary: RunSummary = summarize(results)
    return {
        "schema": "screwit.perf.report/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "strict": strict,
        "summary": {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "overall_ok": summary.overall_ok,
        },
        "budgets": [
            {
                "key": r.key,
                "label": r.label,
                "status": _STATUS_MARK[r.status],
                "measured": r.measured,
                "limit": r.limit,
                "unit": r.unit,
                "direction": r.direction.value,
                "source": r.source,
                "detail": r.detail,
            }
            for r in results
        ],
        "suites": suites,
        "skips": skips,
        "notes": notes,
        "environment": environment,
    }


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines: list[str] = []
    lines.append("# Screw It performance report")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append(f"- Strict mode: `{report['strict']}`")
    lines.append(
        f"- Result: **{s['passed']} pass, {s['failed']} fail, {s['skipped']} skip** "
        f"of {s['total']} budgets — overall {'OK ✅' if s['overall_ok'] else 'FAIL ❌'}"
    )
    lines.append("")

    lines.append("## Budgets")
    lines.append("")
    lines.append("| Budget | Measured | Limit | Source | Status |")
    lines.append("|---|---|---|---|---|")
    for b in report["budgets"]:
        status = Status(b["status"].lower())
        measured = _fmt_value(b["measured"], b["unit"])
        unit = "" if b["unit"] in ("bool",) else f" {b['unit']}"
        measured_cell = measured if b["measured"] is None else f"{measured}{unit}".strip()
        limit_cell = _fmt_limit(b["limit"], b["unit"], b["direction"])
        detail = f" — {b['detail']}" if b["detail"] else ""
        lines.append(
            f"| {b['label']} | {measured_cell} | {limit_cell} | {b['source']} | "
            f"{_STATUS_EMOJI[status]} {b['status']}{detail} |"
        )
    lines.append("")

    if report["skips"]:
        lines.append("## Suites that did not run")
        lines.append("")
        for sk in report["skips"]:
            lines.append(f"- **{sk['suite']}**: {sk['reason']}")
        lines.append("")

    # Raw suite detail (collapsed as fenced JSON so numbers are inspectable).
    if report["suites"]:
        lines.append("## Suite detail")
        lines.append("")
        for name, data in report["suites"].items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")

    if report["notes"]:
        lines.append("## Notes and instrumentation requests")
        lines.append("")
        for n in report["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["environment"], indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write report.json and report.md into out_dir. Returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# Convenience for callers that hold MetricResult objects directly.
def results_to_dicts(results: list[MetricResult]) -> list[dict[str, Any]]:
    return [asdict(r) | {"status": r.status.value, "direction": r.direction.value} for r in results]
