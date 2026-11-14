"""Report writer with fake numbers: JSON shape, markdown rendering, and file writing."""

import json

from budgets import grade_metrics
from report import build_report, render_markdown, write_report


def _fake_report(strict=False):
    measured = {
        "landing.lcp_s": 1.8,
        "landing.js_gzip_kb": 150.0,
        "landing.loads_three": False,
        "landing.loads_spark": False,
        "viewer.fps_p50": 42.0,
        "chat.first_token_s": 0.3,
        # workspace.first_paint_s, viewer.first_mesh_s and asset budgets intentionally missing -> SKIP
    }
    results = grade_metrics(measured, strict=strict)
    return build_report(
        results,
        strict=strict,
        suites={"lighthouse": {"landing": {"lcp_s": 1.8}}},
        skips=[{"suite": "playwright", "reason": "web branch not merged"}],
        notes=["Playwright needs viewer:first-mesh from P1."],
        environment={"repo_rev": "deadbee", "node": "v22"},
    )


def test_report_json_shape():
    rpt = _fake_report()
    assert rpt["schema"] == "screwit.perf.report/1"
    assert rpt["summary"]["passed"] == 6
    assert rpt["summary"]["failed"] == 0
    assert rpt["summary"]["skipped"] == rpt["summary"]["total"] - 6
    assert rpt["summary"]["overall_ok"] is True
    keys = {b["key"] for b in rpt["budgets"]}
    assert "landing.lcp_s" in keys and "assets.mesh_triangles" in keys
    # It is JSON-serialisable.
    json.dumps(rpt)


def test_report_marks_failure():
    measured = {"landing.js_gzip_kb": 500.0, "landing.loads_three": True}
    results = grade_metrics(measured, strict=False)
    rpt = build_report(results, strict=False, suites={}, skips=[], notes=[], environment={})
    assert rpt["summary"]["failed"] == 2
    assert rpt["summary"]["overall_ok"] is False


def test_markdown_renders_table_and_sections():
    md = render_markdown(_fake_report())
    assert "# Screw It performance report" in md
    assert "| Budget | Measured | Limit | Source | Status |" in md
    assert "Landing page LCP" in md
    assert "PASS" in md and "SKIP" in md
    assert "## Suites that did not run" in md
    assert "web branch not merged" in md
    assert "## Notes and instrumentation requests" in md
    assert "viewer:first-mesh" in md
    # No measurement renders as an em dash.
    assert "—" in md


def test_write_report_creates_both_files(tmp_path):
    rpt = _fake_report()
    json_path, md_path = write_report(rpt, tmp_path)
    assert json_path.exists() and md_path.exists()
    reloaded = json.loads(json_path.read_text())
    assert reloaded["summary"] == rpt["summary"]
    assert md_path.read_text().startswith("# Screw It performance report")
