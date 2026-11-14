"""Aggregate suite outputs into perf/out/report.{json,md}.

Called by run.sh after the suites have run. Reads the Lighthouse and Playwright JSON (written by
the Node runners), runs the Python API-latency and asset-size suites directly, grades everything
against budgets.py, and writes the report with report.py.

Everything degrades to SKIP (or FAIL in --strict) when a dependency is missing, so this never
errors before the web/API branches merge.

Usage: python collect.py --out <dir> [--api-base URL] [--web-base URL]
       [--lighthouse PATH] [--playwright PATH] [--repo DIR] [--strict]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

import asset_sizes
import report as report_mod
from api_latency import run_api_suite
from budgets import grade_metrics

HERO_SCANS = ["scn_trap", "scn_knob"]


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _git_rev(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _node_version() -> str:
    try:
        return subprocess.check_output(["node", "--version"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _reconstruction_for(scan_id: str, api_base: str | None, repo: Path) -> dict | None:
    """Fetch reconstruction.json from the API (live) or the local fixtures (fallback)."""
    if api_base:
        try:
            with httpx.Client(base_url=api_base, timeout=10.0) as c:
                r = c.get(f"/api/scans/{scan_id}/reconstruction")
                if r.status_code == 200:
                    return r.json()
        except Exception:  # noqa: BLE001
            pass
    fixture = repo / "fixtures" / "scans" / scan_id / "reconstruction.json"
    if fixture.exists():
        return json.loads(fixture.read_text(encoding="utf-8"))
    return None


def _make_fetcher(api_base: str | None, repo: Path):
    """Return fetch(uri)->bytes, or None if nothing can serve assets.

    Live: the reconstruction returns each asset field as a signed URL beginning with the asset
    path (e.g. "/api/assets/scans/scn_trap/mobile/object.glb?exp=...&sig=..."). The fetcher GETs that
    path against the API base unchanged; a bare key (no leading "/" or scheme) is fetched under the
    signed asset route as a best effort. Fallback: read from the local data/ store when present.
    """
    data_dir = repo / "data"
    if api_base:
        client = httpx.Client(base_url=api_base, timeout=30.0, follow_redirects=True)

        def fetch(uri: str) -> bytes:
            # A signed URL / absolute path is fetched verbatim; a bare key goes under /api/assets/.
            if uri.startswith(("http://", "https://", "/")):
                target = uri
            else:
                target = f"/api/assets/{uri}"
            r = client.get(target)
            r.raise_for_status()
            return r.content

        # Reachability probe: the API answers (even a 401/403/404 means it is up and routing).
        try:
            probe = client.get("/api/demo-scans")
            if probe.status_code >= 500:
                client.close()
                return None
        except Exception:  # noqa: BLE001
            client.close()
            return None
        return fetch

    if data_dir.exists():
        def fetch_local(uri: str) -> bytes:
            return (data_dir / uri).read_bytes()

        return fetch_local
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--web-base", default=None)
    ap.add_argument("--lighthouse", default=None)
    ap.add_argument("--playwright", default=None)
    ap.add_argument("--repo", default=None, help="repo root for fixtures/data (default: ../ of perf)")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)

    suites: dict[str, Any] = {}
    skips: list[dict[str, str]] = []
    notes: list[str] = []
    measured: dict[str, float | bool | None] = {}
    skip_details: dict[str, str] = {}

    # --- Lighthouse ---
    _LH_BUDGETS = ["landing.lcp_s", "landing.js_gzip_kb", "landing.loads_three", "landing.loads_spark"]
    lh = _load_json(args.lighthouse)
    if lh is None:
        reason = f"no lighthouse output at {args.lighthouse!r} (runner not invoked)"
        skips.append({"suite": "lighthouse", "reason": reason})
        for k in _LH_BUDGETS:
            skip_details[k] = f"lighthouse suite did not run: {reason}"
    elif lh.get("skipped"):
        reason = lh.get("reason", "skipped")
        skips.append({"suite": "lighthouse", "reason": reason})
        for k in _LH_BUDGETS:
            skip_details[k] = f"lighthouse suite skipped: {reason}"
    else:
        suites["lighthouse"] = lh
        landing = lh.get("landing", {})
        measured["landing.lcp_s"] = landing.get("lcp_s")
        measured["landing.js_gzip_kb"] = landing.get("js_transferred_kb")
        measured["landing.loads_three"] = landing.get("loads_three")
        measured["landing.loads_spark"] = landing.get("loads_spark")

    # --- Playwright ---
    _PW_BUDGETS = ["workspace.first_paint_s", "viewer.first_mesh_s", "viewer.fps_p50", "chat.first_token_s"]
    pw = _load_json(args.playwright)
    if pw is None:
        reason = (f"no playwright output at {args.playwright!r} (runner not invoked, "
                  "e.g. web server not up or SKIP_BROWSER=1)")
        skips.append({"suite": "playwright", "reason": reason})
        for k in _PW_BUDGETS:
            skip_details[k] = f"playwright suite did not run: {reason}"
    elif pw.get("skipped"):
        reason = pw.get("reason", "skipped")
        skips.append({"suite": "playwright", "reason": reason})
        for k in _PW_BUDGETS:
            skip_details[k] = f"playwright suite skipped: {reason}"
        notes.append(f"Playwright needs these performance marks from P1: {json.dumps(pw.get('marks_needed', {}))}")
    else:
        suites["playwright"] = pw
        mob = pw.get("mobile", {})
        # Whether this host's WebGL rendering is hardware (representative of a phone) or software
        # (SwiftShader on a GPU-less host). GPU-bound budgets grade only when representative.
        representative = mob.get("fps_representative", False)
        shell = mob.get("shell_ms")
        measured["workspace.first_paint_s"] = (shell / 1000.0) if shell is not None else None
        if shell is None:
            skip_details["workspace.first_paint_s"] = (
                "no shell:ready mark and status-bar never became visible within 15 s")
        fm = mob.get("first_mesh_ms")
        fm_desktop = pw.get("desktop", {}).get("first_mesh_ms")
        # First mesh on slow-4G: on a GPU-less host WebGL is software (SwiftShader), so the mesh
        # load+decode time is confounded by software decode and splat-stream contention, not
        # representative of the reference phone. Grade only on representative (hardware) rendering;
        # otherwise report the number (mobile throttled + desktop unthrottled) and SKIP with cause.
        if representative and fm is not None:
            measured["viewer.first_mesh_s"] = fm / 1000.0
        else:
            measured["viewer.first_mesh_s"] = None
            skip_details["viewer.first_mesh_s"] = (
                f"software renderer ({mob.get('webgl_renderer')}), not representative — "
                f"first mesh measured {fm:.0f} ms mobile (4x CPU, slow-4G), {fm_desktop:.0f} ms "
                f"desktop unthrottled; confounded by software decode + splat-stream contention. "
                "Real-phone /perf page is the authority (DEVICE_TEST.md)."
            ) if fm is not None else (
                f"software renderer ({mob.get('webgl_renderer')}); viewer:first-mesh never fired")
        ct = mob.get("chat", {}).get("first_token_ms")
        measured["chat.first_token_s"] = (ct / 1000.0) if ct is not None else None
        if ct is None:
            skip_details["chat.first_token_s"] = (
                mob.get("chat", {}).get("detail")
                or "no chat:first-token mark and no assistant text observed")

        # Frame rate: this host has no GPU, so Chromium renders WebGL in software (SwiftShader).
        # A software fps is NOT representative of the reference phone, so it is reported as
        # informational and NOT graded against the 30 fps budget (which would be a false fail).
        # The real-phone /perf page (DEVICE_TEST.md) is the authority for this budget.
        fps = mob.get("orbit_fps", {})
        if representative:
            measured["viewer.fps_p50"] = fps.get("p50_fps")
        else:
            measured["viewer.fps_p50"] = None  # leave as SKIP with the renderer reason below
            skip_details["viewer.fps_p50"] = (
                f"software renderer ({mob.get('webgl_renderer')}), not representative — "
                f"measured p50={fps.get('p50_fps')} fps over {fps.get('frames')} frames; "
                "real-phone /perf page is the authority (DEVICE_TEST.md)")
        suites["playwright"]["fps_grading"] = {
            "graded_against_budget": bool(representative),
            "webgl_renderer": mob.get("webgl_renderer"),
            "measured_p50_fps": fps.get("p50_fps"),
            "measured_worst_fps": fps.get("worst_fps"),
            "frames": fps.get("frames"),
        }
        if not representative:
            notes.append(
                f"Viewer fps NOT graded: Chromium on this host renders WebGL in software "
                f"(renderer={mob.get('webgl_renderer')!r}), so orbit fps "
                f"(p50={fps.get('p50_fps')}, worst={fps.get('worst_fps')}, frames={fps.get('frames')}) "
                "is not representative of the reference phone. The real-phone /perf page is the "
                "authority for the 30 fps budget (DEVICE_TEST.md)."
            )
        # Record which signal produced the shell/first-token numbers (mark vs DOM fallback).
        if shell is not None:
            notes.append(f"Workspace first paint measured via {mob.get('shell_source')} "
                         f"(mobile {shell:.0f} ms; desktop {pw.get('desktop', {}).get('shell_ms')} ms).")
        if ct is not None:
            notes.append(f"Chat first token measured via {mob.get('chat', {}).get('source')} "
                         f"({ct:.0f} ms, mock mode).")
        if fm is not None:
            notes.append(f"First mesh (informational, software renderer): {fm:.0f} ms mobile "
                         f"(4x CPU, slow-4G), {fm_desktop:.0f} ms desktop unthrottled. "
                         "Not graded here; real-phone /perf is the authority for the under-5 s budget.")

    # --- API latency (run here, directly) ---
    if args.api_base:
        api = run_api_suite(args.api_base)
        if api.get("skipped"):
            skips.append({"suite": "api_latency", "reason": api.get("reason", "skipped")})
        else:
            suites["api_latency"] = api
    else:
        skips.append({"suite": "api_latency", "reason": "no --api-base (API server not started)"})

    # --- Asset sizes ---
    fetcher = _make_fetcher(args.api_base, repo)
    all_findings = []
    for scan_id in HERO_SCANS:
        recon = _reconstruction_for(scan_id, args.api_base, repo)
        if recon is None:
            notes.append(f"asset sizes: no reconstruction for {scan_id}")
            continue
        all_findings.extend(asset_sizes.check_scan_assets(scan_id, recon, fetcher))
    # Informational preview: X5's mobile GLBs exist on disk under fixtures/scans/<id>/mobile/ even
    # before the reconstruction contract references them. Count their triangles directly so the
    # report carries real numbers; these do NOT feed the budget (the contract link is the gate).
    mobile_preview = []
    for scan_id in HERO_SCANS:
        mglb = repo / "fixtures" / "scans" / scan_id / "mobile" / "object.glb"
        if mglb.exists():
            try:
                tris = asset_sizes.count_glb_triangles(mglb.read_bytes())
                mobile_preview.append({"scan_id": scan_id, "path": str(mglb.relative_to(repo)),
                                       "triangles": tris, "size_bytes": mglb.stat().st_size,
                                       "budget_triangles": 50_000, "within_budget": tris <= 50_000})
            except Exception as exc:  # noqa: BLE001
                mobile_preview.append({"scan_id": scan_id, "path": str(mglb), "error": str(exc)})

    if all_findings:
        suites["assets"] = {"findings": [vars(f) for f in all_findings],
                            "fetcher_available": fetcher is not None,
                            "mobile_glb_preview": mobile_preview}
        measured.update(asset_sizes.findings_to_measured(all_findings))

        # Precise per-budget SKIP cause when a count could not be produced.
        def _asset_skip_reason(kind: str) -> str:
            fs = [f for f in all_findings if f.kind == kind]
            absent = [f for f in fs if not f.present]
            present_no_count = [f for f in fs if f.present and f.count is None]
            if present_no_count:
                return "; ".join(f"{f.scan_id}: {f.detail}" for f in present_no_count)
            if absent and len(absent) == len(fs):
                field = "splat_mobile_uri" if kind == "splat" else "mesh_mobile_uri"
                return (f"{field} not set on any hero reconstruction "
                        f"({', '.join(f.scan_id for f in absent)})")
            if absent:
                field = "splat_mobile_uri" if kind == "splat" else "mesh_mobile_uri"
                return f"{field} absent for {', '.join(f.scan_id for f in absent)}"
            return "no count produced"

        if asset_sizes.findings_to_measured(all_findings)["assets.splat_gaussians"] is None:
            skip_details["assets.splat_gaussians"] = _asset_skip_reason("splat")
        if asset_sizes.findings_to_measured(all_findings)["assets.mesh_triangles"] is None:
            skip_details["assets.mesh_triangles"] = _asset_skip_reason("mesh")

        missing = [f for f in all_findings if not f.present]
        if missing:
            fields = sorted({("splat_mobile_uri" if f.kind == "splat" else "mesh_mobile_uri")
                             for f in missing})
            note = (f"Mobile asset fields not set on the reconstruction: {', '.join(fields)}. "
                    "Those budgets SKIP until the reconstruction contract carries them.")
            if mobile_preview:
                note += (" X5's mobile mesh GLBs exist on disk (fixtures/scans/<id>/mobile/object.glb) "
                         "and are previewed below with real triangle counts.")
            notes.append(note)
    else:
        skips.append({"suite": "assets", "reason": "no reconstruction available for any hero scan"})

    # --- Grade and write ---
    results = grade_metrics(measured, strict=args.strict, skip_details=skip_details)
    environment = {
        "repo_rev": _git_rev(repo),
        "python": platform.python_version(),
        "node": _node_version(),
        "api_base": args.api_base,
        "web_base": args.web_base,
        "hero_scans": HERO_SCANS,
        "strict": args.strict,
        "bind_policy": "127.0.0.1 only (AGENTS.md decision 1)",
    }
    rpt = report_mod.build_report(
        results, strict=args.strict, suites=suites, skips=skips, notes=notes, environment=environment
    )
    json_path, md_path = report_mod.write_report(rpt, out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    s = rpt["summary"]
    print(f"budgets: {s['passed']} pass / {s['failed']} fail / {s['skipped']} skip (overall_ok={s['overall_ok']})")
    # Exit non-zero only in strict mode with failures, so a normal run is informational.
    return 1 if (args.strict and not s["overall_ok"]) else 0


if __name__ == "__main__":
    sys.exit(main())
