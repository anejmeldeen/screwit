#!/usr/bin/env python3
"""Compare scan measurements with caliper readings for a smoke.py run.

  compare.py report.json --caliper caliper.json [--scan-mm scan.json]

caliper.json: {"name": mm, ...}. scan.json (same shape) holds the scan-side values from the repo's
measurement code or by hand. If screwit.recon.scale / screwit.perception are importable their public
callables are listed in the output; this tool does not call them because no entry-point name is
fixed by the contracts yet, so the scan-vs-caliper error is computed from --scan-mm.
Writes compare.json and compare.md next to report.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_MODULES = ("screwit.recon.scale", "screwit.perception")


def load_readings(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of name -> mm")
    out: dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{path}: {k!r} must be a positive number of millimetres, got {v!r}")
        out[str(k)] = float(v)
    return out


def errors(scan: dict[str, float], caliper: dict[str, float]) -> dict[str, Any]:
    rows = []
    for name in sorted(caliper):
        if name not in scan:
            rows.append({"name": name, "caliper_mm": caliper[name], "scan_mm": None, "error_mm": None, "error_pct": None})  # noqa: E501
            continue
        err = scan[name] - caliper[name]
        rows.append({"name": name, "caliper_mm": caliper[name], "scan_mm": scan[name],
                     "error_mm": err, "error_pct": 100.0 * err / caliper[name]})
    paired = [r for r in rows if r["error_mm"] is not None]
    summary = {
        "paired": len(paired),
        "unpaired_caliper": len(rows) - len(paired),
        "extra_scan_names": sorted(set(scan) - set(caliper)),
    }
    if paired:
        summary["mean_abs_error_mm"] = sum(abs(r["error_mm"]) for r in paired) / len(paired)
        summary["max_abs_error_mm"] = max(abs(r["error_mm"]) for r in paired)
        summary["max_abs_error_pct"] = max(abs(r["error_pct"]) for r in paired)
    return {"rows": rows, "summary": summary}


def probe_repo_modules() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mod in REPO_MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            result[mod] = {"importable": False, "reason": f"{type(e).__name__}: {e}"}
            continue
        public = sorted(n for n in dir(m) if not n.startswith("_") and callable(getattr(m, n)))
        result[mod] = {"importable": True, "public_callables": public}
    return result


def pick_scale_info(report: dict[str, Any]) -> dict[str, Any]:
    """Model-reported scale numbers from the smoke report, for context next to the caliper error."""
    out: dict[str, Any] = {}
    for stage in ("da3", "mapanything"):
        for r in report.get("stages", {}).get(stage, {}).get("runs", []):
            if r.get("status") == "ok":
                key = f"{stage}_{r['views']}"
                out[key] = r.get("scale_factor") if stage == "da3" else r.get("metric_scaling_factor")
    return out


def render_md(result: dict[str, Any]) -> str:
    lines = ["# Scan vs caliper", ""]
    s = result.get("errors")
    if s is None:
        lines.append("No scan readings supplied (`--scan-mm`): error table skipped.")
    else:
        lines += ["| name | caliper mm | scan mm | error mm | error % |", "|---|---|---|---|---|"]
        for r in s["rows"]:
            f = lambda v: "" if v is None else f"{v:.2f}"  # noqa: E731
            lines.append(f"| {r['name']} | {f(r['caliper_mm'])} | {f(r['scan_mm'])} | {f(r['error_mm'])} | {f(r['error_pct'])} |")  # noqa: E501
        lines += ["", "```json", json.dumps(s["summary"], indent=2), "```"]
    lines += ["", "## Repo modules", ""]
    for mod, info in result["repo_modules"].items():
        if info["importable"]:
            lines.append(f"- `{mod}`: importable, public callables: {', '.join(info['public_callables']) or 'none'}")
        else:
            lines.append(f"- `{mod}`: not importable ({info['reason']}); section skipped")
    lines += ["", "## Model-reported scale (from report.json)", "", "```json",
              json.dumps(result["model_scale"], indent=2), "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path)
    p.add_argument("--caliper", type=Path, required=True)
    p.add_argument("--scan-mm", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        report = json.loads(args.report.read_text())
        caliper = load_readings(args.caliper)
        scan = load_readings(args.scan_mm) if args.scan_mm else None
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    result: dict[str, Any] = {
        "errors": errors(scan, caliper) if scan is not None else None,
        "repo_modules": probe_repo_modules(),
        "model_scale": pick_scale_info(report),
    }
    out = args.report.parent
    (out / "compare.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "compare.md").write_text(render_md(result))
    print(f"wrote {out / 'compare.json'} and {out / 'compare.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
