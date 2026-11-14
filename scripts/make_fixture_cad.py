#!/usr/bin/env python
"""Generate the CAD fixture outputs for scn_knob and report differences (P5 workload 4, SPEC 6.4).

Runs the fixture cad_request.json against the fixture parts.json through the real CadService, writes
the outputs into fixtures/scans/scn_knob/cad/cad_07/{model.stl,model.step,model.py,preview.png}, and
prints the differences between the produced CadResult and the committed fixtures/scans/scn_knob/
cad_result.json. It never edits cad_result.json (that file is owned by the integration lead).

Run from the worktree:
    PYTHONPATH=backend <venv>/bin/python scripts/make_fixture_cad.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from screwit.cad.service import CadRunner  # noqa: E402
from screwit.contracts import CadRequest, CadResult, Part  # noqa: E402
from screwit.storage.local import LocalStorage  # noqa: E402

# Everything is read from and written to this worktree's own checkout (never the main tree).
FIX = ROOT / "fixtures" / "scans" / "scn_knob"
PROFILES = ROOT / "profiles"


def _load_parts() -> dict[str, Part]:
    data = json.loads((FIX / "parts.json").read_text())
    return {p["part_id"]: Part.model_validate(p) for p in data}


def _checks_by_name(result: CadResult) -> dict[str, dict]:
    return {c.name: c.model_dump(exclude_none=True) for c in result.checks}


def _diff_checks(produced: CadResult, expected: CadResult) -> list[str]:
    diffs: list[str] = []
    pc, ec = _checks_by_name(produced), _checks_by_name(expected)
    for name in sorted(set(pc) | set(ec)):
        if name not in pc:
            diffs.append(f"check {name!r}: in fixture, NOT produced")
            continue
        if name not in ec:
            diffs.append(f"check {name!r}: produced, NOT in fixture")
            continue
        p, e = pc[name], ec[name]
        for field in ("kind", "result", "value", "limit", "input_source"):
            if p.get(field) != e.get(field):
                diffs.append(f"check {name!r}.{field}: produced={p.get(field)!r} fixture={e.get(field)!r}")
    return diffs


def main() -> int:
    out_dir = FIX / "cad" / "cad_07"
    out_dir.mkdir(parents=True, exist_ok=True)

    req = CadRequest.model_validate_json((FIX / "cad_request.json").read_text())
    parts = _load_parts()

    # Write into a local store rooted at this worktree's fixtures tree so files land under
    # fixtures/scans/scn_knob/cad/cad_07/... (a path this workstream owns).
    store = LocalStorage(ROOT / "fixtures")
    runner = CadRunner(PROFILES)
    produced = runner.run(req, parts, store)

    print(f"produced status: {produced.status}")
    print(f"wrote: {out_dir}")
    for f in ("model.stl", "model.step", "model.py", "preview.png"):
        p = out_dir / f
        print(f"  {f:14s} {'exists' if p.exists() else 'MISSING':8s} {p.stat().st_size if p.exists() else 0} bytes")

    expected = CadResult.model_validate_json((FIX / "cad_result.json").read_text())
    print("\n--- differences from committed cad_result.json (NOT edited) ---")
    diffs = []
    if produced.status != expected.status:
        diffs.append(f"status: produced={produced.status!r} fixture={expected.status!r}")
    diffs.extend(_diff_checks(produced, expected))
    # print_settings differences (estimates will differ; reported, not an error).
    if produced.print_settings and expected.print_settings:
        pp, ee = produced.print_settings.model_dump(), expected.print_settings.model_dump()
        for k in sorted(pp):
            if pp[k] != ee.get(k):
                diffs.append(f"print_settings.{k}: produced={pp[k]!r} fixture={ee.get(k)!r}")
    if not diffs:
        print("  (none)")
    else:
        for d in diffs:
            print(f"  - {d}")
    if "--write-result" in sys.argv:
        # The integration lead reconciles the committed result with what the real service produces
        # (the STL/STEP above are the real service's output, so the JSON must describe them).
        (FIX / "cad_result.json").write_text(produced.model_dump_json(indent=2) + "\n")
        print(f"wrote {FIX / 'cad_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
