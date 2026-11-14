"""Honest accuracy evaluation for the hero objects (SPEC 6.2 step 4, P3 workload bullet 9).

For every feature Measurement that has both a scan value and a caliper value, report the per-
dimension error (scan - caliper). Report the snap hit rate and, crucially, the **false-snap rate**:
the fraction of snapped Measurements whose snapped diameter disagrees with the caliper diameter by
more than a tolerance. No accuracy figure is claimed that this script did not measure.

Usage:
    python -m screwit.perception.evaluate <parts.json> [<parts.json> ...]
    python -m screwit.perception.evaluate --fixtures   # all fixtures/scans/*/parts.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter

from screwit.contracts import Part

FALSE_SNAP_TOLERANCE_MM = 1.0  # a snap is "false" if its diameter is this far from the caliper


@dataclass
class DimError:
    part_id: str
    feature_id: str
    field: str
    scan_mm: float
    caliper_mm: float

    @property
    def error_mm(self) -> float:
        return self.scan_mm - self.caliper_mm


@dataclass
class EvalReport:
    errors: list[DimError] = field(default_factory=list)
    snapped: int = 0
    false_snaps: int = 0

    @property
    def n(self) -> int:
        return len(self.errors)

    @property
    def mae_mm(self) -> float | None:
        return sum(abs(e.error_mm) for e in self.errors) / self.n if self.n else None

    @property
    def rmse_mm(self) -> float | None:
        return math.sqrt(sum(e.error_mm**2 for e in self.errors) / self.n) if self.n else None

    @property
    def max_abs_mm(self) -> float | None:
        return max((abs(e.error_mm) for e in self.errors), default=None)

    @property
    def false_snap_rate(self) -> float | None:
        return self.false_snaps / self.snapped if self.snapped else None

    def as_dict(self) -> dict:
        return {
            "dimensions_compared": self.n,
            "mae_mm": self.mae_mm,
            "rmse_mm": self.rmse_mm,
            "max_abs_error_mm": self.max_abs_mm,
            "snapped_measurements": self.snapped,
            "false_snaps": self.false_snaps,
            "false_snap_rate": self.false_snap_rate,
            "per_dimension": [
                {"part_id": e.part_id, "feature_id": e.feature_id, "field": e.field,
                 "scan_mm": e.scan_mm, "caliper_mm": e.caliper_mm,
                 "error_mm": round(e.error_mm, 3)} for e in self.errors],
        }


def _measurements(part: Part):
    from screwit.contracts.models import FEATURE_MEASUREMENT_FIELDS

    for f in part.features:
        for fld in FEATURE_MEASUREMENT_FIELDS[f.type]:
            yield f.feature_id, fld, getattr(f, fld)


def evaluate_parts(parts: list[Part]) -> EvalReport:
    report = EvalReport()
    for part in parts:
        for fid, fld, m in _measurements(part):
            if m.snapped_mm is not None:
                report.snapped += 1
                if m.caliper_mm is not None and \
                        abs(2 * m.snapped_mm - 2 * m.caliper_mm) > FALSE_SNAP_TOLERANCE_MM:
                    report.false_snaps += 1
            if m.scan_mm is not None and m.caliper_mm is not None:
                report.errors.append(DimError(part.part_id, fid, fld, m.scan_mm, m.caliper_mm))
    return report


def evaluate_file(path: Path) -> EvalReport:
    parts = TypeAdapter(list[Part]).validate_json(path.read_text())
    return evaluate_parts(parts)


def _fixture_parts_files() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    return sorted((root / "fixtures" / "scans").glob("*/parts.json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan-vs-caliper accuracy and false-snap rate.")
    ap.add_argument("parts", nargs="*", type=Path, help="parts.json files")
    ap.add_argument("--fixtures", action="store_true", help="evaluate every fixture parts.json")
    args = ap.parse_args(argv)

    files = list(args.parts)
    if args.fixtures:
        files += _fixture_parts_files()
    if not files:
        ap.error("pass one or more parts.json paths, or --fixtures")

    out = {}
    for f in files:
        out[str(f)] = evaluate_file(f).as_dict()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
