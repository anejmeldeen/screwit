"""Standard-size snapping (SPEC 6.2 step 4).

Snapping compares the scan **diameter** (2 * radius.scan_mm) against a table chosen by the
feature's role, so entries within one table are far enough apart to be told apart at this
accuracy:

- exterior ``cylinder`` features use the **tube table** (tubular drain outer diameters);
- ``bore`` features use the **bore table** (the nut/washer bores that accept those tubes).

A snap fires only when the scan diameter is within ``WINDOW_MM`` (2.0 mm) of one entry *and*
the next-nearest entry in the same table is at least ``MIN_SEPARATION_MM`` (6 mm) away. Metric
bolt sizes (M3-M10) and common shafts (6.0 vs 6.35 mm) are never snapped; they are only
suggestions in the agent's wording. The snapped **radius** stored is half the standard diameter.

A snapped value is treated like a scan value for everything physical (confirm-before-printing,
advisory mate checks); it only drives the buy recommendation. A person clears a false snap by
entering ``caliper_mm``.
"""

from __future__ import annotations

from dataclasses import dataclass

WINDOW_MM = 2.0
MIN_SEPARATION_MM = 6.0

# Standard tubular drain outer diameters, in millimetres (SPEC 6.2 step 4):
#   1-1/4 in = 31.75 mm, 1-1/2 in = 38.1 mm. These are 6.35 mm apart, just over the
#   separation floor, which is why only these two sizes are snapped.
TUBE_TABLE: dict[str, float] = {
    "tubular_1_1_4_in": 31.75,
    "tubular_1_1_2_in": 38.1,
}

# Bores that accept those tubes carry the same nominal diameters (SPEC 6.2 step 4).
BORE_TABLE: dict[str, float] = {
    "bore_for_tubular_1_1_4_in": 31.75,
    "bore_for_tubular_1_1_2_in": 38.1,
}

# Feature role -> its snap table. Only exterior cylinders and bores snap; every other
# feature type (d_shaft, plane, hole, thread_suspected) is never snapped.
ROLE_TABLE: dict[str, dict[str, float]] = {
    "cylinder": TUBE_TABLE,
    "bore": BORE_TABLE,
}


@dataclass(frozen=True)
class SnapResult:
    snapped_diameter_mm: float
    snapped_radius_mm: float
    snap_rule: str


def _snap_in_table(diameter_mm: float, table: dict[str, float]) -> SnapResult | None:
    """Return the snap for ``diameter_mm`` in ``table``, or None when the rule forbids it.

    A snap fires only when ``diameter_mm`` is within ``WINDOW_MM`` of the nearest table entry
    *and* that entry is separated from its own nearest neighbour in the table by at least
    ``MIN_SEPARATION_MM`` (so two standard sizes cannot be confused at this accuracy). With the
    tube/bore tables (31.75 and 38.1 mm, 6.35 mm apart) both entries clear the floor, so e.g. a
    37.0 mm diameter snaps to 38.1 even though 31.75 is only ~5.25 mm from the measured value.
    """
    values = sorted(table.values())
    ranked = sorted(table.items(), key=lambda kv: abs(kv[1] - diameter_mm))
    rule, nominal = ranked[0]
    if abs(nominal - diameter_mm) > WINDOW_MM:
        return None  # not close enough to any standard size
    # Separation is a property of the matched entry versus its neighbours in the table.
    neighbours = [v for v in values if v != nominal]
    if neighbours:
        nearest_gap = min(abs(v - nominal) for v in neighbours)
        if nearest_gap < MIN_SEPARATION_MM:
            return None  # this standard size is too close to another to tell apart
    return SnapResult(snapped_diameter_mm=nominal, snapped_radius_mm=nominal / 2.0, snap_rule=rule)


def snap_diameter(feature_type: str, scan_radius_mm: float | None) -> SnapResult | None:
    """Snap a feature's scan radius to a standard size, or None if no snap fires.

    ``feature_type`` picks the table by role (SPEC 6.2 step 4). Only ``cylinder`` (tube table)
    and ``bore`` (bore table) ever snap; anything else returns None. The comparison is on
    diameter = 2 * ``scan_radius_mm``.
    """
    if scan_radius_mm is None:
        return None
    table = ROLE_TABLE.get(feature_type)
    if table is None:
        return None
    return _snap_in_table(2.0 * scan_radius_mm, table)
