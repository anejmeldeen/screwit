"""Curated parts catalogue and search (SPEC 6.3 parts search, 7 P4 #6; AGENTS.md object-agnostic).

The search logic is object-agnostic: it matches a query against each item's own `match_terms` and
category, and returns nothing when no item genuinely matches. All object-specific knowledge -- which
household repair parts exist, their names, prices, store search URLs and any standard-size table --
lives in catalogue/parts.json, not in this code. For an object with no catalogue match the result is
an empty list, so the agent reports "no standard part found" rather than naming a wrong part.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from screwit.contracts import BomLine, BuyLink

# catalogue/parts.json lives at the repo root; this file is backend/screwit/agent/catalogue.py.
_CATALOGUE_PATH = Path(__file__).resolve().parents[3] / "catalogue" / "parts.json"

# Standard diameter tables keyed by an item's `size_table` field. Adding a new table is a data change
# here, not a logic change; an item with no `size_table` is never size-matched. The one table shipped
# is US tubular drain sizes, used only by the plumbing catalogue entries that opt into it.
_SIZE_TABLES: dict[str, dict[str, float]] = {
    "tubular": {"1-1/4 in": 31.75, "1-1/2 in": 38.1},
}
_SIZE_SNAP_MM = 2.0  # a measured diameter matches a table entry within this many mm (SPEC 6.2).

_MIN_SCORE = 2  # a query must genuinely hit an item's terms/category; below this, no result.


@lru_cache(maxsize=1)
def _catalogue() -> list[dict]:
    return json.loads(_CATALOGUE_PATH.read_text())


def nearest_size(table: str, diameter_mm: float | None) -> str | None:
    """Nearest standard size in `table` within the snap tolerance of the diameter, else None."""
    if diameter_mm is None or table not in _SIZE_TABLES:
        return None
    entries = _SIZE_TABLES[table]
    best = min(entries.items(), key=lambda kv: abs(kv[1] - diameter_mm))
    return best[0] if abs(best[1] - diameter_mm) <= _SIZE_SNAP_MM else None


def size_from_diameter_mm(diameter_mm: float | None) -> str | None:
    """Convenience: nearest tubular size for a diameter (used by the trap demo and tests)."""
    return nearest_size("tubular", diameter_mm)


def search(query: str, measurement_mm: float | None = None,
           category: str | None = None) -> list[BomLine]:
    """Catalogue items matching the query (and optional category), size-filtered when a measurement is
    given. Returns [] when nothing genuinely matches, so the agent never substitutes a wrong part."""
    tokens = [t for t in query.lower().replace("-", " ").split() if t]
    hits: list[tuple[int, dict]] = []
    for item in _catalogue():
        score = _score(item, tokens, category, measurement_mm)
        if score >= _MIN_SCORE:
            hits.append((score, item))
    hits.sort(key=lambda t: t[0], reverse=True)
    return [_to_bom_line(item) for _, item in hits]


def _score(item: dict, tokens: list[str], category: str | None, measurement_mm: float | None) -> int:
    score = 0
    terms = {t.lower() for t in item.get("match_terms", [])}
    name_tokens = set(item["name"].lower().replace("-", " ").split())
    for tok in tokens:
        if tok in terms:
            score += 2
        elif tok in name_tokens or tok == item["category"]:
            score += 1
    if category and item["category"] == category:
        score += 3
    # A size match only reinforces an item the query already hit; it never creates a match on its own.
    if score and measurement_mm is not None and item.get("size_table"):
        size = nearest_size(item["size_table"], measurement_mm)
        if size and size == item.get("nominal_size"):
            score += 4
    return score


def _to_bom_line(item: dict, qty: int = 1) -> BomLine:
    return BomLine(
        name=item["name"],
        spec=item["spec"],
        qty=qty,
        est_price_usd=item["est_price_usd"],
        buy_links=[BuyLink(vendor=link["vendor"], url=link["url"]) for link in item["buy_links"]],
    )
