"""Curated catalogue search (SPEC 6.3, 7 P4 #6; AGENTS.md object-agnostic).

The catalogue spans household repair parts, not only plumbing, with real store SEARCH URLs. Search is
object-agnostic: a query for a part outside the catalogue returns nothing rather than a wrong item.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from screwit.agent import catalogue

CATALOGUE_PATH = Path(__file__).resolve().parents[3] / "catalogue" / "parts.json"


def test_catalogue_spans_household_categories():
    items = json.loads(CATALOGUE_PATH.read_text())
    names = " ".join(i["name"] for i in items).lower()
    assert "1-1/2 in slip-joint washer" in names
    assert "reducing slip washer" in names  # 1-1/2 x 1-1/4 reducing (SPEC 7 P4 #6)
    categories = {i["category"] for i in items}
    # Plumbing stays, plus non-plumbing household repair items (AGENTS.md: object can be anything).
    assert {"washer", "slip_nut", "p_trap", "faucet_cartridge", "screw"} <= categories
    assert {"hinge", "drawer_slide", "furniture_foot", "knob", "wall_anchor"} <= categories


def test_all_links_are_search_urls_not_product_pages():
    """Real store SEARCH URLs only, no invented product pages (no fabricated /p/<id> paths)."""
    items = json.loads(CATALOGUE_PATH.read_text())
    search_shape = re.compile(r"/s/|/search|/s\?k=")
    for item in items:
        for link in item["buy_links"]:
            url = link["url"]
            assert url.startswith("https://")
            assert search_shape.search(url), url
            assert "/p/" not in url and "/pd/" not in url  # not a specific product page


def test_search_finds_non_plumbing_items():
    assert any("hinge" in h.name.lower() for h in catalogue.search("cabinet hinge"))
    assert any("drawer" in h.name.lower() for h in catalogue.search("drawer slide"))
    assert any("foot" in h.name.lower() for h in catalogue.search("furniture leg foot"))
    assert any("anchor" in h.name.lower() for h in catalogue.search("drywall wall anchor"))
    assert any("hose" in h.name.lower() for h in catalogue.search("garden hose washer"))


def test_search_returns_nothing_for_object_outside_catalogue():
    """An object with no catalogue match yields [] -- never a wrong substitute part (AGENTS.md)."""
    for query in ["bicycle derailleur", "camera shutter", "violin bridge", "carburetor float"]:
        assert catalogue.search(query) == [], query


def test_search_sizes_from_diameter():
    hits = catalogue.search("slip washer", measurement_mm=38.1)
    assert hits
    assert any("1-1/2" in h.name for h in hits)


def test_nearest_tubular_size():
    assert catalogue.size_from_diameter_mm(31.75) == "1-1/4 in"
    assert catalogue.size_from_diameter_mm(38.1) == "1-1/2 in"
    assert catalogue.size_from_diameter_mm(50.0) is None  # too far from any tubular size


def test_size_match_never_creates_a_match_alone():
    """A measurement only reinforces a query hit; it cannot pull in an item the query did not match."""
    # A diameter that snaps to a tubular size, but a query that matches a knob, must not return washers.
    hits = catalogue.search("knob", measurement_mm=38.1)
    assert all(h.name.lower().find("washer") == -1 for h in hits)
    assert any("knob" in h.name.lower() for h in hits)


def test_search_returns_bomlines_with_links():
    hits = catalogue.search("p-trap kit", category="p_trap")
    assert hits and hits[0].buy_links
    assert hits[0].est_price_usd > 0
