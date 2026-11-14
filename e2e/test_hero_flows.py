"""Browser end-to-end tests of the two hero flows against the built site + mock API.

Selectors use text and ARIA roles, not CSS classes, so they survive styling changes. A few places
want a stable hook (the viewer canvas, the status bar, a chat card); those data-testid names are
listed in the X3 report as a request to P1 rather than guessed here, and the tests fall back to
role/text where they can.

Every test skips cleanly (via the `stack` fixture) until P1's web app and P4's API are present.
A screenshot is written per step into e2e/artifacts/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from _stack import ARTIFACTS, Stack  # noqa: E402
from playwright.sync_api import Page, expect  # noqa: E402

TRAP_ID = "scn_trap"
KNOB_ID = "scn_knob"


def _shot(page: Page, name: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(ARTIFACTS / f"{name}.png"), full_page=True)


def _open_workspace(page: Page, stack: Stack, scan_id: str, step: str) -> None:
    page.goto(f"{stack.web_url}/scan/{scan_id}", wait_until="networkidle")
    _shot(page, f"{step}_workspace")


def test_landing_shows_both_demo_scans(stack: Stack, browser_ready, page: Page):
    page.goto(stack.web_url, wait_until="networkidle")
    _shot(page, "landing")
    # Titles come from fixtures/demo_scans.json; match on their visible text, not markup.
    expect(page.get_by_text("P-trap", exact=False)).to_be_visible()
    expect(page.get_by_text("knob", exact=False)).to_be_visible()


def test_trap_workspace_stages_and_parts(stack: Stack, browser_ready, page: Page):
    _open_workspace(page, stack, TRAP_ID, "trap")
    # Status bar: every stage reads done. The seven stage names are the SPEC 4.1 vocabulary.
    body = page.locator("body")
    for stage in ("Frames", "Geometry", "Scale", "Object mask", "Splat", "Mesh", "Parts"):
        expect(body).to_contain_text(stage, ignore_case=True)
    # Parts panel lists the four trap parts by label.
    for label in ("tailpiece", "J-bend", "slip nut", "slip-joint washer"):
        expect(page.get_by_text(label, exact=False).first).to_be_visible()
    _shot(page, "trap_parts")


def test_trap_leak_renders_repair_plan_and_parts_card(stack: Stack, browser_ready, page: Page):
    _open_workspace(page, stack, TRAP_ID, "trap_chat")
    box = page.get_by_role("textbox")
    box.first.fill("it's leaking under here")
    box.first.press("Enter")
    # Repair plan + parts-to-buy card from the offline agent (the default when no Claude key is set),
    # grounded in the trap's scene graph: the washer is the top cause and the BOM line.
    expect(page.get_by_text("Repair plan", exact=False).first).to_be_visible(timeout=15000)
    expect(page.get_by_text("bucket", exact=False).first).to_be_visible()
    expect(page.get_by_text("slip-joint washer", exact=False).first).to_be_visible()
    expect(page.get_by_text("Parts to buy", exact=False).first).to_be_visible()
    # The last step is the leak test.
    expect(page.get_by_text("test for correct operation and leaks", exact=False).first).to_be_visible()
    _shot(page, "trap_repair_plan")


def test_knob_confirm_then_cad_result(stack: Stack, browser_ready, page: Page):
    _open_workspace(page, stack, KNOB_ID, "knob_chat")
    box = page.get_by_role("textbox").first
    box.fill("make me a replacement knob")
    box.press("Enter")
    # Agent asks for the shaft size first (confirm-before-printing).
    # Wait for the agent's actual question (the word "shaft" alone also matches the part label "D-shaft"),
    # and for the reply to finish streaming before answering.
    expect(page.get_by_text("calipers", exact=False).first).to_be_visible(timeout=15000)
    _shot(page, "knob_ask_shaft")

    box.fill("6 mm")
    # Send is enabled only when the box has text AND the previous reply has finished streaming.
    expect(page.get_by_test_id("chat-send")).to_be_enabled(timeout=15000)
    page.get_by_test_id("chat-send").click()
    # CAD result card with required checks passing, downloadable files, and the D-bore mate checks.
    expect(page.get_by_text("watertight", exact=False).first).to_be_visible(timeout=15000)
    expect(page.get_by_text("mate_p2_f1_flat_orientation_deg", exact=False).first).to_be_visible()
    expect(page.get_by_role("link", name="Download STL")).to_be_visible()
    _shot(page, "knob_cad_result")


def test_artifacts_dir_is_writable():
    # Independent of the stack: the runner relies on this path existing and being writable.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    probe = ARTIFACTS / ".probe"
    probe.write_text("ok")
    assert probe.read_text() == "ok"
    probe.unlink()
    assert Path(ARTIFACTS).is_dir()
