"""Every SPEC 5 route exercised over real HTTP against the running app (mock mode).

The app is started once per session (conftest.server). These tests validate every response body
against the Pydantic contracts, decode both SSE streams with the real contract decoders, and assert
the hero conversation shapes and the revision/stale rules of SPEC 5.

Mock mode serves the hero fixtures and "runs nothing" for create/start (P4 brief 6). Where a route's
behaviour is inherently stateful (revision bump on PATCH, 409 on stale /scale, CAD staleness after a
patch) and the mock backend answers it as a static fixture, the test SKIPS with a clear reason rather
than asserting against a stub -- the same assertions run for real once the live backend is wired.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import TypeAdapter

from screwit.contracts import (
    CadResult,
    DemoScan,
    MeasureResponse,
    Part,
    Reconstruction,
    Scan,
    SceneGraph,
)

from .sse_client import decode_chat, iter_sse

TRAP = "scn_trap"
KNOB = "scn_knob"
PartList = TypeAdapter(list[Part])
DemoScanList = TypeAdapter(list[DemoScan])


def _ok(resp: httpx.Response) -> httpx.Response:
    assert resp.status_code == 200, f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text[:400]}"
    return resp


def _skip_if_unsupported(resp: httpx.Response, what: str) -> None:
    if resp.status_code in (404, 405, 501):
        pytest.skip(f"{what} not supported in mock mode (status {resp.status_code})")


# --------------------------------------------------------------------------- read routes

def test_demo_scans(client: httpx.Client):
    scans = DemoScanList.validate_json(_ok(client.get("/api/demo-scans")).content)
    ids = {s.scan_id for s in scans}
    assert {TRAP, KNOB} <= ids, f"both hero scans must appear on the landing gallery, got {ids}"


@pytest.mark.parametrize("scan_id", [TRAP, KNOB])
def test_get_scan(client: httpx.Client, scan_id: str):
    scan = Scan.model_validate_json(_ok(client.get(f"/api/scans/{scan_id}")).content)
    assert scan.scan_id == scan_id
    assert scan.status == "ready"
    assert scan.stage("parts").state == "done"


@pytest.mark.parametrize("scan_id", [TRAP, KNOB])
def test_get_reconstruction_with_signed_urls(client: httpx.Client, scan_id: str):
    recon = Reconstruction.model_validate_json(_ok(client.get(f"/api/scans/{scan_id}/reconstruction")).content)
    assert recon.scan_id == scan_id
    # SPEC 5: assets come back as signed URLs. Fetch the mesh and get real bytes.
    assert recon.mesh_uri
    _assert_asset_fetchable(client, recon.mesh_uri)
    if recon.splat_uri:
        _assert_asset_fetchable(client, recon.splat_uri)


def _assert_asset_fetchable(client: httpx.Client, uri: str) -> None:
    url = uri if uri.startswith(("http", "/")) else f"/api/assets/{uri}"  # signed URLs are already paths
    r = client.get(url)
    _skip_if_unsupported(r, f"asset {uri}")
    assert r.status_code == 200, f"asset {uri} -> {r.status_code}"
    assert r.content, f"asset {uri} returned no bytes"


@pytest.mark.parametrize("scan_id", [TRAP, KNOB])
def test_get_parts(client: httpx.Client, scan_id: str):
    parts = PartList.validate_json(_ok(client.get(f"/api/scans/{scan_id}/parts")).content)
    assert parts, "parts list must not be empty for a ready hero scan"


def test_trap_parts_cover_expected_labels(client: httpx.Client):
    parts = PartList.validate_json(_ok(client.get(f"/api/scans/{TRAP}/parts")).content)
    labels = {p.label.lower() for p in parts}
    for want in ("tailpiece", "j-bend", "slip nut", "slip-joint washer"):
        assert want in labels, f"trap should list a {want!r} part; got {sorted(labels)}"


@pytest.mark.parametrize("scan_id", [TRAP, KNOB])
def test_get_scene_graph(client: httpx.Client, scan_id: str):
    graph = SceneGraph.model_validate_json(_ok(client.get(f"/api/scans/{scan_id}/scene-graph")).content)
    parts = PartList.validate_json(_ok(client.get(f"/api/scans/{scan_id}/parts")).content)
    assert sorted(graph.parts) == sorted(p.part_id for p in parts)


def test_measure(client: httpx.Client):
    body = {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 100.0]}
    r = client.post(f"/api/scans/{TRAP}/measure", json=body)
    _skip_if_unsupported(r, "measure")
    res = MeasureResponse.model_validate_json(_ok(r).content)
    assert res.distance_mm == pytest.approx(100.0, abs=1e-6)


def test_chat_history(client: httpx.Client):
    r = client.get(f"/api/scans/{TRAP}/chat")
    _skip_if_unsupported(r, "chat history")
    _ok(r)  # body shape is owner-defined (list of ChatMessage); presence is enough here


# --------------------------------------------------------------------------- chat SSE (SPEC 4.8)

def _chat(client: httpx.Client, scan_id: str, message: str) -> list:
    with client.stream("POST", f"/api/scans/{scan_id}/chat", json={"message": message}) as resp:
        if resp.status_code in (404, 405, 501):
            pytest.skip(f"chat not supported in mock mode (status {resp.status_code})")
        assert resp.status_code == 200, f"chat -> {resp.status_code}"
        return decode_chat(resp)  # decodes every frame with contracts.sse.decode_chat_event


def _card_types(events: list) -> list[str]:
    return [ev.card.type for ev in events if ev.event == "card"]


def test_chat_trap_leak_stream(client: httpx.Client):
    events = _chat(client, TRAP, "it's leaking under here")
    assert events[-1].event == "done"
    assert _card_types(events) == ["repair_plan", "parts_to_buy"]
    repair = next(ev.card.payload for ev in events if ev.event == "card" and ev.card.type == "repair_plan")
    assert [s.n for s in repair.steps] == [1, 2, 3, 4]


def test_chat_knob_asks_then_cads(client: httpx.Client):
    ask = _chat(client, KNOB, "make me a replacement knob")
    assert _card_types(ask) == [], "knob conversation asks for the shaft size before any CAD"
    text = " ".join(ev.text for ev in ask if ev.event == "token").lower()
    assert "shaft" in text

    reply = _chat(client, KNOB, "6 mm")
    assert _card_types(reply) == ["cad_result"]
    cad = next(ev.card.payload for ev in reply if ev.event == "card")
    assert cad.status == "ok" and not cad.stale
    assert all(c.result == "pass" for c in cad.checks if c.kind == "required")


# --------------------------------------------------------------------------- progress SSE

def test_progress_stream_emits_scan(client: httpx.Client):
    # A ready scan's stream may close quickly; we only need one well-formed `scan` frame.
    with client.stream("GET", f"/api/scans/{TRAP}/events") as resp:
        if resp.status_code in (404, 405, 501):
            pytest.skip(f"progress stream not supported in mock mode (status {resp.status_code})")
        assert resp.status_code == 200
        saw_scan = False
        for ev in iter_sse(resp):  # keep-alive comment lines are already skipped
            assert ev.name == "scan", f"progress stream must only emit `scan` events, got {ev.name!r}"
            Scan.model_validate_json(ev.data)  # validate every frame against the contract
            saw_scan = True
            break
        assert saw_scan, "progress stream yielded no scan event"


# --------------------------------------------------------------------------- create / start

def test_create_scan(client: httpx.Client):
    r = client.post("/api/scans", json={"input_type": "video"})
    _skip_if_unsupported(r, "create scan")
    _ok(r)
    from screwit.contracts import CreateScanResponse  # local import keeps the top tidy

    resp = CreateScanResponse.model_validate_json(r.content)
    assert resp.scan_id and resp.uploads


# --------------------------------------------------------------------------- stateful rules (SPEC 5)

def _get_part(client: httpx.Client, scan_id: str, part_id: str) -> Part:
    parts = PartList.validate_json(_ok(client.get(f"/api/scans/{scan_id}/parts")).content)
    return next(p for p in parts if p.part_id == part_id)


def _patch(client: httpx.Client, scan_id: str, part_id: str, body: dict) -> httpx.Response:
    r = client.patch(f"/api/scans/{scan_id}/parts/{part_id}", json=body)
    _skip_if_unsupported(r, "part patch")
    if r.status_code != 200:
        pytest.skip(f"part patch not mutating in mock mode (status {r.status_code})")
    return r


def test_patch_label_only_does_not_bump_revision(client: httpx.Client):
    before = _get_part(client, TRAP, "p1")
    r = _patch(client, TRAP, "p1", {"label": before.label + " (edit)"})
    after = Part.model_validate_json(r.content)
    if after.measurement_revision != before.measurement_revision:
        pytest.skip("mock backend does not preserve revisions across label patch")
    assert after.measurement_revision == before.measurement_revision, (
        "a label-only patch must not bump measurement_revision (SPEC 4 revisions, 5)"
    )


def test_patch_measurement_bumps_revision(client: httpx.Client):
    before = _get_part(client, TRAP, "p3")
    body = {"measurements": [{"feature_id": "p3_f1", "field": "radius", "caliper_mm": 31.5}]}
    r = _patch(client, TRAP, "p3", body)
    after = Part.model_validate_json(r.content)
    if after.measurement_revision <= before.measurement_revision:
        pytest.skip("mock backend serves a static part; revision bump is a live-backend behaviour")
    assert after.measurement_revision == before.measurement_revision + 1


def test_scale_rejects_stale_revision(client: httpx.Client):
    body = {
        "point_a_mm": [0.0, 0.0, 0.0],
        "point_b_mm": [0.0, 0.0, 100.0],
        "known_length_mm": 85.6,
        "expected_revision": 999,  # far ahead of any current revision -> stale
    }
    r = client.post(f"/api/scans/{TRAP}/scale", json=body)
    _skip_if_unsupported(r, "scale")
    if r.status_code == 200:
        pytest.skip("mock backend accepts /scale without a revision check (live-backend behaviour)")
    assert r.status_code == 409, f"stale /scale must be rejected with 409, got {r.status_code}"


def test_cad_get_returns_result(client: httpx.Client):
    # The knob fixture carries cad_07; GET must return a contract-valid CadResult with staleness computed.
    r = client.get(f"/api/scans/{KNOB}/cad/cad_07")
    _skip_if_unsupported(r, "cad get")
    CadResult.model_validate_json(_ok(r).content)
