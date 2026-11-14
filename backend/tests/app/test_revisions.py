"""Revision, 409 and staleness rules (SPEC 5): scale, PATCH, merge, CAD."""

from __future__ import annotations

from screwit.contracts import CadResult, Part

KNOB_CAD_BODY = {
    "cad_id": "cad_test_01",
    "scan_id": "scn_knob",
    "reconstruction_revision": 1,
    "part_measurement_revisions": {"p2": 3},
    "mode": "template",
    "template": "knob",
    "params": {"shaft_diameter": 6.0},
    "param_sources": {"shaft_diameter": "caliper"},
    "codegen_prompt": None,
    "mates_with": [{"part_id": "p2", "feature_id": "p2_f1", "fit": "snug", "radial_clearance_mm": 0.12}],
    "placement": {"translate_mm": [0, 0, 0], "rotation_quat": [0, 0, 0, 1]},
    "material": "PETG",
    "printer_profile": "prusa_mk4_petg_0.2",
}


def test_cad_post_happy_path(client):
    r = client.post("/api/scans/scn_knob/cad", json=KNOB_CAD_BODY)
    assert r.status_code == 200
    result = CadResult.model_validate(r.json())
    assert result.status == "ok"
    assert result.stale is False
    assert result.cad_id == "cad_test_01"


def test_cad_post_409_on_revision_mismatch(client):
    body = dict(KNOB_CAD_BODY, part_measurement_revisions={"p2": 99})
    r = client.post("/api/scans/scn_knob/cad", json=body)
    assert r.status_code == 409


def test_cad_get_and_stale_after_patch(client):
    client.post("/api/scans/scn_knob/cad", json=KNOB_CAD_BODY)
    r = client.get("/api/scans/scn_knob/cad/cad_test_01")
    assert r.status_code == 200
    assert CadResult.model_validate(r.json()).stale is False

    # Edit p2 -> measurement_revision bumps -> the CAD result that pinned p2 becomes stale.
    patch = {"measurements": [{"feature_id": "p2_f1", "field": "length", "caliper_mm": 20.0}]}
    pr = client.patch("/api/scans/scn_knob/parts/p2", json=patch)
    assert pr.status_code == 200
    assert Part.model_validate(pr.json()).measurement_revision == 4

    r2 = client.get("/api/scans/scn_knob/cad/cad_test_01")
    assert CadResult.model_validate(r2.json()).stale is True


def test_patch_rename_does_not_bump_revision(client):
    before = client.get("/api/scans/scn_knob/parts").json()
    rev = next(p["measurement_revision"] for p in before if p["part_id"] == "p2")
    r = client.patch("/api/scans/scn_knob/parts/p2", json={"label": "control knob shaft"})
    assert r.status_code == 200
    part = Part.model_validate(r.json())
    assert part.label == "control knob shaft"
    assert part.measurement_revision == rev  # no bump on a pure rename


def test_patch_measurement_bumps_and_sets_user_edited(client):
    patch = {"measurements": [{"feature_id": "p1_f1", "field": "extent_u", "caliper_mm": 120.0}]}
    r = client.patch("/api/scans/scn_knob/parts/p1", json=patch)
    assert r.status_code == 200
    part = Part.model_validate(r.json())
    assert part.measurement_revision == 2
    assert part.source == "user_edited"


def test_scale_409_on_stale_expected_revision(client):
    # scn_trap reconstruction_revision is 1; expecting 0 is stale.
    body = {"point_a_mm": [0, 0, 0], "point_b_mm": [100, 0, 0], "known_length_mm": 100.0,
            "expected_revision": 0}
    assert client.post("/api/scans/scn_trap/scale", json=body).status_code == 409


def test_scale_accepts_matching_revision(client):
    body = {"point_a_mm": [0, 0, 0], "point_b_mm": [100, 0, 0], "known_length_mm": 100.0,
            "expected_revision": 1}
    assert client.post("/api/scans/scn_trap/scale", json=body).status_code == 200


def test_merge_creates_new_part_and_retires_sources(client):
    r = client.post("/api/scans/scn_trap/parts/merge",
                    json={"part_ids": ["p3", "p4"], "label": "nut+washer"})
    assert r.status_code == 200
    merged = Part.model_validate(r.json())
    assert merged.measurement_revision == 1
    ids = {p["part_id"] for p in client.get("/api/scans/scn_trap/parts").json()}
    assert "p3" not in ids and "p4" not in ids
    assert merged.part_id in ids


def test_merge_unknown_part_404(client):
    assert client.post("/api/scans/scn_trap/parts/merge",
                       json={"part_ids": ["p3", "pX"]}).status_code == 404


def test_merge_stales_cad_referencing_a_retired_part(client):
    # Build a CAD job on scn_knob that pins p2, then merge p1+p2 (retiring p2) -> the job goes stale.
    client.post("/api/scans/scn_knob/cad", json=KNOB_CAD_BODY)
    assert CadResult.model_validate(
        client.get("/api/scans/scn_knob/cad/cad_test_01").json()).stale is False

    r = client.post("/api/scans/scn_knob/parts/merge", json={"part_ids": ["p1", "p2"]})
    assert r.status_code == 200
    ids = {p["part_id"] for p in client.get("/api/scans/scn_knob/parts").json()}
    assert "p2" not in ids  # retired

    assert CadResult.model_validate(
        client.get("/api/scans/scn_knob/cad/cad_test_01").json()).stale is True
