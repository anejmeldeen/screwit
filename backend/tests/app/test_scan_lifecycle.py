"""Scan create/start lifecycle and mesh stage-skip (SPEC 5, 4.1)."""

from __future__ import annotations

from screwit.contracts import Scan
from screwit.contracts.models import MESH_INPUT_SKIPPED


def test_mesh_scan_skips_the_right_stages(client):
    sid = client.post("/api/scans", json={"input_type": "mesh", "mesh_units": "metres"}).json()["scan_id"]
    scan = Scan.model_validate(client.get(f"/api/scans/{sid}").json())
    skipped = {s.name for s in scan.stages if s.state == "skipped"}
    assert skipped == set(MESH_INPUT_SKIPPED)


def test_video_scan_starts_no_skipped_stages(client):
    sid = client.post("/api/scans", json={"input_type": "video"}).json()["scan_id"]
    scan = Scan.model_validate(client.get(f"/api/scans/{sid}").json())
    assert all(s.state == "pending" for s in scan.stages)


def test_start_404_for_unknown_scan(client):
    assert client.post("/api/scans/nope/start").status_code == 404


def test_start_400_without_upload(client):
    # Mock FixtureStorage.head returns None for a key that is not in the fixtures tree.
    sid = client.post("/api/scans", json={"input_type": "video"}).json()["scan_id"]
    r = client.post(f"/api/scans/{sid}/start")
    assert r.status_code == 400
    assert "upload" in r.json()["detail"].lower()
