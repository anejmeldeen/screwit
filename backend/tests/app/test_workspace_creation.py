"""Workspace names persist and a photo submission contains every selected file."""
import json

import pytest

from screwit.app.db import PhotoUploadRow, db_session, reset_engine
from screwit.app.jobs import _PhotoBatchStorage


def test_workspace_title_persists_in_dashboard(client):
    created = client.post("/api/scans", json={"input_type": "video", "title": "  Kitchen tap repair  "})
    assert created.status_code == 200
    scan_id = created.json()["scan_id"]
    reset_engine()
    workspace = next(b for b in client.get("/api/builds").json() if b["scan_id"] == scan_id)
    assert workspace["title"] == "Kitchen tap repair"
    assert workspace["status"] == "uploaded"


@pytest.mark.parametrize("title", ["", "   ", "x" * 121])
def test_invalid_titles_do_not_create_workspaces(client, title):
    before = client.get("/api/builds").json()
    assert client.post("/api/scans", json={"input_type": "video", "title": title}).status_code == 422
    assert client.get("/api/builds").json() == before


def test_photo_upload_keys_match_scan_and_batch(client):
    created = client.post("/api/scans", json={"input_type": "photos", "photo_count": 3, "title": "Chair"})
    assert created.status_code == 200
    data = created.json()
    scan = client.get(f"/api/scans/{data['scan_id']}").json()
    keys = [upload["key"] for upload in data["uploads"]]
    assert scan["input"]["uri"] == keys[0]
    with db_session() as session:
        assert json.loads(session.get(PhotoUploadRow, data["scan_id"]).keys_json) == keys


def test_photos_all_required_before_start(client):
    created = client.post("/api/scans", json={"input_type": "photos", "photo_count": 2}).json()
    first = created["uploads"][0]["key"]
    storage = client.app.state.registry.storage
    storage.head = lambda key: {"size": 20} if key == first else None
    response = client.post(f"/api/scans/{created['scan_id']}/start")
    assert response.status_code == 400
    assert "every selected file" in response.json()["detail"]


def test_photo_batch_supplies_every_image_to_pipeline(tmp_path):
    class Storage:
        def local_path(self, key, scratch):
            return tmp_path / key
    for name in ["a.jpg", "b.jpg"]:
        (tmp_path / name).write_bytes(name.encode())
    batch = _PhotoBatchStorage(Storage(), "a.jpg", ["a.jpg", "b.jpg"])
    directory = batch.local_path("a.jpg", tmp_path / "job")
    assert [p.read_bytes() for p in sorted(directory.iterdir())] == [b"a.jpg", b"b.jpg"]
    assert batch.local_path("other.glb", tmp_path) == tmp_path / "other.glb"
