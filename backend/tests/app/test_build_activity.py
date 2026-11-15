"""Dashboard timestamps reflect saved changes, never merely viewing a workspace."""

from datetime import UTC, datetime

import pytest

from screwit.app.db import BuildActivityRow, db_session
from screwit.app.store import ScanStore
from screwit.contracts import ChatMessage


def test_existing_examples_have_a_stable_timestamp(client):
    builds = client.get("/api/builds").json()
    for build in builds:
        assert build["updated_at"] == build["created_at"]
        assert build["input_type"] in {"video", "photos", "mesh"}
    client.get("/api/scans/scn_knob/parts")
    assert client.get("/api/builds").json() == builds


def test_editing_a_part_updates_dashboard_and_sort_order(client):
    old = {b["scan_id"]: b for b in client.get("/api/builds").json()}
    parts = client.get("/api/scans/scn_trap/parts").json()
    response = client.patch(f"/api/scans/scn_trap/parts/{parts[0]['part_id']}", json={"label": "Updated part"})
    assert response.status_code == 200
    builds = client.get("/api/builds").json()
    assert builds[0]["scan_id"] == "scn_trap"
    assert datetime.fromisoformat(builds[0]["updated_at"]) > datetime.fromisoformat(old["scn_trap"]["updated_at"])
    assert builds[0]["created_at"] == old["scn_trap"]["created_at"]
    assert next(b for b in builds if b["scan_id"] == "scn_knob") == old["scn_knob"]
    assert client.get("/api/builds").json() == builds


@pytest.mark.parametrize("change", ["scan", "reconstruction", "parts", "graph", "chat", "cad"])
def test_saved_changes_are_recorded(client, change):
    reg = client.app.state.registry
    store = ScanStore(reg.data)
    scan_id = "scn_knob"
    old_time = datetime(2000, 1, 1, tzinfo=UTC)
    with db_session() as session:
        session.add(BuildActivityRow(scan_id=scan_id, updated_at=old_time))
        session.commit()
    if change == "scan":
        store.save_scan(store.get_scan(scan_id))
    elif change == "reconstruction":
        store.save_reconstruction(store.get_reconstruction(scan_id))
    elif change == "parts":
        store.save_parts(scan_id, store.list_parts(scan_id))
    elif change == "graph":
        store.save_scene_graph(store.get_scene_graph(scan_id), 1)
    elif change == "chat":
        store.append_chat(scan_id, ChatMessage(message_id="test", role="user", text="hello",
                                               created_at=datetime.now(UTC)))
    elif change == "cad":
        store.save_cad_job(reg.data.cad_request(scan_id), reg.data.cad_result(scan_id))
    build = next(b for b in client.get("/api/builds").json() if b["scan_id"] == scan_id)
    assert datetime.fromisoformat(build["updated_at"]) > old_time


def test_new_workspace_timestamp_persists(client):
    response = client.post("/api/scans", json={"input_type": "video"})
    assert response.status_code == 200
    scan_id = response.json()["scan_id"]
    first = next(b for b in client.get("/api/builds").json() if b["scan_id"] == scan_id)
    assert datetime.fromisoformat(first["updated_at"]) >= datetime.fromisoformat(first["created_at"])
    from screwit.app.db import reset_engine

    reset_engine()
    assert next(b for b in client.get("/api/builds").json() if b["scan_id"] == scan_id) == first
