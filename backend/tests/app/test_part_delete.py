"""Part deletion: DELETE /parts/{part_id} (screwit.app.part_delete), its undo step, and the remove_part
chat tool. Mock mode against the fixture scans, plus live mode against a LocalStorage."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from screwit.agent.remove_part import remove_part, tool_spec
from screwit.app.store import ScanStore
from screwit.contracts import CadRequest, CadResult, Part, SceneGraph

TRAP = "/api/scans/scn_trap"
KNOB = "/api/scans/scn_knob"


def _ids(client: TestClient, base: str) -> list[str]:
    return [p["part_id"] for p in client.get(f"{base}/parts").json()]


def _graph(client: TestClient, base: str) -> dict:
    response = client.get(f"{base}/scene-graph")
    assert response.status_code == 200, response.text
    return response.json()


def _pairs(graph: dict) -> list[tuple[str, str, str]]:
    return [(r["a"], r["b"], r["type"]) for r in graph["relations"]]


def _store(client: TestClient) -> ScanStore:
    return ScanStore(client.app.state.registry.data)


# ---------------------------------------------------------------- the route
def test_delete_removes_the_part_from_parts_and_scene_graph(client):
    before = _ids(client, TRAP)
    response = client.delete(f"{TRAP}/parts/p3")
    assert response.status_code == 204 and response.content == b""

    assert _ids(client, TRAP) == [pid for pid in before if pid != "p3"]
    graph = _graph(client, TRAP)
    SceneGraph.model_validate(graph)
    assert graph["parts"] == ["p1", "p2", "p4"]
    assert _pairs(graph) == []  # every fixture relation names p3 (threaded, fastened, seals)
    assert [h["part_id"] for h in graph["failure_hypotheses"]] == ["p4"]
    assert client.get(f"{TRAP}/parts/history").json()["can_undo"] is True


def test_delete_keeps_relations_and_hypotheses_of_other_parts(client):
    assert client.delete(f"{TRAP}/parts/p1").status_code == 204
    graph = _graph(client, TRAP)
    assert graph["parts"] == ["p2", "p3", "p4"]
    assert _pairs(graph) == [("p3", "p2", "threaded"), ("p4", "p3", "seals")]
    assert [h["part_id"] for h in graph["failure_hypotheses"]] == ["p4", "p3"]


def test_delete_keeps_the_mesh_for_undo(client):
    part = next(p for p in client.get(f"{TRAP}/parts").json() if p["part_id"] == "p3")
    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    assert client.get(part["mesh_uri"]).status_code == 200


def test_unknown_part_is_404_and_fixed_part_routes_still_work(client):
    assert client.delete(f"{TRAP}/parts/p99").status_code == 404
    # The fixed /parts/... paths answer other methods; DELETE on them names an unknown part (not 405).
    for name in ("merge", "undo", "history", "restore-example"):
        assert client.delete(f"{TRAP}/parts/{name}").status_code == 404
    assert client.get(f"{TRAP}/parts/history").json() == {"can_undo": False, "can_restore_example": True}
    assert client.post(f"{TRAP}/parts/undo").status_code == 409
    assert client.post(f"{TRAP}/parts/merge", json={"part_ids": ["p3", "p4"]}).status_code == 200
    assert client.post(f"{TRAP}/parts/undo").status_code == 200
    assert _ids(client, TRAP) == ["p1", "p2", "p3", "p4"]


def test_other_accounts_anonymous_and_cross_site_requests_cannot_delete(client):
    alice = TestClient(client.app, client=("127.0.0.1", 50001))
    assert alice.post("/api/auth/register",
                      json={"username": "alice", "password": "a-password"}).status_code == 201
    assert alice.delete(f"{TRAP}/parts/p1").status_code == 404
    assert TestClient(client.app, client=("127.0.0.1", 50002)).delete(f"{TRAP}/parts/p1").status_code == 401
    assert client.delete(f"{TRAP}/parts/p1", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert "p1" in _ids(client, TRAP)
    assert client.get(f"{TRAP}/parts/history").json()["can_undo"] is False


# ---------------------------------------------------------------- undo
def test_undo_restores_the_part_in_place_with_its_graph_entry(client):
    parts = client.get(f"{TRAP}/parts").json()
    graph = _graph(client, TRAP)
    assert client.delete(f"{TRAP}/parts/p2").status_code == 204

    undo = client.post(f"{TRAP}/parts/undo")
    assert undo.status_code == 200, undo.text
    assert undo.json() == parts  # same parts, same list order
    assert client.get(f"{TRAP}/parts").json() == parts
    assert _graph(client, TRAP) == graph  # relations and failure hypotheses are back
    assert client.get(f"{TRAP}/parts/history").json()["can_undo"] is False


def test_undo_steps_back_through_deletes_and_merges(client):
    parts = client.get(f"{TRAP}/parts").json()
    graph = _graph(client, TRAP)
    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    assert client.post(f"{TRAP}/parts/merge", json={"part_ids": ["p1", "p2"]}).status_code == 200
    assert client.delete(f"{TRAP}/parts/p4").status_code == 204
    for _ in range(3):
        assert client.post(f"{TRAP}/parts/undo").status_code == 200
    assert client.get(f"{TRAP}/parts").json() == parts
    assert _graph(client, TRAP) == graph
    assert client.post(f"{TRAP}/parts/undo").status_code == 409


def test_delete_matches_the_graph_to_the_parts_after_a_merge(client):
    """Merge leaves the stored scene graph as it was; a delete afterwards matches the graph to the parts
    again, and undoing the delete restores a graph that matches the parts it restores."""
    merged = client.post(f"{TRAP}/parts/merge", json={"part_ids": ["p3", "p4"]}).json()["part_id"]
    assert client.delete(f"{TRAP}/parts/p1").status_code == 204
    graph = _graph(client, TRAP)
    assert graph["parts"] == ["p2", merged] == _ids(client, TRAP)
    assert graph["relations"] == [] and graph["failure_hypotheses"] == []
    assert client.post(f"{TRAP}/parts/undo").status_code == 200
    assert _graph(client, TRAP)["parts"] == ["p1", "p2", merged]
    assert sorted(_ids(client, TRAP)) == sorted(["p1", "p2", merged])


def test_graph_changes_read_back_as_latest_above_older_revisions(client):
    """The store reads back the graph row with the highest revision. A row saved above the parts'
    revision (add_part saves at the scan's) must not hide the graph a delete or its undo saves."""
    store = _store(client)
    store.save_scene_graph(store.get_scene_graph("scn_trap"), 5)
    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    assert "p3" not in _graph(client, TRAP)["parts"]
    assert client.post(f"{TRAP}/parts/undo").status_code == 200
    assert "p3" in _graph(client, TRAP)["parts"]


def test_undo_refuses_to_overwrite_a_later_edit(client):
    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    assert client.patch(f"{TRAP}/parts/p1", json={"label": "renamed"}).status_code == 200
    response = client.post(f"{TRAP}/parts/undo")
    assert response.status_code == 409
    assert "changed" in response.json()["detail"]


def test_restore_example_brings_back_parts_and_scene_graph(client):
    graph = _graph(client, TRAP)
    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    restored = client.post(f"{TRAP}/parts/restore-example")
    assert restored.status_code == 200
    assert _ids(client, TRAP) == ["p1", "p2", "p3", "p4"]
    assert _graph(client, TRAP) == graph
    assert client.get(f"{TRAP}/parts/history").json()["can_undo"] is False


# ---------------------------------------------------------------- CAD staleness
def _pin_cad(client: TestClient, scan_id: str, cad_id: str, part_id: str) -> None:
    data = client.app.state.registry.data
    store = _store(client)
    request = CadRequest.model_validate({**data.cad_request("scn_knob").model_dump(mode="json"),
                                         "scan_id": scan_id, "cad_id": cad_id, "mates_with": [],
                                         "reconstruction_revision": store.get_scan(scan_id).reconstruction_revision,
                                         "part_measurement_revisions": {
                                             part_id: store.get_part(scan_id, part_id).measurement_revision}})
    result = data.cad_result("scn_knob").model_copy(update={"cad_id": cad_id, "stale": False})
    store.save_cad_job(request, result)


def test_delete_stales_only_cad_that_pinned_the_part(client):
    _pin_cad(client, "scn_trap", "cad_pins_p3", "p3")
    _pin_cad(client, "scn_trap", "cad_pins_p1", "p1")
    store = _store(client)

    def stale(cad_id: str) -> tuple[bool, bool]:
        """(the flag stored with the result, what GET /cad reports)"""
        served = CadResult.model_validate(client.get(f"{TRAP}/cad/{cad_id}").json()).stale
        return store.get_cad_job("scn_trap", cad_id)[1].stale, served

    assert stale("cad_pins_p3") == (False, False) and stale("cad_pins_p1") == (False, False)
    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    assert stale("cad_pins_p3") == (True, True)
    assert stale("cad_pins_p1") == (False, False)


# ---------------------------------------------------------------- added parts and the last part
def _add_part(client: TestClient, scan_id: str, part_id: str, attach_to: str) -> None:
    """A part as add_part saves it: source user_edited, an added_ id, a relation in the scene graph."""
    store = _store(client)
    base = store.list_parts(scan_id)[0]
    added = Part.model_validate({**base.model_dump(mode="json"), "part_id": part_id, "label": "Cat ear",
                                 "source": "user_edited", "measurement_revision": 1,
                                 "features": [], "dimensions": []})
    store.save_part(scan_id, added)
    graph = store.get_scene_graph(scan_id)
    relation = {"a": part_id, "b": attach_to, "type": "rests_on", "confidence": 1.0, "note": "added in chat"}
    store.save_scene_graph(SceneGraph.model_validate({
        **graph.model_dump(mode="json"), "parts": [*graph.parts, part_id],
        "relations": [*graph.model_dump(mode="json")["relations"], relation]}),
        store.get_scan(scan_id).reconstruction_revision)


def test_delete_an_added_part_and_undo_it(client):
    _add_part(client, "scn_knob", "added_0123456789ab", attach_to="p1")
    assert _ids(client, KNOB) == ["p1", "p2", "added_0123456789ab"]
    assert client.delete(f"{KNOB}/parts/added_0123456789ab").status_code == 204
    assert _ids(client, KNOB) == ["p1", "p2"]
    graph = _graph(client, KNOB)
    assert graph["parts"] == ["p1", "p2"] and _pairs(graph) == [("p2", "p1", "fastened")]

    assert client.post(f"{KNOB}/parts/undo").status_code == 200
    assert _ids(client, KNOB) == ["p1", "p2", "added_0123456789ab"]
    graph = _graph(client, KNOB)
    assert ("added_0123456789ab", "p1", "rests_on") in _pairs(graph)


def test_deleting_the_last_part_of_a_workspace_and_undo(client):
    store = _store(client)
    scan = store.get_scan("scn_knob").model_copy(update={"scan_id": "scn_single"})
    store.save_scan(scan, owner="admin")
    store.save_parts("scn_single", [store.get_part("scn_knob", "p1")])
    store.save_scene_graph(SceneGraph.model_validate({
        **store.get_scene_graph("scn_knob").model_dump(mode="json"), "scan_id": "scn_single",
        "parts": ["p1"], "relations": [], "failure_hypotheses": []}), scan.reconstruction_revision)
    base = "/api/scans/scn_single"

    assert client.delete(f"{base}/parts/p1").status_code == 204
    assert client.get(f"{base}/parts").json() == []
    assert _graph(client, base)["parts"] == []
    assert client.get(f"{base}/parts/history").json()["can_undo"] is True
    undo = client.post(f"{base}/parts/undo")
    assert undo.status_code == 200, undo.text
    assert [p["part_id"] for p in undo.json()] == ["p1"]
    assert _ids(client, base) == ["p1"] and _graph(client, base)["parts"] == ["p1"]


def test_the_last_part_of_an_example_is_kept(client):
    """An example's parts come from the fixtures whenever the database holds none, so its last part
    cannot go: an empty list would bring all the original parts back."""
    assert client.delete(f"{KNOB}/parts/p1").status_code == 204
    response = client.delete(f"{KNOB}/parts/p2")
    assert response.status_code == 409
    assert "at least one part" in response.json()["detail"]
    assert _ids(client, KNOB) == ["p2"]
    assert client.post(f"{KNOB}/parts/undo").status_code == 200
    assert _ids(client, KNOB) == ["p1", "p2"]


# ---------------------------------------------------------------- live mode storage files
def test_live_mode_updates_storage_files_and_undo_restores_them(client, tmp_path):
    from screwit.storage import LocalStorage

    fixtures = Path(os.environ["SCREWIT_FIXTURES_DIR"]) / "scans" / "scn_trap"
    storage = LocalStorage(tmp_path / "live")
    stored = json.loads((fixtures / "parts.json").read_text())
    extra = {**stored[0], "part_id": "p9", "label": "only in parts.json"}
    storage.put_bytes("scans/scn_trap/parts.json", json.dumps(stored + [extra]).encode(), "application/json")
    storage.put_bytes("scans/scn_trap/scene_graph.json", (fixtures / "scene_graph.json").read_bytes(),
                      "application/json")
    storage.put_bytes("scans/scn_trap/parts/p3.glb", b"glTF", "model/gltf-binary")
    reg = client.app.state.registry
    client.app.state.registry = replace(reg, settings=replace(reg.settings, mode="live"), storage=storage)

    assert client.delete(f"{TRAP}/parts/p3").status_code == 204
    files = [p["part_id"] for p in json.loads(storage.get_bytes("scans/scn_trap/parts.json"))]
    assert files == ["p1", "p2", "p4", "p9"]  # removed as stored: storage's own entry survives
    graph = json.loads(storage.get_bytes("scans/scn_trap/scene_graph.json"))
    assert "p3" not in graph["parts"] and graph["relations"] == []
    assert [h["part_id"] for h in graph["failure_hypotheses"]] == ["p4"]
    assert storage.head("scans/scn_trap/parts/p3.glb") is not None
    assert _ids(client, TRAP) == ["p1", "p2", "p4"]

    assert client.post(f"{TRAP}/parts/undo").status_code == 200
    # Undo writes the database's snapshot from before the delete back to storage.
    files = [p["part_id"] for p in json.loads(storage.get_bytes("scans/scn_trap/parts.json"))]
    assert files == ["p1", "p2", "p3", "p4"]
    graph = json.loads(storage.get_bytes("scans/scn_trap/scene_graph.json"))
    assert graph["parts"] == ["p1", "p2", "p3", "p4"]
    assert ("p3", "p2", "threaded") in _pairs(graph)
    assert _ids(client, TRAP) == ["p1", "p2", "p3", "p4"]


# ---------------------------------------------------------------- the remove_part tool
def test_remove_part_tool_spec():
    spec = tool_spec()
    assert spec["name"] == "remove_part"
    assert spec["input_schema"]["required"] == ["part_id"]
    assert spec["input_schema"]["properties"] == {"part_id": {
        "type": "string", "description": spec["input_schema"]["properties"]["part_id"]["description"]}}
    assert "Undo" in spec["description"] and "asked" in spec["description"]


@pytest.mark.parametrize("args, error, fragment", [
    ("p3", "bad_request", "object of arguments"),
    ({}, "bad_request", "part_id"),
    ({"part_id": 3}, "bad_request", "part_id"),
    ({"part_id": "../p3"}, "bad_request", "part_id"),
    ({"part_id": "p3", "force": True}, "bad_request", "unknown argument"),
    ({"part_id": "p99"}, "unknown_part", "p1, p2, p3, p4"),
])
def test_remove_part_tool_rejects_bad_input(client, args, error, fragment):
    reg = client.app.state.registry
    result = remove_part(reg, ScanStore(reg.data), "scn_trap", args)
    assert result["error"] == error and fragment in result["message"]
    assert _ids(client, TRAP) == ["p1", "p2", "p3", "p4"]


def test_remove_part_tool_deletes_undoably_and_refuses_an_examples_last_part(client):
    reg = client.app.state.registry
    store = ScanStore(reg.data)
    result = remove_part(reg, store, "scn_trap", {"part_id": "p3"})
    assert {k: result[k] for k in ("status", "part_id", "label")} == {
        "status": "removed", "part_id": "p3", "label": "slip nut"}
    json.dumps(result)  # the loop sends it to the model as JSON
    assert _ids(client, TRAP) == ["p1", "p2", "p4"]
    assert "p3" not in _graph(client, TRAP)["parts"]
    assert remove_part(reg, store, "scn_trap", {"part_id": "p3"})["error"] == "unknown_part"
    assert client.post(f"{TRAP}/parts/undo").status_code == 200
    assert _ids(client, TRAP) == ["p1", "p2", "p3", "p4"]

    assert remove_part(reg, store, "scn_knob", {"part_id": "p1"})["status"] == "removed"
    refused = remove_part(reg, store, "scn_knob", {"part_id": "p2"})
    assert refused["error"] == "refused" and "at least one part" in refused["message"]
