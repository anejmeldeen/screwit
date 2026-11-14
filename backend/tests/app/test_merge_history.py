import io

import numpy as np
import trimesh


def geometry(client, part):
    response = client.get(part["mesh_uri"])
    assert response.status_code == 200
    return trimesh.load(io.BytesIO(response.content), file_type="glb", force="scene").to_geometry()


def test_merge_keeps_geometry_and_undo_restores_parts(client):
    url = "/api/scans/scn_trap/parts"
    before = client.get(url).json()
    sources = [p for p in before if p["part_id"] in {"p3", "p4"}]
    expected = trimesh.util.concatenate([geometry(client, p) for p in sources])
    response = client.post(url + "/merge", json={"part_ids": ["p3", "p4"]})
    assert response.status_code == 200, response.text
    merged = geometry(client, response.json())
    assert len(merged.faces) == len(expected.faces)
    np.testing.assert_allclose(merged.bounds, expected.bounds)
    assert client.get(url + "/history").json()["can_undo"]
    undo = client.post(url + "/undo")
    assert undo.status_code == 200, undo.text
    assert sorted(undo.json(), key=lambda p: p["part_id"]) == sorted(before, key=lambda p: p["part_id"])
    assert not client.get(url + "/history").json()["can_undo"]


def test_repeated_merge_undo_and_restore_example(client):
    url = "/api/scans/scn_trap/parts"
    before = client.get(url).json()
    first = client.post(url + "/merge", json={"part_ids": ["p3", "p4"]}).json()
    response = client.post(url + "/merge", json={"part_ids": ["p1", first["part_id"]]})
    assert response.status_code == 200, response.text
    assert client.post(url + "/undo").status_code == 200
    assert client.post(url + "/undo").status_code == 200
    assert len(client.get(url).json()) == len(before)
    client.post(url + "/merge", json={"part_ids": ["p3", "p4"]})
    restored = client.post(url + "/restore-example")
    assert restored.status_code == 200
    assert len(restored.json()) == len(before)
    assert not client.get(url + "/history").json()["can_undo"]


def test_undo_does_not_overwrite_later_edits(client):
    url = "/api/scans/scn_trap/parts"
    merged = client.post(url + "/merge", json={"part_ids": ["p3", "p4"]}).json()
    assert client.patch(url + "/" + merged["part_id"], json={"label": "My combined part"}).status_code == 200
    assert client.post(url + "/undo").status_code == 409
