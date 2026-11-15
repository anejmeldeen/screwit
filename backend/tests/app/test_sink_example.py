"""The sink example has five real components and preserves them through edits/exports."""
import io

import numpy as np
import trimesh


def test_sink_example_has_five_watertight_components(client):
    parts = client.get("/api/scans/scn_sink/parts").json()
    assert len(parts) == 5
    assert [p["label"] for p in parts] == ["Sink basin", "Mixer faucet", "Drain and tailpiece", "P-trap", "Outlet pipe"]
    for part in parts:
        response = client.get(part["mesh_uri"])
        assert response.status_code == 200
        mesh = trimesh.load(io.BytesIO(response.content), file_type="glb", force="scene").to_geometry()
        assert mesh.is_watertight
        assert mesh.volume > 0
        np.testing.assert_allclose(mesh.extents, part["obb"]["extent"], atol=1e-4)
    assert any(d["scan_id"] == "scn_sink" for d in client.get("/api/demo-scans").json())


def test_sink_combine_undo_and_stl_preserve_geometry(client):
    base = "/api/scans/scn_sink"
    before = client.get(base + "/export?format=stl")
    assert before.status_code == 200
    original = trimesh.load(io.BytesIO(before.content), file_type="stl")
    merged = client.post(base + "/parts/merge", json={"part_ids": ["p3", "p4"]})
    assert merged.status_code == 200
    assert len(client.get(base + "/parts").json()) == 4
    after = client.get(base + "/export?format=stl")
    mesh = trimesh.load(io.BytesIO(after.content), file_type="stl")
    assert len(mesh.faces) == len(original.faces)
    np.testing.assert_allclose(mesh.bounds, original.bounds)
    restored = client.post(base + "/parts/undo")
    assert restored.status_code == 200
    assert len(restored.json()) == 5


def test_nested_combine_keeps_first_selected_color(client):
    base = "/api/scans/scn_sink/parts"
    first = client.post(base + "/merge", json={"part_ids": ["p4", "p1"]}).json()
    assert first["color_source_id"] == "p4"
    second = client.post(base + "/merge", json={"part_ids": [first["part_id"], "p2"]}).json()
    saved = next(p for p in client.get(base).json() if p["part_id"] == second["part_id"])
    assert saved["color_source_id"] == "p4"
