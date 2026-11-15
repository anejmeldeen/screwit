"""Every route in mock mode (SPEC 5, 8): reads from fixtures, writes to the temp DB."""

from __future__ import annotations

from screwit.contracts import (
    CreateScanResponse,
    DemoScan,
    MeasureResponse,
    Part,
    Reconstruction,
    Scan,
    SceneGraph,
)


def test_demo_scans(client):
    r = client.get("/api/demo-scans")
    assert r.status_code == 200
    scans = [DemoScan.model_validate(d) for d in r.json()]
    assert {s.scan_id for s in scans} == {
        "scn_sink", "scn_trap", "scn_knob", "scn_headphones", "scn_headphones_cat",
    }


def test_get_scan(client):
    r = client.get("/api/scans/scn_trap")
    assert r.status_code == 200
    scan = Scan.model_validate(r.json())
    assert scan.status == "ready"


def test_get_scan_404(client):
    assert client.get("/api/scans/nope").status_code == 404


def test_reconstruction_signs_assets(client):
    r = client.get("/api/scans/scn_trap/reconstruction")
    assert r.status_code == 200
    recon = Reconstruction.model_validate(r.json())
    assert recon.mesh_uri.startswith("/api/assets/")
    assert all(k.startswith("/api/assets/") for k in recon.keyframes)


def test_reconstruction_signs_mobile_uris_when_present(client):
    """Phone-sized variants (AGENTS.md Performance budgets) are signed like the other URIs."""
    from screwit.app.store import ScanStore

    store = ScanStore(client.app.state.registry.data)
    recon = store.get_reconstruction("scn_trap")
    with_mobile = recon.model_copy(update={
        "reconstruction_revision": recon.reconstruction_revision + 1,
        "splat_mobile_uri": "scans/scn_trap/splat/scene_mobile.spz",
        "mesh_mobile_uri": "scans/scn_trap/mesh/object_mobile.glb",
    })
    store.save_reconstruction(with_mobile)

    r = client.get("/api/scans/scn_trap/reconstruction")
    out = Reconstruction.model_validate(r.json())
    assert out.splat_mobile_uri.startswith("/api/assets/")
    assert out.mesh_mobile_uri.startswith("/api/assets/")


def test_reconstruction_mobile_uris_none_stay_none(client):
    """A scan without phone variants returns null, not a signed URL for an empty key."""
    from screwit.app.store import ScanStore

    store = ScanStore(client.app.state.registry.data)
    recon = store.get_reconstruction("scn_trap")
    store.save_reconstruction(recon.model_copy(update={
        "reconstruction_revision": recon.reconstruction_revision + 1,
        "splat_mobile_uri": None,
        "mesh_mobile_uri": None,
    }))
    recon = Reconstruction.model_validate(client.get("/api/scans/scn_trap/reconstruction").json())
    assert recon.splat_mobile_uri is None
    assert recon.mesh_mobile_uri is None


def test_list_parts(client):
    r = client.get("/api/scans/scn_trap/parts")
    assert r.status_code == 200
    parts = [Part.model_validate(p) for p in r.json()]
    assert {p.part_id for p in parts} == {"p1", "p2", "p3", "p4"}


def test_scene_graph(client):
    r = client.get("/api/scans/scn_trap/scene-graph")
    assert r.status_code == 200
    SceneGraph.model_validate(r.json())


def test_measure_two_parts(client):
    r = client.post("/api/scans/scn_trap/measure", json={"a": "p1", "b": "p2"})
    assert r.status_code == 200
    resp = MeasureResponse.model_validate(r.json())
    assert resp.distance_mm > 0


def test_measure_two_points(client):
    r = client.post("/api/scans/scn_trap/measure",
                    json={"a": [0, 0, 0], "b": [0, 0, 10]})
    assert MeasureResponse.model_validate(r.json()).distance_mm == 10.0


def test_measure_unknown_part_404(client):
    assert client.post("/api/scans/scn_trap/measure", json={"a": "pX", "b": "p1"}).status_code == 404


def test_create_scan_returns_presigned(client):
    r = client.post("/api/scans", json={"input_type": "video"})
    assert r.status_code == 200
    resp = CreateScanResponse.model_validate(r.json())
    assert resp.scan_id.startswith("scn_")
    assert len(resp.uploads) == 1
    assert resp.uploads[0].key.endswith(".mov")


def test_create_scan_photos_count(client):
    r = client.post("/api/scans", json={"input_type": "photos", "photo_count": 3})
    resp = CreateScanResponse.model_validate(r.json())
    assert len(resp.uploads) == 3


def test_create_scan_mesh_requires_units(client):
    assert client.post("/api/scans", json={"input_type": "mesh"}).status_code == 422
    r = client.post("/api/scans", json={"input_type": "mesh", "mesh_units": "millimetres"})
    assert r.status_code == 200


def test_created_scan_is_retrievable(client):
    sid = client.post("/api/scans", json={"input_type": "video"}).json()["scan_id"]
    r = client.get(f"/api/scans/{sid}")
    assert r.status_code == 200
    assert Scan.model_validate(r.json()).status == "uploaded"


def test_chat_history_starts_empty(client):
    r = client.get("/api/scans/scn_trap/chat")
    assert r.status_code == 200
    assert r.json() == []
