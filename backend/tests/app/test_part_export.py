"""Round-trip real downloads: units, node transforms, solids and failure responses."""

import io
from pathlib import Path

import numpy as np
import pytest
import trimesh

from screwit.cad.part_export import convert


def test_step_download_round_trip(client, tmp_path):
    cq = pytest.importorskip("cadquery")
    response = client.get("/api/scans/scn_trap/parts/p1/export?format=stp")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "model/step"
    assert "attachment" in response.headers["content-disposition"]
    path = tmp_path / "tailpiece.stp"
    path.write_bytes(response.content)
    shape = cq.importers.importStep(str(path)).val()
    source = Path(client.app.state.settings.fixtures_dir) / "scans/scn_trap/parts/p1.glb"
    mesh = trimesh.load(source, force="scene").to_geometry()
    box = shape.BoundingBox()
    assert shape.isValid()
    assert len(shape.Solids()) == 1
    np.testing.assert_allclose([box.xlen, box.ylen, box.zlen], mesh.extents, atol=1e-4)
    assert shape.Volume() == pytest.approx(abs(mesh.volume), rel=1e-5)


def test_stl_download_and_missing_part(client):
    response = client.get("/api/scans/scn_knob/parts/p1/export?format=stl")
    assert response.status_code == 200
    mesh = trimesh.load(io.BytesIO(response.content), file_type="stl")
    assert len(mesh.faces) > 0
    assert client.get("/api/scans/scn_knob/parts/missing/export").status_code == 404
    assert client.get("/api/scans/scn_knob/parts/p1/export?format=exe").status_code == 422


def test_step_preserves_transformed_disconnected_parts(tmp_path):
    cq = pytest.importorskip("cadquery")
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box([10, 20, 30]))
    transform = np.eye(4)
    transform[:3, 3] = [50, 20, 10]
    scene.add_geometry(trimesh.creation.box([8, 12, 16]), transform=transform)
    source, target = tmp_path / "combined.glb", tmp_path / "combined.stp"
    source.write_bytes(scene.export(file_type="glb"))
    convert(source, target)
    shape = cq.importers.importStep(str(target)).val()
    box = shape.BoundingBox()
    assert len(shape.Solids()) == 2
    np.testing.assert_allclose([box.xmin, box.ymin, box.zmin], scene.bounds[0], atol=1e-5)
    np.testing.assert_allclose([box.xmax, box.ymax, box.zmax], scene.bounds[1], atol=1e-5)


def test_step_rejects_open_mesh_with_actionable_message(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    source = tmp_path / "open.glb"
    source.write_bytes(mesh.export(file_type="glb"))
    with pytest.raises(ValueError, match="open edges"):
        convert(source, tmp_path / "open.stp")


def test_whole_object_exports_current_combined_geometry(client, tmp_path):
    cq = pytest.importorskip("cadquery")
    url = "/api/scans/scn_knob"
    merged = client.post(url + "/parts/merge", json={"part_ids": ["p1", "p2"]})
    assert merged.status_code == 200
    part_response = client.get(url + "/parts/" + merged.json()["part_id"] + "/export?format=stl")
    response = client.get(url + "/export?format=stl")
    assert response.status_code == 200, response.text
    expected = trimesh.load(io.BytesIO(part_response.content), file_type="stl")
    actual = trimesh.load(io.BytesIO(response.content), file_type="stl")
    np.testing.assert_allclose(actual.bounds, expected.bounds)
    assert len(actual.faces) == len(expected.faces)
    step = client.get(url + "/export?format=stp")
    assert step.status_code == 200, step.text
    path = tmp_path / "object.stp"
    path.write_bytes(step.content)
    shape = cq.importers.importStep(str(path)).val()
    assert shape.isValid()
    assert len(shape.Solids()) == 2
