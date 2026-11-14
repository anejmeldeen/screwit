"""PerceptionImpl services (SPEC 5 parts routes, 6.2 step 6, 6.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pydantic import TypeAdapter

from screwit.contracts import (
    MeasureRequest,
    MergeRequest,
    Part,
    PartPatch,
    SceneGraph,
)
from screwit.contracts.models import MeasurementEdit
from screwit.perception.service import GRAPH_KEY, PARTS_KEY, PerceptionImpl

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "fixtures" / "scans"


def _seed(storage, scan_id: str):
    """Load the committed JSON fixtures into storage and synthesise the part GLBs they reference."""
    parts = TypeAdapter(list[Part]).validate_json((FIX / scan_id / "parts.json").read_text())
    storage.put_bytes(PARTS_KEY.format(scan_id=scan_id),
                      (FIX / scan_id / "parts.json").read_bytes(), "application/json")
    storage.put_bytes(GRAPH_KEY.format(scan_id=scan_id),
                      (FIX / scan_id / "scene_graph.json").read_bytes(), "application/json")
    for p in parts:
        box = trimesh.creation.box(extents=list(p.obb.extent))
        box.apply_translation(list(p.centroid_mm))
        box.visual.vertex_colors = np.tile([180, 180, 180, 255], (len(box.vertices), 1))
        storage.put_bytes(p.mesh_uri, box.export(file_type="glb"), "model/gltf-binary")
    return parts


def test_patch_rename_does_not_bump_revision(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    before = {p.part_id: p for p in svc._load_parts("scn_trap", storage)}["p2"]
    out = svc.patch_part("scn_trap", "p2", PartPatch(label="new label"), storage)
    assert out.label == "new label"
    assert out.measurement_revision == before.measurement_revision  # label edit: no bump
    assert out.source == before.source


def test_patch_caliper_bumps_revision_and_overrides_snap(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    before = {p.part_id: p for p in svc._load_parts("scn_trap", storage)}["p2"]
    # p2_f1 radius is currently snapped; a caliper value must override the snap.
    patch = PartPatch(measurements=[MeasurementEdit(feature_id="p2_f1", field="radius",
                                                    caliper_mm=18.0)])
    out = svc.patch_part("scn_trap", "p2", patch, storage)
    feat = out.feature("p2_f1")
    assert feat.radius.caliper_mm == 18.0
    assert feat.radius.preferred == "caliper"
    assert feat.radius.snapped_mm is None  # caliper cleared the snap
    assert out.measurement_revision == before.measurement_revision + 1
    assert out.source == "user_edited"


def test_patch_user_confirmed_keeps_snap_and_bumps(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    patch = PartPatch(measurements=[MeasurementEdit(feature_id="p2_f1", field="radius",
                                                    user_confirmed=True)])
    out = svc.patch_part("scn_trap", "p2", patch, storage)
    feat = out.feature("p2_f1")
    assert feat.radius.user_confirmed is True
    assert feat.radius.snapped_mm is not None  # a bare confirm keeps the snap
    assert feat.radius.param_source == "user_confirmed"


def test_patch_rejects_unknown_feature_or_field(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    with pytest.raises(ValueError):
        svc.patch_part("scn_trap", "p2",
                       PartPatch(measurements=[MeasurementEdit(feature_id="nope", field="radius",
                                                               caliper_mm=1.0)]), storage)
    with pytest.raises(ValueError):
        svc.patch_part("scn_trap", "p2",
                       PartPatch(measurements=[MeasurementEdit(feature_id="p2_f1", field="depth",
                                                               caliper_mm=1.0)]), storage)


def test_merge_parts_retires_sources_and_resets_revision(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    merged = svc.merge_parts("scn_trap", MergeRequest(part_ids=["p3", "p4"], label="nut+washer"),
                             storage)
    assert merged.measurement_revision == 1
    assert merged.label == "nut+washer"
    assert merged.part_id not in ("p3", "p4")
    remaining = {p.part_id for p in svc._load_parts("scn_trap", storage)}
    assert "p3" not in remaining and "p4" not in remaining
    assert merged.part_id in remaining
    assert set(svc.retired_ids()) == {"p3", "p4"}
    graph = SceneGraph.model_validate_json(
        storage.get_bytes(GRAPH_KEY.format(scan_id="scn_trap")).decode())
    assert set(graph.parts) == remaining
    for r in graph.relations:  # no relation references a retired id
        assert r.a in remaining and r.b in remaining


def test_measure_point_to_point(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    res = svc.measure("scn_trap", MeasureRequest(a=(0, 0, 0), b=(0, 0, 100)), storage)
    assert res.distance_mm == pytest.approx(100.0)


def test_measure_part_to_part(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    res = svc.measure("scn_trap", MeasureRequest(a="p1", b="p3"), storage)
    assert res.distance_mm > 0


def test_rescale_parts_is_analytic(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    before = {p.part_id: p for p in svc._load_parts("scn_trap", storage)}
    out = {p.part_id: p for p in svc.rescale_parts("scn_trap", 2.0, 2, storage)}
    b, a = before["p1"], out["p1"]
    assert a.reconstruction_revision == 2
    assert a.measurement_revision == b.measurement_revision + 1
    assert np.allclose(a.centroid_mm, np.array(b.centroid_mm) * 2.0)
    assert np.allclose(a.obb.extent, np.array(b.obb.extent) * 2.0)
    bf, af = b.feature("p1_f1"), a.feature("p1_f1")
    assert af.radius.scan_mm == pytest.approx(bf.radius.scan_mm * 2.0)
    # caliper values are physical and are NOT scaled.
    if bf.radius.caliper_mm is not None:
        assert af.radius.caliper_mm == bf.radius.caliper_mm


def test_render_view_returns_png(storage):
    _seed(storage, "scn_trap")
    svc = PerceptionImpl()
    png = svc.render_view("scn_trap", ["p1", "p3"], 30.0, storage)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_parts_pipeline_end_to_end_writes_outputs(storage):
    """The full parts() path on a synthetic two-part object: lift, measure, label, persist."""
    from screwit.contracts import Cameras
    from screwit.interfaces import FrameSet

    # Two stacked boxes labelled 1 (lower) and 2 (upper); hand the fake a per-frame label image.
    lower = trimesh.creation.box(extents=(40, 40, 40))
    lower.apply_translation((0, 0, 20))
    upper = trimesh.creation.box(extents=(30, 30, 30))
    upper.apply_translation((0, 0, 60))
    mesh = trimesh.util.concatenate([lower, upper])
    mesh_path = Path(storage.root) / "mesh.glb"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)

    import math
    w = h = 200
    K = ((300.0, 0, w / 2), (0, 300.0, h / 2), (0, 0, 1.0))
    frames, labels, depths = [], [], []
    verts = np.asarray(mesh.vertices, float)
    vert_label = np.where(verts[:, 2] < 40, 1, 2).astype(np.int32)
    for i in range(10):
        ang = 2 * math.pi * i / 10
        eye = np.array([160 * math.cos(ang), 160 * math.sin(ang), 70.0])
        fwd = np.array([0, 0, 40.0]) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=0)
        t = -R @ eye
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = t
        frames.append({"frame": f"f{i}.jpg", "width": w, "height": h,
                       "world_to_camera": tuple(tuple(float(x) for x in r) for r in m), "K": K})
        # Dense label/depth image by projecting+splatting vertices.
        lab = np.zeros((h, w), np.int32)
        dep = np.zeros((h, w), np.float32)
        zb = np.full((h, w), np.inf)
        cam = verts @ R.T + t
        z = cam[:, 2]
        uvw = cam @ np.asarray(K).T
        px = uvw[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z)
        py = uvw[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z)
        for k in range(len(verts)):
            if z[k] <= 0:
                continue
            for dj in range(-3, 4):
                for di in range(-3, 4):
                    jj, ii = int(round(px[k])) + dj, int(round(py[k])) + di
                    if 0 <= ii < h and 0 <= jj < w and z[k] < zb[ii, jj]:
                        zb[ii, jj] = z[k]
                        dep[ii, jj] = z[k]
                        lab[ii, jj] = vert_label[k]
        labels.append(lab)
        depths.append(dep)

    cams = Cameras(scan_id="syn", reconstruction_revision=1, frames=frames)
    fs = FrameSet(images=[], depth_mm=depths, confidence=[], cameras=cams, full_res_paths=[])
    fs.part_label_images = labels  # the fake part segmenter path

    svc = PerceptionImpl()
    parts, graph = svc.parts("syn", fs, [li > 0 for li in labels], mesh_path, 1, storage)
    assert len(parts) == 2
    # Every produced Part validates and its GLB + thumbnail were written.
    for p in parts:
        assert storage.head(p.mesh_uri) is not None
        assert storage.head(p.thumbnail_uri) is not None
    assert json.loads(storage.get_bytes(PARTS_KEY.format(scan_id="syn")).decode())
    assert sorted(graph.parts) == sorted(p.part_id for p in parts)
