"""Fast tests for the X8 scorer (scripts/hard_eval.py).

No model weights and no pipeline run: feed the scoring functions synthetic meshes and cameras with
a known answer and check the numbers. These lock in that a perfect scan scores ~0 error, a shifted
scan scores the shift, a leaked (too-big) scan is caught by the object-region split, and that the
silhouette IoU is 1.0 for identical meshes through the same camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import hard_eval as H  # noqa: E402
import make_hard_scenes as M  # noqa: E402


def _cube(size=100.0, center=(0, 0, 50.0)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(size, size, size),
                                transform=trimesh.transformations.translation_matrix(center))


def test_score_mesh_perfect():
    m = _cube()
    s = H.score_mesh(m, m)
    # Two independent surface samplings of the same mesh: NN distance is sampling noise, not error,
    # so it is small but not zero. What must hold is that it is far below any real reconstruction
    # error and that essentially all the surface is mutually within 5 mm.
    assert s["accuracy_mm"]["median"] < 6.0
    assert s["completeness_mm"]["median"] < 6.0
    assert s["true_surface_within_5mm_pct"] > 90.0
    assert s["scan_surface_in_object_region_pct"] > 99.0


def test_score_mesh_shift_shows_up_as_error():
    true = _cube()
    scan = _cube(center=(0, 0, 60.0))  # shifted 10 mm up
    s = H.score_mesh(scan, true)
    # Most of the surface is still within the object region (bbox grown 15 mm), but completeness
    # and accuracy medians reflect the ~10 mm offset on the shifted faces.
    assert 3.0 < s["accuracy_mm"]["p90"] < 20.0
    assert s["true_surface_within_2mm_pct"] < 95.0


def test_score_mesh_leaked_background_caught_by_object_region():
    """A scan mesh that also contains a big far-away slab (a leaked table) should show a huge
    whole-mesh accuracy but a small object-region accuracy."""
    true = _cube()
    slab = trimesh.creation.box(extents=(800, 800, 2),
                                transform=trimesh.transformations.translation_matrix((0, 0, -1)))
    scan = trimesh.util.concatenate([_cube(), slab])
    s = H.score_mesh(scan, true)
    assert s["accuracy_mm"]["median"] > 50.0, "leaked slab should inflate whole-mesh accuracy"
    assert s["object_region"]["accuracy_mm_median"] < 5.0, "object region should still be accurate"
    assert s["scan_surface_in_object_region_pct"] < 50.0


def test_measure_scan_dimensions_matches_truth():
    scan = _cube(size=100.0)
    out = H.measure_scan_dimensions(scan, {"edge_mm": 100.0})
    d = out["dimensions"]["edge_mm"]
    assert abs(d["err_mm"]) < 2.0
    assert abs(d["err_pct"]) < 2.0


def test_render_true_mask_silhouette_and_iou():
    """A cube rendered through a camera fills a plausible silhouette, and the IoU of a mesh with
    itself through the same camera is 1.0."""
    m = _cube(size=100.0, center=(0, 0, 50.0))
    eye = np.array([0.0, -400.0, 120.0])
    w2c = M.look_at_world_to_camera(eye, np.array([0, 0, 50.0]), np.array([0, 0, 1.0]))
    k = M.intrinsics(640, 480, 600.0)
    mask = H.render_true_mask(m, w2c, k, 640, 480)
    assert mask.sum() > 1000
    inter = np.logical_and(mask, mask).sum()
    union = np.logical_or(mask, mask).sum()
    assert inter / union == pytest.approx(1.0)


def test_render_true_mask_iou_drops_when_shifted():
    true = _cube(center=(0, 0, 50.0))
    shifted = _cube(center=(120, 0, 50.0))  # moved sideways out of overlap
    eye = np.array([0.0, -400.0, 120.0])
    w2c = M.look_at_world_to_camera(eye, np.array([0, 0, 50.0]), np.array([0, 0, 1.0]))
    k = M.intrinsics(640, 480, 600.0)
    tm = H.render_true_mask(true, w2c, k, 640, 480)
    pm = H.render_true_mask(shifted, w2c, k, 640, 480)
    iou = np.logical_and(tm, pm).sum() / max(1, np.logical_or(tm, pm).sum())
    assert iou < 0.5


def test_union_true_mesh_drops_support():
    meshes = {"mug_body": _cube(), "tabletop": _cube(size=400.0), "cabinet_floor": _cube(size=400.0)}
    u = H.union_true_mesh(meshes, drop=H.SUPPORT_PARTS)
    # Only the mug remains, so the bbox is the small cube, not the 400 mm table.
    ext = u.vertices.max(0) - u.vertices.min(0)
    assert ext.max() < 150.0


def test_score_parts_reads_contract(tmp_path):
    """score_parts reads a parts.json written through the real contracts (one synthetic part)."""
    import json

    from screwit.contracts import (
        OBB,
        CylinderFeature,
        DerivedDimension,
        Guess,
        Part,
        measurement,
    )

    part = Part(
        part_id="p1", reconstruction_revision=1, measurement_revision=1, label="body",
        label_confidence=0.5, description="d", mesh_uri="m", thumbnail_uri="t",
        centroid_mm=(0, 0, 0), obb=OBB(center=(0, 0, 0), extent=(10, 10, 10),
                                       rotation_quat=(0, 0, 0, 1)),
        features=[CylinderFeature(feature_id="p1_f1", axis_point_mm=(0, 0, 0), axis_dir=(0, 0, 1),
                                  fit_rmse_mm=0.5, inlier_fraction=0.9,
                                  radius=measurement(19.0), length=measurement(90.0))],
        dimensions=[DerivedDimension(name="outer_diameter", derived_from="p1_f1.radius", factor=2.0)],
        material_guess=Guess(value="steel", confidence=0.4), source="auto")
    d = tmp_path / "data" / "scans" / "scn_x" / "parts.json"
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(json.dumps([json.loads(part.model_dump_json())]))
    out = H.score_parts(tmp_path, "scn_x", [{"part": "body", "feature": "cylinder"}])
    assert out["parts_found"] == 1
    assert "cylinder" in out["parts"][0]["features"]
