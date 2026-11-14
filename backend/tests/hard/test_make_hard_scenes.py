"""Fast tests for the X8 hard-scene generator (scripts/make_hard_scenes.py).

No model weights and no full video renders: these render a handful of frames with the numpy
rasterizer and check the pieces that make the ground truth trustworthy -- that the board a real
ChArUco detector sees reprojects onto the generator's own cameras to sub-pixel accuracy (so the
scan-frame cameras and board positions are mutually consistent), that the rasterizer paints, labels
and z-buffers correctly, and that a portrait file decodes upright through the real pipeline frame
extractor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import make_hard_scenes as M  # noqa: E402


@pytest.fixture(scope="module")
def k():
    return M.intrinsics(M.RENDER_W, M.RENDER_H, 1150.0)


def test_board_corners_match_scale_module():
    """The generator's scan-frame board corners must match P2's scale.board_scan_points exactly, so
    a scale recovered against the real board is scored in the right frame."""
    from screwit.recon import scale as scale_mod

    ids, scan = M.board_scan_corners()
    ref = scale_mod.board_scan_points()
    assert scan.shape == ref.shape
    assert np.allclose(scan, ref, atol=1e-6), "board corners differ from scale.board_scan_points"


def test_board_decodes_and_reprojects(k):
    """A real ChArUco detector sees the board in the rendered orbit, and the detected corners
    reproject onto the generator's own cameras to < 1.5 px -- the core ground-truth check."""
    scene, meta = M.build_gso_primitive("mug")
    cams = M.scene_camera_path(meta["kind"], 24)
    detector = cv2.aruco.CharucoDetector(M.charuco_board())
    _ids, corners_scan = M.board_scan_corners()
    seen = 0
    errs = []
    for eye, w2c in cams:
        out = M.render(scene, w2c, k, eye)
        gray = cv2.cvtColor(out.rgb, cv2.COLOR_RGB2GRAY)
        cc, cids, _, _ = detector.detectBoard(gray)
        if cc is None or len(cc) < 1:
            continue
        seen += 1 if len(cc) >= 8 else 0
        proj, _z, _ = M._project(corners_scan[cids.flatten()], w2c, k)
        errs.append(float(np.nanmean(np.linalg.norm(proj - cc.reshape(-1, 2), axis=1))))
    assert seen >= 12, f"board seen in only {seen}/24 frames"
    assert np.mean(errs) < 1.5, f"corner reprojection too large: {np.mean(errs):.2f} px"


def test_rasterizer_paints_labels_and_zbuffers(k):
    """A single box in front of the camera fills pixels, carries its object/part id, and the nearer
    of two overlapping faces wins the z-buffer."""
    verts, faces = M.box(np.array([0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0]))
    near = M.Mesh("near", verts, faces, "matte_gray", part="near")
    far = M.Mesh("far", verts + np.array([0, 0, 300.0]), faces, "chrome", part="far")
    scene = M.Scene([near, far], M.board_texture(px_per_mm=2.0),
                    lights=[(np.array([0.0, -300.0, 300.0]), 0.8)])
    eye = np.array([0.0, -400.0, 0.0])
    w2c = M.look_at_world_to_camera(eye, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    out = M.render(scene, w2c, k, eye)
    assert (out.obj_id == 0).sum() > 500, "near box did not paint"
    # The near box must occlude the far one: no 'far' pixels where the near box is.
    near_mask = out.obj_id == 0
    far_mask = out.obj_id == 1
    assert not np.any(near_mask & far_mask)
    # Depth under the near box is ~ the near face distance (eye at y=-400, box face at y=-50).
    zs = out.depth[near_mask]
    assert 300 < np.median(zs) < 400


def test_object_and_part_masks_are_written_shapes(k):
    """object/part id buffers have the same HxW as the frame and label the painted object."""
    scene, meta = M.build_gso_primitive("thin_tool")
    cams = M.scene_camera_path(meta["kind"], 4)
    out = M.render(scene, cams[1][1], k, cams[1][0])
    assert out.obj_id.shape == (M.RENDER_H, M.RENDER_W)
    assert out.part_id.shape == (M.RENDER_H, M.RENDER_W)
    assert set(np.unique(out.obj_id)) - {-1}, "no object painted"


def test_known_dimensions_present():
    for name in ("under_sink", "knob_panel", "gso_mug"):
        _scene, meta = M.SCENES[name]()
        assert meta["dimensions_mm"], f"{name} has no key dimensions"
        assert all(v > 0 for v in meta["dimensions_mm"].values() if isinstance(v, (int, float)))


def test_cylinder_and_tube_radii():
    """A cylinder's and a tube's vertices sit at the requested radius (mesh geometry is exact)."""
    v, _f = M.cylinder(np.array([10.0, 20.0]), 0.0, 50.0, 8.0, segments=64)
    r = np.hypot(v[:64, 0] - 10.0, v[:64, 1] - 20.0)
    assert np.allclose(r, 8.0, atol=1e-6)
    tv, _tf = M.tube(np.array([0.0, 0.0]), 0.0, 10.0, 12.0, 8.0, segments=64)
    r_out = np.hypot(tv[:64, 0], tv[:64, 1])
    assert np.allclose(r_out, 12.0, atol=1e-6)


def test_portrait_video_decodes_upright(tmp_path):
    """The stored portrait file has landscape samples + a 90deg display matrix, and the REAL
    extractor decodes it upright. Encodes a few portrait frames directly (no 20 s loop)."""
    # A few upright portrait frames (tall): a bright board-ish pattern so extraction keeps them.
    rng = np.random.default_rng(0)
    frames = [(rng.integers(0, 255, (1280, 720, 3))).astype(np.uint8) for _ in range(60)]
    video = tmp_path / "p.mp4"
    M._encode(frames, video, portrait=True)
    chk = M._verify_portrait_decode(video, frames[0])
    assert chk["decoded_upright"] is True
    assert chk["probe_rotation_handled"] is True
    assert chk["decoded_frame_wh"][1] > chk["decoded_frame_wh"][0]  # portrait display
