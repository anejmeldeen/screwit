"""Scale stage: Umeyama, multi-view triangulation, board scale recovery, tilt flag, fallbacks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from screwit.recon import scale as S
from screwit.recon.geometry import FakeGeometryModel


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3)) * 10
    from scipy.spatial.transform import Rotation as Rot

    r_true = Rot.random(random_state=1).as_matrix()
    s_true, t_true = 2.5, np.array([4.0, -1.0, 3.0])
    dst = s_true * (src @ r_true.T) + t_true
    r, t, s = S.umeyama(src, dst)
    assert np.isclose(s, s_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)
    assert np.allclose(r, r_true, atol=1e-6)


def test_triangulate_point_dlt():
    # Two cameras looking at a known point.
    from screwit.recon.geometry import look_at

    k = np.array([[400.0, 0, 160], [0, 400, 160], [0, 0, 1]])
    pt = np.array([5.0, -3.0, 50.0])
    projs, pixels = [], []
    for eye in ([0, 0, -200.0], [60, 20, -180.0]):
        w2c = look_at(np.array(eye), pt, np.array([0, 1.0, 0]))
        p = k @ w2c[:3, :4]
        uvw = p @ np.append(pt, 1.0)
        projs.append(p)
        pixels.append(uvw[:2] / uvw[2])
    rec = S.triangulate_point(projs, pixels)
    assert np.allclose(rec, pt, atol=1e-3)


def test_rescale_factor_from_two_points():
    a, b = (0.0, 0.0, 0.0), (0.0, 0.0, 50.0)
    s = S.rescale_factor_from_two_points(a, b, 85.6)
    assert np.isclose(s, 85.6 / 50.0)


def _board_scale(fake):
    td = Path(tempfile.mkdtemp())
    idx = fake.render_frames(None, None, td / "frames")
    res = fake.run(td / "frames" / "model", [e.frame for e in idx.frames])
    by = {e.frame: e for e in idx.frames}
    per_view, grid = {}, {}
    for vi, name in enumerate(res.frames):
        e = by[name]
        grid[vi] = e.grid
        img = np.asarray(Image.open(td / "frames" / e.frame).convert("RGB"))
        ids, corners = S.detect_board_corners(img)
        if ids is not None:
            per_view[vi] = {int(c): corners[k] for k, c in enumerate(ids)}
    return S.scale_from_board(res.intrinsics, res.extrinsics, per_view, grid), res


def test_board_scale_recovery_and_z_up():
    fake = FakeGeometryModel(n_views=12, model_scale=0.5)
    sr, res = _board_scale(fake)
    assert sr.info.method == "aruco"
    assert sr.info.residual_mm < 2.0
    assert sr.frame_origin == "board"
    assert not sr.flags  # clean scan
    # scale recovered (model_scale 0.5 => model->scan scale 2.0).
    assert abs(sr.s - 2.0) < 0.05
    # Cameras land above the board (+Z), proving the planar sign disambiguation.
    from screwit.recon.geometry import apply_similarity_to_cameras

    extr_scan = apply_similarity_to_cameras(res.extrinsics, sr.r, sr.t, sr.s)
    centres = np.array([-extr_scan[i, :3, :3].T @ extr_scan[i, :3, 3]
                        for i in range(extr_scan.shape[0])])
    assert centres[:, 2].mean() > 0
    # The recovered model->scan similarity is the true one: a proper rotation, not a mirror image
    # (the fake textures the board unmirrored, origin bottom-left, as a real print lies).
    r_true, t_true, s_true = fake.true_similarity_model_to_scan()
    assert np.linalg.det(sr.r) > 0
    np.testing.assert_allclose(sr.r, r_true, atol=2e-3)
    np.testing.assert_allclose(sr.t, t_true, atol=1.5)
    assert abs(sr.s - s_true) < 0.01 * s_true


def test_board_scan_points_are_right_handed_from_the_printed_origin():
    """Corner coordinates follow the printed arrow: origin bottom-left, X right, Y up the page."""
    pts = S.board_scan_points()
    obj = S.board_object_points()
    # Corner 0 is top-left on the sheet (OpenCV), so it has the largest scan-frame Y.
    assert pts[0, 1] == pts[:, 1].max()
    assert pts[0, 0] == obj[0, 0]
    # Moving right on the sheet is +X; moving down the sheet is -Y.
    assert pts[1, 0] > pts[0, 0] and pts[1, 1] == pts[0, 1]
    assert pts[4, 1] < pts[0, 1]


def test_scale_from_learned_fallback():
    sr = S.scale_from_learned(1.07)
    assert sr.info.method == "learned"
    assert sr.frame_origin == "object_bottom_center"
    assert sr.s == 1.07
    assert "learned_scale_approximate" in sr.flags


def test_support_plane_normal_points_up():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-50, 50, 500), rng.uniform(-50, 50, 500),
                           rng.normal(0, 0.5, 500)])  # flat-ish around Z=0
    pts = np.vstack([pts, [[0, 0, 40.0]]])  # one point above
    n, d = S.support_plane(pts)
    assert abs(abs(n[2]) - 1.0) < 0.1
    assert n[2] > 0  # oriented up


def _look_at(eye_mm, target_mm=(75.0, 105.0, 0.0)):
    """World->camera (OpenCV axes) for a camera at eye looking at target, scan frame in mm."""
    eye, target = np.asarray(eye_mm, float), np.asarray(target_mm, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    return r, -r @ eye


def test_repose_from_board_recovers_each_view_when_model_poses_disagree():
    """The model's poses are garbage and its depth has a different scale in every view; posing each
    view from its board corners recovers the true cameras in mm and depth in metres."""
    from screwit.recon.geometry import GaussianCloud, GeometryResult

    w, h, sx, sy = 280, 504, 0.25, 0.25          # model grid; full res is 1120 x 2016
    k = np.array([[400.0, 0, w / 2], [0, 400.0, h / 2], [0, 0, 1]])
    k_full = np.array([[1600.0, 0, (w / 2 + 0.5) / sx - 0.5], [0, 1600.0, (h / 2 + 0.5) / sy - 0.5], [0, 0, 1]])
    pts = S.board_scan_points()
    eyes = [(-250.0, -300.0, 450.0), (350.0, -250.0, 400.0), (300.0, 450.0, 500.0), (-300.0, 400.0, 420.0)]
    units_per_mm = [1 / 900.0, 1 / 1200.0, 1 / 1000.0, 1 / 700.0]   # each view's depth scale differs
    per_view, grids, depth, truth = {}, {}, [], []
    rng = np.random.default_rng(1)
    for vi, eye in enumerate(eyes):
        r, t = _look_at(eye)
        truth.append(np.asarray(eye))
        cam = pts @ r.T + t
        uv = cam[:, :2] / cam[:, 2:] @ k_full[:2, :2].T + k_full[:2, 2]
        per_view[vi] = {cid: uv[cid] for cid in range(len(pts))}
        grids[vi] = {"scale_x": sx, "scale_y": sy}
        d = np.zeros((h, w), np.float32)
        for (u, v), z in zip(uv, cam[:, 2], strict=True):
            x, y = int(round(u * sx)), int(round(v * sy))
            d[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = z * units_per_mm[vi]
        depth.append(d)
    bogus = np.tile(np.eye(4), (4, 1, 1))
    bogus[:, :3, 3] = rng.normal(size=(4, 3))                   # poses that do not agree at all
    empty = GaussianCloud(means=np.zeros((0, 3), np.float32), log_scales=np.zeros((0, 3), np.float32),
                          quats_wxyz=np.zeros((0, 4), np.float32), opacities=np.zeros(0, np.float32),
                          f_dc=np.zeros((0, 3), np.float32))
    result = GeometryResult(frames=[f"{i:04d}.jpg" for i in range(4)], width=w, height=h,
                            intrinsics=np.tile(k, (4, 1, 1)), extrinsics=bogus, depth=depth,
                            confidence=[np.ones((h, w), np.float32)] * 4, gaussians=empty)
    new, res = S.repose_from_board(result, per_view, grids)
    assert new.frames == result.frames
    assert res.s == 1000.0 and res.frame_origin == "board" and res.info.residual_mm < 2.0
    for vi in range(4):
        e = new.extrinsics[vi]
        centre_mm = -e[:3, :3].T @ e[:3, 3] * res.s
        assert np.allclose(centre_mm, truth[vi], atol=3.0), (vi, centre_mm, truth[vi])
        x, y = int(round(per_view[vi][0][0] * sx)), int(round(per_view[vi][0][1] * sy))
        r, t = _look_at(eyes[vi])
        assert np.isclose(new.depth[vi][y, x] * res.s, (pts[0] @ r.T + t)[2], rtol=0.01)


def test_repose_from_board_needs_two_posed_views():
    from screwit.recon.geometry import GeometryResult

    result = GeometryResult(frames=["a.jpg"], width=4, height=4, intrinsics=np.eye(3)[None],
                            extrinsics=np.eye(4)[None], depth=[np.zeros((4, 4), np.float32)],
                            confidence=[np.zeros((4, 4), np.float32)], gaussians=None)
    assert S.repose_from_board(result, {0: {0: np.zeros(2), 1: np.ones(2)}}, {0: {"scale_x": 1, "scale_y": 1}}) is None


def test_board_batches_only_when_the_board_is_seen_in_many_frames(monkeypatch):
    from screwit.recon.pipeline import BOARD_BATCH_VIEWS, ReconPipelineImpl

    pick = ReconPipelineImpl._board_batch_views
    names = [f"{i:04d}.jpg" for i in range(56)]
    capped = names[::7]                                        # one model call holds 8 views
    seen = {n: (20 if i % 4 else 2) for i, n in enumerate(names)}   # board clear in 42 frames
    views = pick(names, capped, seen)
    assert len(views) == BOARD_BATCH_VIEWS and all(seen[v] >= S.MIN_PNP_CORNERS for v in views)
    assert views == sorted(views)                               # spread in clip order
    assert pick(names, capped, {n: 2 for n in names}) == []     # board barely seen: run once
    assert pick(names, names[:30], seen) == []                  # the model already holds enough
    monkeypatch.setenv("SCREWIT_BOARD_BATCHES", "0")
    assert pick(names, capped, seen) == []
