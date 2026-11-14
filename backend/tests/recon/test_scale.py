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
