"""Geometry helpers: extrinsic padding, view capping, camera similarity, and the runtime
gaussian-to-depth alignment for both 'quirk present' and 'quirk absent' (CPU findings 7)."""

from __future__ import annotations

import numpy as np

from screwit.recon import geometry as GE


def test_pad_extrinsics_3x4_to_4x4():
    extr = np.zeros((2, 3, 4))
    extr[:, :3, :3] = np.eye(3)
    extr[:, :3, 3] = [1, 2, 3]
    padded = GE._pad_extrinsics(extr)
    assert padded.shape == (2, 4, 4)
    assert np.allclose(padded[:, 3, :], [0, 0, 0, 1])
    # already 4x4 is passed through
    assert GE._pad_extrinsics(padded).shape == (2, 4, 4)


def test_cap_views_only_on_cpu():
    names = [f"{i:04d}.jpg" for i in range(40)]
    capped = GE.cap_views(names, device="cpu")
    assert len(capped) == GE.CPU_VIEW_CAP
    assert capped[0] == names[0] and capped[-1] == names[-1]
    # On a GPU device nothing is dropped.
    assert GE.cap_views(names, device="cuda") == names


def test_camera_similarity_keeps_rotation_orthonormal():
    from scipy.spatial.transform import Rotation as Rot

    extr = np.eye(4)[None].copy()
    extr[0, :3, :3] = Rot.random(random_state=5).as_matrix()
    extr[0, :3, 3] = [10.0, -4.0, 7.0]
    r = Rot.random(random_state=6).as_matrix()
    out = GE.apply_similarity_to_cameras(extr, r, np.array([3.0, 1.0, -2.0]), 2.5)
    rn = out[0, :3, :3]
    assert np.allclose(rn @ rn.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rn), 1.0, atol=1e-9)


def _run_fake(quirk, n_views=10):
    import tempfile
    from pathlib import Path

    fake = GE.FakeGeometryModel(n_views=n_views, gaussian_quirk=quirk)
    td = Path(tempfile.mkdtemp())
    idx = fake.render_frames(None, None, td / "frames")
    res = fake.run(td / "frames" / "model", [e.frame for e in idx.frames])
    return fake, res, td, idx


def test_align_gaussians_quirk_absent():
    _, res, _, _ = _run_fake(quirk=1.0)
    aligned, al = GE.align_gaussians_to_depth(res)
    assert al.applied
    assert abs(al.scale - 1.0) < 0.02
    assert al.rmse_mm < 0.5
    assert len(aligned) == len(res.gaussians)


def test_align_gaussians_quirk_present():
    _, res, _, _ = _run_fake(quirk=1.7)
    aligned, al = GE.align_gaussians_to_depth(res)
    assert al.applied
    # Recovers the inverse of the injected offset (gaussian frame was 1.7x the depth frame).
    assert abs(al.scale - 1.0 / 1.7) < 0.02
    assert al.rmse_mm < 0.5


def test_align_skips_when_not_pixel_aligned():
    """If the gaussian count does not match N*H*W (e.g. a pruned PLY), alignment is skipped."""
    _, res, _, _ = _run_fake(quirk=1.0)
    res.gaussians = res.gaussians.select(np.arange(10))  # break the slab structure
    aligned, al = GE.align_gaussians_to_depth(res)
    assert not al.applied
    assert al.scale == 1.0
    assert len(aligned) == 10


def test_to_cameras_contract_row_major_opencv():
    _, res, _, _ = _run_fake(quirk=1.0)
    cams = GE.to_cameras("scn_x", 1, res)
    assert cams.scan_id == "scn_x"
    assert len(cams.frames) == len(res.frames)
    cf = cams.frames[0]
    assert cf.width == res.width and cf.height == res.height
    assert len(cf.world_to_camera) == 4 and len(cf.world_to_camera[0]) == 4
    assert len(cf.K) == 3


def test_fake_depth_is_model_units_not_mm():
    """The nested/model depth is in model units; for the fake that is scan-mm * model_scale."""
    fake, res, _, _ = _run_fake(quirk=1.0)
    # model_scale defaults to 0.5, so model-unit depths are about half the scan distances (~200mm).
    mean_depth = np.mean([d[d > 0].mean() for d in res.depth])
    assert 100 < mean_depth < 350  # not the ~400mm scan distance


def test_gaussians_from_prediction_uses_real_da3_layout():
    """Shapes as measured on DA3NESTED-GIANT-LARGE-1.1 (CPU, 2 views): means/scales (1,N,3),
    rotations (1,N,4) wxyz, opacities (1,N) probabilities, harmonics (1,N,3,9) channel-major.
    The DC colour is coefficient 0 of each channel."""
    from types import SimpleNamespace

    n = 5
    rng = np.random.default_rng(0)
    harm = rng.normal(size=(1, n, 3, 9)).astype(np.float32)
    g = SimpleNamespace(
        means=rng.normal(size=(1, n, 3)).astype(np.float32),
        scales=np.full((1, n, 3), 0.01, np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], np.float32), (1, n, 1)),
        opacities=np.full((1, n), 0.5, np.float32),
        harmonics=harm,
    )
    cloud = GE._gaussians_from_prediction(SimpleNamespace(gaussians=g))
    assert len(cloud) == n
    np.testing.assert_allclose(cloud.f_dc, harm[0, :, :, 0])
    np.testing.assert_allclose(cloud.log_scales, np.log(0.01), rtol=1e-5)
    np.testing.assert_allclose(cloud.opacities, 0.0, atol=1e-6)  # logit(0.5)
    np.testing.assert_allclose(cloud.quats_wxyz[:, 0], 1.0)
