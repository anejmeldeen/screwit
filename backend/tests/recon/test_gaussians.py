"""Gaussian transforms, PLY round-trip, and the SPZ golden verified with an INDEPENDENT decoder."""

from __future__ import annotations

import numpy as np
import pytest

from screwit.recon import gaussians as G

from ._spz_decoder import decode, ensure_decoder


def _cloud(quats_wxyz, means=None, log_scales=None):
    n = quats_wxyz.shape[0]
    return G.GaussianCloud(
        means=means if means is not None else np.zeros((n, 3), np.float32),
        log_scales=log_scales if log_scales is not None else np.full((n, 3), np.log(2.0), np.float32),
        quats_wxyz=quats_wxyz.astype(np.float32),
        opacities=np.full((n,), 2.0, np.float32),
        f_dc=np.zeros((n, 3), np.float32),
    )


def test_ply_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    cloud = G.GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32) * 50,
        log_scales=rng.normal(size=(n, 3)).astype(np.float32),
        quats_wxyz=q.astype(np.float32),
        opacities=rng.normal(size=n).astype(np.float32),
        f_dc=rng.normal(size=(n, 3)).astype(np.float32),
    )
    path = tmp_path / "scene.ply"
    G.write_ply(cloud, path)
    back = G.read_gaussians_ply(path)
    assert np.allclose(back.means, cloud.means, atol=1e-4)
    assert np.allclose(back.log_scales, cloud.log_scales, atol=1e-4)
    assert np.allclose(back.f_dc, cloud.f_dc, atol=1e-4)
    # quaternion equal up to sign and normalisation
    assert np.allclose(np.abs((back.quats_wxyz * cloud.quats_wxyz).sum(1)), 1.0, atol=1e-3)


def test_scale_factor_correction():
    cloud = _cloud(np.array([[1, 0, 0, 0]], np.float32),
                   means=np.array([[10.0, 0, 0]], np.float32),
                   log_scales=np.array([[0.0, 0.0, 0.0]], np.float32))
    out = G.apply_scale_factor(cloud, 1.7)
    assert np.allclose(out.means, [[17.0, 0, 0]])
    assert np.allclose(out.log_scales, np.log(1.7))


def test_similarity_moves_rotates_and_scales():
    cloud = _cloud(np.array([[1, 0, 0, 0]], np.float32), means=np.array([[1.0, 0, 0]], np.float32))
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # +90 deg about Z
    out = G.apply_similarity(cloud, r, np.array([5.0, 0, 0]), 2.0)
    assert np.allclose(out.means[0], [5.0, 2.0, 0.0], atol=1e-5)
    assert np.allclose(out.log_scales, np.log(2.0) + np.log(2.0))


def test_crop_prune_downsample():
    rng = np.random.default_rng(1)
    means = rng.uniform(-200, 200, size=(5000, 3)).astype(np.float32)
    cloud = _cloud(np.tile([1, 0, 0, 0], (5000, 1)).astype(np.float32), means=means)
    cropped = G.crop_to_bbox(cloud, np.array([-50, -50, -50.0]), np.array([50, 50, 50.0]), 10.0)
    assert np.all(np.abs(cropped.means) <= 60 + 1e-5)
    down = G.voxel_downsample(cloud, max_count=200)
    assert len(down) <= 200


def test_prune_opacity_drops_transparent():
    cloud = G.GaussianCloud(
        means=np.zeros((3, 3), np.float32),
        log_scales=np.zeros((3, 3), np.float32),
        quats_wxyz=np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
        opacities=np.array([-5.0, 0.0, 5.0], np.float32),  # sigmoids ~0.007, 0.5, 0.993
        f_dc=np.zeros((3, 3), np.float32),
    )
    kept = G.prune_opacity(cloud, min_opacity=0.1)
    assert len(kept) == 2  # drops only the -5 logit


@pytest.mark.skipif(ensure_decoder() is None, reason="no C++ spz decoder (no compiler)")
def test_spz_golden_known_rotation(tmp_path):
    """Golden: one elongated gaussian with a known rotation round-trips with the correct
    orientation, verified by Niantic spz's own loadSpz (an INDEPENDENT decoder)."""
    ang = np.deg2rad(40.0)
    qw, qz = np.cos(ang / 2), np.sin(ang / 2)
    expected_xyzw = np.array([0.0, 0.0, qz, qw])
    cloud = G.GaussianCloud(
        means=np.array([[12.3, -4.5, 100.0]], np.float32),
        log_scales=np.array([[np.log(25.0), np.log(2.0), np.log(2.0)]], np.float32),
        quats_wxyz=np.array([[qw, 0, 0, qz]], np.float32),
        opacities=np.array([2.0], np.float32),
        f_dc=np.array([[1.0, -0.5, 0.2]], np.float32),
    )
    path = tmp_path / "scene.spz"
    manifest = G.write_spz(cloud, path)
    assert manifest["spz_version"] == 4
    assert manifest["pack_from"] == "UNSPECIFIED"
    assert manifest["spark_min_version"] == "2.4.0"

    dec = decode(path, 1)
    assert dec["num_points"] == 1
    assert dec["sh_degree"] == 0
    g = dec["gaussians"][0]
    # Position within the 1/2^12 (~0.25 mm) fixed-point resolution.
    assert np.allclose(g["pos"], [12.3, -4.5, 100.0], atol=0.02)
    # Orientation preserved (quaternion up to sign) -- the point of the golden.
    dot = abs(float(np.dot(g["rot_xyzw"], expected_xyzw)))
    assert dot > 0.999, (g["rot_xyzw"], expected_xyzw, dot)
    # The elongated axis survives: one log scale near ln(25), the other two near ln(2).
    assert max(g["scale"]) > 3.0 and sorted(g["scale"])[0] < 1.0


@pytest.mark.skipif(ensure_decoder() is None, reason="no C++ spz decoder (no compiler)")
def test_spz_positions_many(tmp_path):
    rng = np.random.default_rng(3)
    n = 2000
    means = rng.uniform(-150, 150, size=(n, 3)).astype(np.float32)
    cloud = _cloud(np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32), means=means)
    path = tmp_path / "m.spz"
    G.write_spz(cloud, path)
    dec = decode(path, 1)
    assert dec["num_points"] == n
    assert np.allclose(dec["gaussians"][0]["pos"], means[0], atol=0.02)
