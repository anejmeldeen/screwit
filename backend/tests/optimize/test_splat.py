"""X5 splat optimizer tests: gaussian cap, importance + stratification, spz round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from screwit.optimize import make_mobile_variants
from screwit.optimize.splat import (
    MAX_MOBILE_GAUSSIANS,
    LocalGaussianIO,
    MobileGaussians,
    importance,
    optimize_splat,
    read_ply_local,
    select_gaussians,
    write_ply_local,
)


def _random_cloud(n: int, seed: int = 0, extent_mm: float = 100.0) -> MobileGaussians:
    rng = np.random.default_rng(seed)
    means = rng.uniform(-extent_mm, extent_mm, size=(n, 3)).astype(np.float32)
    log_scales = rng.uniform(-4.0, 0.0, size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opac = rng.uniform(-3.0, 3.0, size=n).astype(np.float32)
    f_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    return MobileGaussians(means=means, log_scales=log_scales, quats_wxyz=quats,
                           opacities=opac, f_dc=f_dc)


def test_gaussian_cap_respected():
    cloud = _random_cloud(1000)
    reduced = select_gaussians(cloud, max_count=400)
    assert len(reduced) <= 400
    assert len(reduced) == 400


def test_cap_400k_on_large_cloud():
    cloud = _random_cloud(450_000, seed=7)
    reduced = select_gaussians(cloud, max_count=MAX_MOBILE_GAUSSIANS)
    assert len(reduced) <= MAX_MOBILE_GAUSSIANS


def test_small_cloud_unchanged():
    cloud = _random_cloud(50)
    reduced = select_gaussians(cloud, max_count=400)
    assert len(reduced) == 50


def test_stratification_covers_space_evenly():
    """A cloud with 90% of its gaussians crammed into one corner should still, after stratified
    selection, keep representatives from the sparse region -- a pure top-importance pick would not."""
    rng = np.random.default_rng(1)
    dense = rng.uniform(-100, -90, size=(9000, 3))          # tight corner
    sparse = rng.uniform(-100, 100, size=(1000, 3))          # spread across the volume
    means = np.vstack([dense, sparse]).astype(np.float32)
    # Give the dense corner the highest opacity so importance alone would favour it.
    opac = np.concatenate([np.full(9000, 3.0), np.full(1000, -1.0)]).astype(np.float32)
    n = means.shape[0]
    cloud = MobileGaussians(
        means=means,
        log_scales=np.zeros((n, 3), np.float32),
        quats_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=opac,
        f_dc=np.zeros((n, 3), np.float32),
    )
    reduced = select_gaussians(cloud, max_count=500)
    kept = reduced.means
    # How many kept gaussians lie outside the dense corner?
    outside_corner = np.any(kept > -90, axis=1)
    assert outside_corner.sum() >= 50, "stratified selection dropped the sparse region entirely"


def test_importance_is_opacity_times_volume():
    cloud = MobileGaussians(
        means=np.zeros((2, 3), np.float32),
        log_scales=np.array([[0, 0, 0], [1, 1, 1]], np.float32),
        quats_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (2, 1)),
        opacities=np.array([0.0, 0.0], np.float32),  # sigmoid = 0.5 for both
        f_dc=np.zeros((2, 3), np.float32),
    )
    imp = importance(cloud)
    # Same opacity, larger volume -> larger importance; ratio == exp(3).
    assert imp[1] > imp[0]
    assert np.isclose(imp[1] / imp[0], np.exp(3.0), rtol=1e-6)


def test_ply_roundtrip_and_spz_written(tmp_path):
    cloud = _random_cloud(2000, seed=3)
    ply = tmp_path / "scene.ply"
    write_ply_local(cloud, ply)
    back = read_ply_local(ply)
    assert len(back) == len(cloud)
    assert np.allclose(back.means, cloud.means, atol=1e-5)
    assert np.allclose(back.log_scales, cloud.log_scales, atol=1e-5)

    out = tmp_path / "scene.spz"
    stats = optimize_splat(ply, out, LocalGaussianIO(), max_count=500)
    assert stats.in_gaussians == 2000
    assert stats.out_gaussians <= 500
    assert out.exists() and stats.out_bytes > 0

    # Validate the spz header (magic "NGSP", version 4, matching count).
    raw = out.read_bytes()
    import struct

    magic, version, count = struct.unpack("<III", raw[:12])
    assert magic == 0x5053474E
    assert version == 4
    assert count == stats.out_gaussians


def test_make_mobile_variants_with_splat(tmp_path):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=4, radius=40.0)
    pos = np.asarray(m.vertices)
    span = np.ptp(pos, axis=0) + 1e-9
    rgba = np.c_[((pos - pos.min(0)) / span * 255).astype(np.uint8),
                 np.full((len(pos), 1), 255, np.uint8)]
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
    mesh_path = tmp_path / "object_full.glb"
    m.export(str(mesh_path), file_type="glb")

    cloud = _random_cloud(3000, seed=5)
    splat_path = tmp_path / "scene.ply"
    write_ply_local(cloud, splat_path)

    try:
        from screwit.optimize.mesh import gltfpack_binary

        gltfpack_binary()
    except Exception:
        pytest.skip("gltfpack not installed")

    variants = make_mobile_variants(mesh_path, splat_path, tmp_path / "mobile",
                                    gaussian_io=LocalGaussianIO())
    assert variants.splat_mobile_path is not None
    assert variants.splat_mobile_path.name == "scene.spz"
    assert variants.splat_stats.out_gaussians <= MAX_MOBILE_GAUSSIANS
    assert variants.mesh_mobile_path.exists()


def test_splat_without_io_raises(tmp_path):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    mesh_path = tmp_path / "object_full.glb"
    m.export(str(mesh_path), file_type="glb")
    try:
        from screwit.optimize.mesh import gltfpack_binary

        gltfpack_binary()
    except Exception:
        pytest.skip("gltfpack not installed")
    with pytest.raises(ValueError, match="no gaussian_io"):
        make_mobile_variants(mesh_path, tmp_path / "scene.ply", tmp_path / "mobile")
