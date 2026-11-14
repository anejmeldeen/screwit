"""Primitive fitting and measurement (SPEC 6.2 step 4)."""

from __future__ import annotations

import numpy as np
import pytest

from screwit.perception import measure_fit


def _cylinder_points(radius, length, axis=(0, 0, 1), center=(0, 0, 0), n=4000, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(-length / 2, length / 2, n)
    pts = (radius * np.cos(theta)[:, None] * u + radius * np.sin(theta)[:, None] * v
           + h[:, None] * axis)
    return pts + np.asarray(center, float)


def test_cylinder_recovers_radius_within_1mm_at_half_mm_noise():
    # SPEC: noisy cylinders recover radius within 1 mm at 0.5 mm noise.
    rng = np.random.default_rng(42)
    pts = _cylinder_points(19.05, 120.0, center=(0, 0, 250))
    pts = pts + rng.normal(0, 0.5, pts.shape)
    fit = measure_fit.fit_cylinder(pts, thresh_mm=1.5)
    assert abs(fit.radius_mm - 19.05) < 1.0
    assert abs(abs(fit.axis_dir[2]) - 1.0) < 0.05  # axis ~ Z
    assert fit.length_mm == pytest.approx(120.0, abs=5.0)


def test_cylinder_axis_point_near_truth():
    pts = _cylinder_points(10.0, 60.0, center=(5, -3, 40))
    fit = measure_fit.fit_cylinder(pts, thresh_mm=1.0)
    ap = np.asarray(fit.axis_point_mm)
    # The recovered axis point lies on the true axis (x=5, y=-3, any z).
    assert abs(ap[0] - 5) < 1.0 and abs(ap[1] + 3) < 1.0


def test_plane_fit_normal_and_extent():
    rng = np.random.default_rng(1)
    xs = rng.uniform(-60, 60, 3000)
    ys = rng.uniform(-50, 50, 3000)
    pts = np.stack([xs, ys, np.full_like(xs, 12.0)], axis=1)
    pts += rng.normal(0, 0.2, pts.shape)
    pf = measure_fit.fit_plane(pts, thresh_mm=1.0)
    assert abs(abs(pf.normal[2]) - 1.0) < 0.05  # normal ~ Z
    # The in-plane u/v basis is arbitrary, so compare the unordered pair of extents.
    assert sorted((pf.extent_u_mm, pf.extent_v_mm)) == pytest.approx([100, 120], abs=5)


def test_obb_extent_matches_box():
    # A dense regular grid on a box so the OBB extents are well determined.
    g = np.linspace(-1, 1, 20)
    gx, gy, gz = np.meshgrid(g, g, g)
    pts = np.stack([gx.ravel() * 30, gy.ravel() * 10, gz.ravel() * 5], axis=1)
    obb = measure_fit.oriented_bounding_box(pts)
    ext = sorted(obb.extent_mm)
    assert ext[0] == pytest.approx(10, abs=2)
    assert ext[1] == pytest.approx(20, abs=2)
    assert ext[2] == pytest.approx(60, abs=2)
    q = np.asarray(obb.rotation_quat)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6
