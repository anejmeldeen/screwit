"""Support-plane levelling and table removal (recon.scale.level_to_plane / support_band_mask).

Two real-scan failures these fix, generically and without any object category (SPEC 3.1,
AGENTS.md "the demo object can be anything"):

* "slanted, floating in the air" -- the board rotation was not trusted, so the scan was never
  rotated to put the support surface at Z-up/z=0. ``level_to_plane`` turns a fitted support plane
  into the one rigid correction that lands it flat on z = 0.
* "recorded at an elevation, the round table came out on top of the object" -- the support surface
  was kept as part of the object. ``support_band_mask`` marks everything on or below the support
  plane for removal, whatever shape it is.
"""

from __future__ import annotations

import numpy as np

from screwit.recon import scale as sc


def _tilted_plane(tilt_deg: float, height_mm: float, axis=(1.0, 0.0, 0.0)):
    """A plane whose unit normal is ``tilt_deg`` off +Z, passing ``height_mm`` from the origin along
    that normal. Returns (normal, d) for ``normal . x + d = 0``, normal pointing to the object side."""
    a = np.radians(tilt_deg)
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    n = np.cos(a) * np.array([0.0, 0.0, 1.0]) + np.sin(a) * axis
    n = n / np.linalg.norm(n)
    p0 = height_mm * n
    return n, float(-n @ p0)


def test_level_to_plane_makes_plane_z_up_at_zero():
    n, d = _tilted_plane(22.0, 40.0, axis=(0.6, 0.8, 0.0))
    r, t = sc.level_to_plane(n, d)
    # proper rotation, no mirroring
    assert abs(np.linalg.det(r) - 1.0) < 1e-6
    # the normal maps to +Z
    assert np.allclose(r @ n, [0, 0, 1], atol=1e-6)
    # a point on the plane lands on z = 0, and a point 50 mm above lands at z = 50
    p_on = -d * n                      # closest plane point to origin: (n.x + d = 0)
    p_up = p_on + 50.0 * n
    assert abs((r @ p_on + t)[2]) < 1e-6
    assert abs((r @ p_up + t)[2] - 50.0) < 1e-6


def test_level_to_plane_handles_antiparallel_normal():
    r, t = sc.level_to_plane(np.array([0.0, 0.0, -1.0]), 5.0)
    assert abs(np.linalg.det(r) - 1.0) < 1e-6
    assert np.allclose(r @ np.array([0.0, 0.0, -1.0]), [0, 0, 1], atol=1e-6)


def test_support_band_drops_surface_keeps_object():
    rng = np.random.default_rng(0)
    # a flat round table on z = 0 and a box object sitting on it (z in [0, 60])
    r = np.sqrt(rng.random(4000)) * 150.0
    th = rng.random(4000) * 2 * np.pi
    table = np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 0.8, 4000)], axis=1)
    obj = rng.uniform([-20, -20, 0.0], [20, 20, 60.0], size=(2000, 3))
    pts = np.concatenate([table, obj])

    drop = sc.support_band_mask(pts, np.array([0.0, 0.0, 1.0]), 0.0, band_mm=5.0)
    kept = pts[~drop]
    # almost all of the table is gone, almost all of the object survives
    assert drop[:4000].mean() > 0.95
    assert (~drop[4000:]).mean() > 0.9
    # what is left reaches up to the object height and no longer spreads to the table radius
    assert kept[:, 2].max() > 55.0
    assert np.linalg.norm(kept[:, :2], axis=1).max() < 60.0


def test_level_then_strip_table_on_elevated_tilted_scene():
    """End to end on a synthetic point cloud: a tilted table with a box object. After levelling,
    the object stands upright and the table (now on z ~= 0) is stripped by the band mask -- so the
    object no longer floats and the table is not part of it."""
    rng = np.random.default_rng(1)
    n, d = _tilted_plane(18.0, 25.0, axis=(1.0, 0.0, 0.0))
    # build an upright table+object in the plane's own frame, then rotate INTO the tilted scene
    r_up, t_up = sc.level_to_plane(n, d)       # scene -> upright
    r_dn, t_dn = r_up.T, -r_up.T @ t_up        # upright -> scene
    rr = np.sqrt(rng.random(5000)) * 140.0
    th = rng.random(5000) * 2 * np.pi
    table_u = np.stack([rr * np.cos(th), rr * np.sin(th), rng.normal(0, 0.7, 5000)], axis=1)
    obj_u = rng.uniform([-18, -18, 0.0], [18, 18, 70.0], size=(2500, 3))
    scene = (np.concatenate([table_u, obj_u]) @ r_dn.T) + t_dn

    # recover the plane from the scene points (what depthclean.support_plane would give) and level
    leveled = (scene @ r_up.T) + t_up
    assert abs(leveled[:5000, 2].mean()) < 2.0        # table sits at z ~= 0 after levelling
    drop = sc.support_band_mask(leveled, np.array([0.0, 0.0, 1.0]), 0.0, band_mm=5.0)
    obj_only = leveled[~drop]
    assert obj_only[:, 2].min() > -3.0                 # object no longer dips below the surface
    assert obj_only[:, 2].max() > 60.0                 # full object height kept
    assert np.linalg.norm(obj_only[:, :2], axis=1).max() < 60.0   # table radius gone


def test_strip_support_surface_removes_table_keeps_object_base():
    """A table disc far wider than the object, with a box standing on it, after levelling. The
    footprint-aware strip removes the table ring but keeps the object including its base on the
    surface (so a known object still measures its true height)."""
    from screwit.recon.pipeline import ReconPipelineImpl

    p = ReconPipelineImpl.__new__(ReconPipelineImpl)
    p.support_band_mm = 5.0
    rng = np.random.default_rng(0)
    obj = rng.uniform([-20, -20, 0.0], [20, 20, 100.0], size=(3000, 3))
    rr = np.sqrt(rng.random(6000)) * 150.0
    th = rng.random(6000) * 2 * np.pi
    table = np.stack([rr * np.cos(th), rr * np.sin(th), rng.normal(0, 0.6, 6000)], axis=1)
    pts = np.concatenate([obj, table])

    kept = p._strip_support_surface(pts, (np.array([0.0, 0.0, 1.0]), 0.0))
    # table (radius 150) is gone; object footprint (~28 mm) survives with a little slack
    assert np.linalg.norm(kept[:, :2], axis=1).max() < 60.0
    assert kept[:, 2].max() > 98.0                 # full object height kept
    assert (kept[:, 2] < 5.0).sum() > 100          # the object base on the surface is kept


def test_strip_support_surface_noop_without_object_core():
    """If nothing rises clearly above the support band (no solid body), nothing is stripped."""
    from screwit.recon.pipeline import ReconPipelineImpl

    p = ReconPipelineImpl.__new__(ReconPipelineImpl)
    p.support_band_mm = 5.0
    rng = np.random.default_rng(1)
    flat = np.stack([rng.uniform(-100, 100, 4000), rng.uniform(-100, 100, 4000),
                     rng.normal(0, 0.5, 4000)], axis=1)
    kept = p._strip_support_surface(flat, (np.array([0.0, 0.0, 1.0]), 0.0))
    assert len(kept) == len(flat)
