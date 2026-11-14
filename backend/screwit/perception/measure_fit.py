"""Geometry fitting and measurement (SPEC 6.2 step 4).

All inputs and outputs are scan-frame millimetres (SPEC 3.1). The heavy libraries (Open3D,
pyransac3d) are imported lazily so importing this module never drags them in; the pure-numpy
helpers (axis-aligned extents, least-squares cylinder refinement) work with numpy alone and are
what the unit tests exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------- OBB

@dataclass(frozen=True)
class ObbFit:
    center_mm: Vec3
    extent_mm: Vec3
    rotation_quat: tuple[float, float, float, float]  # [x, y, z, w]


def _rotmat_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Row-major rotation matrix -> unit quaternion [x, y, z, w]."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0)
    q = q / n
    if q[3] < 0:  # keep w >= 0 for a canonical sign
        q = -q
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _upright_box(o3d, pts: np.ndarray):
    """Minimum-area rectangle of the XY footprint (rotating calipers over the convex hull) times
    the Z extent, as an Open3D OrientedBoundingBox."""
    from scipy.spatial import ConvexHull

    xy = pts[:, :2]
    hull = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        norm = np.linalg.norm(e)
        if norm < 1e-9:
            continue
        u = e / norm
        v = np.array([-u[1], u[0]])
        pu, pv = xy @ u, xy @ v
        area = np.ptp(pu) * np.ptp(pv)
        if best is None or area < best[0]:
            best = (area, u, v, pu, pv)
    z = pts[:, 2]
    if best is None:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return o3d.geometry.OrientedBoundingBox((lo + hi) / 2, np.eye(3), np.maximum(hi - lo, 1e-9))
    _, u, v, pu, pv = best
    cu, cv = (pu.min() + pu.max()) / 2, (pv.min() + pv.max()) / 2
    centre = np.array([*(cu * u + cv * v), (z.min() + z.max()) / 2])
    rot = np.column_stack([[u[0], u[1], 0.0], [v[0], v[1], 0.0], [0.0, 0.0, 1.0]])
    extent = np.array([np.ptp(pu), np.ptp(pv), np.ptp(z)])
    return o3d.geometry.OrientedBoundingBox(centre, rot, np.maximum(extent, 1e-9))


def robust_extents(points_mm: np.ndarray, obb: ObbFit, lo_pct: float = 1.0,
                   hi_pct: float = 99.0) -> tuple[float, float, float]:
    """Face-to-face size along the box axes from the 1st-99th percentiles of the points.

    A TSDF mesh bulges by a few millimetres along silhouette edges, which sets the max extent; the
    faces themselves hold many vertices, so the percentiles sit on the faces. On the CPU scan of a
    100 mm cube the max extents were 107/106/96 mm and the mid-height face-to-face widths 101-103.
    """
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    x, y, z, w = obb.rotation_quat
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    local = (pts - np.asarray(obb.center_mm)) @ rot
    ext = np.percentile(local, hi_pct, axis=0) - np.percentile(local, lo_pct, axis=0)
    return tuple(float(e) for e in ext)


def oriented_bounding_box(points_mm: np.ndarray) -> ObbFit:
    """Tight oriented bounding box of a point set: the smaller of Open3D's minimal box and the
    upright (Z = support normal) minimum-footprint box, after trimming stray points.

    Falls back to an axis-aligned box when Open3D is unavailable or degenerate, which keeps the
    CPU-fake and tests independent of a working Open3D build.
    """
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    if len(pts) < 4:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        c = (lo + hi) / 2.0
        return ObbFit(tuple(c), tuple(np.maximum(hi - lo, 0.0)), (0.0, 0.0, 0.0, 1.0))
    try:
        import open3d as o3d

        # Trim the 0.5% most distant points: TSDF meshes from real depth carry a few stray vertices
        # that would otherwise set the box size.
        centre = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - centre, axis=1)
        core = pts[dist <= np.percentile(dist, 99.5)] if len(pts) > 200 else pts
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(core))
        # PCA boxes are arbitrary for symmetric shapes (a cube's PCA axes are degenerate), so take
        # the smaller of the minimal-volume box and the upright box (Z is the support normal).
        candidates = []
        try:
            candidates.append(pc.get_minimal_oriented_bounding_box(robust=True))
        except Exception:  # noqa: BLE001 - older Open3D or degenerate input
            candidates.append(pc.get_oriented_bounding_box())
        candidates.append(_upright_box(o3d, core))
        obb = min(candidates, key=lambda b: float(np.prod(np.asarray(b.extent))))
        R = np.asarray(obb.R)
        if np.linalg.det(R) < 0:  # reflections are not rotations
            R = R @ np.diag([1.0, 1.0, -1.0])
        return ObbFit(tuple(np.asarray(obb.center)), tuple(np.asarray(obb.extent)),
                      _rotmat_to_quat_xyzw(R))
    except Exception:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        c = (lo + hi) / 2.0
        return ObbFit(tuple(c), tuple(np.maximum(hi - lo, 0.0)), (0.0, 0.0, 0.0, 1.0))


# --------------------------------------------------------------------------- cylinder

@dataclass(frozen=True)
class CylinderFit:
    axis_point_mm: Vec3
    axis_dir: Vec3        # unit
    radius_mm: float
    length_mm: float
    fit_rmse_mm: float
    inlier_fraction: float


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(a, ref))
    v = np.cross(a, u)
    return u, v


def _cylinder_residuals(params: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance of each point to the cylinder surface, axis parameterised by two angles."""
    px, py, pz, theta, phi, r = params
    axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    axis = _unit(axis)
    base = np.array([px, py, pz])
    d = pts - base
    proj = d - np.outer(d @ axis, axis)
    return np.linalg.norm(proj, axis=1) - r


def fit_cylinder(points_mm: np.ndarray, *, thresh_mm: float = 1.0,
                 max_iters: int = 300) -> CylinderFit:
    """Fit a cylinder: pyransac3d for a robust seed, then a scipy least-squares refinement.

    pyransac3d's cylinder fit is noisy on its own (SPEC 6.2 step 4), so the RANSAC result only
    seeds the least-squares refine; when pyransac3d is unavailable the seed comes from a PCA of
    the points, which the refinement then tightens.
    """
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    seed_axis = None
    seed_point = centroid
    seed_r = None
    n_inliers = len(pts)
    try:
        import warnings

        import pyransac3d as pyrsc

        cyl = pyrsc.Cylinder()
        with warnings.catch_warnings():
            # pyransac3d's own cylinder fit divides by zero on degenerate RANSAC samples; the
            # least-squares refinement below recovers, so silence its internal numeric warnings.
            warnings.simplefilter("ignore", RuntimeWarning)
            center, axis, radius, inliers = cyl.fit(pts, thresh=thresh_mm, maxIteration=max_iters)
        seed_axis = _unit(np.asarray(axis, dtype=float))
        seed_point = np.asarray(center, dtype=float)
        seed_r = float(radius)
        n_inliers = len(inliers) if inliers is not None else len(pts)
    except Exception:
        pass
    if seed_axis is None:
        # PCA seed: the long principal axis is the cylinder axis.
        d = pts - centroid
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        seed_axis = _unit(vt[0])
        u, v = _perp_basis(seed_axis)
        radial = np.hypot(d @ u, d @ v)
        seed_r = float(np.median(radial))

    theta = np.arccos(np.clip(seed_axis[2], -1.0, 1.0))
    phi = np.arctan2(seed_axis[1], seed_axis[0])
    p0 = np.array([seed_point[0], seed_point[1], seed_point[2], theta, phi, seed_r])
    sol = least_squares(_cylinder_residuals, p0, args=(pts,), method="lm", max_nfev=2000)
    px, py, pz, theta, phi, r = sol.x
    axis = _unit(np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]))
    base = np.array([px, py, pz])

    resid = _cylinder_residuals(sol.x, pts)
    rmse = float(np.sqrt(np.mean(resid**2)))
    inlier_fraction = float(np.mean(np.abs(resid) <= thresh_mm))
    if n_inliers != len(pts):
        inlier_fraction = max(inlier_fraction, n_inliers / len(pts))

    # Length and a stable axis point: project points onto the axis through the centroid.
    t = (pts - centroid) @ axis
    length = float(t.max() - t.min())
    axis_point = centroid + ((pts - base) @ axis).mean() * axis
    if axis[2] < 0:  # canonical direction, mostly +Z for vertical plumbing
        axis = -axis
    return CylinderFit(tuple(axis_point), tuple(axis), abs(float(r)), length, rmse, inlier_fraction)


# --------------------------------------------------------------------------- plane

@dataclass(frozen=True)
class PlaneFit:
    point_mm: Vec3
    normal: Vec3          # unit
    extent_u_mm: float
    extent_v_mm: float
    fit_rmse_mm: float
    inlier_fraction: float


def fit_plane(points_mm: np.ndarray, *, thresh_mm: float = 1.0) -> PlaneFit:
    """Fit a plane (Open3D ``segment_plane``), falling back to a numpy SVD plane fit."""
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    normal = None
    inlier_fraction: float | None = None
    try:
        import open3d as o3d

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        model, inliers = pc.segment_plane(distance_threshold=thresh_mm, ransac_n=3,
                                          num_iterations=200)
        normal = _unit(np.asarray(model[:3], dtype=float))
        inlier_fraction = len(inliers) / len(pts) if len(pts) else 1.0
    except Exception:
        d = pts - centroid
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        normal = _unit(vt[-1])  # smallest-variance direction is the plane normal
    d = pts - centroid
    dist = d @ normal
    rmse = float(np.sqrt(np.mean(dist**2)))
    if inlier_fraction is None:
        inlier_fraction = float(np.mean(np.abs(dist) <= thresh_mm))
    u, v = _perp_basis(normal)
    eu = float((d @ u).max() - (d @ u).min())
    ev = float((d @ v).max() - (d @ v).min())
    return PlaneFit(tuple(centroid), tuple(normal), eu, ev, rmse, inlier_fraction)
