"""Small geometry helpers for CAD placement and mate checks (scan-frame millimetres).

Quaternions are [x, y, z, w] as in the contracts (SPEC 3.1). Rotations use right-handed, Z-up frames.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def quat_to_matrix(q: Quat) -> np.ndarray:
    """3x3 rotation matrix from a unit quaternion [x, y, z, w]."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def transform_point(q: Quat, t: Vec3, p: Vec3) -> np.ndarray:
    """Apply rotation then translation to a point: R p + t."""
    return quat_to_matrix(q) @ np.asarray(p, dtype=float) + np.asarray(t, dtype=float)


def transform_direction(q: Quat, d: Vec3) -> np.ndarray:
    """Rotate a direction (no translation), returned as a unit vector."""
    v = quat_to_matrix(q) @ np.asarray(d, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def unit(v: np.ndarray | Vec3) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def angle_between_deg(a: np.ndarray | Vec3, b: np.ndarray | Vec3) -> float:
    """Unsigned angle between two directions, in degrees (0..180)."""
    ua, ub = unit(a), unit(b)
    c = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return math.degrees(math.acos(c))


def coaxial_offset_mm(axis_point_a: np.ndarray, axis_dir_a: np.ndarray,
                      axis_point_b: Vec3, axis_dir_b: Vec3) -> float:
    """Perpendicular distance between two axis lines where they are closest.

    For (near-)parallel axes this is the lateral offset between them; the shortest distance between the
    two infinite lines is a robust single number for how far off-centre the bore sits from the shaft.
    """
    pa = np.asarray(axis_point_a, dtype=float)
    da = unit(axis_dir_a)
    pb = np.asarray(axis_point_b, dtype=float)
    db = unit(axis_dir_b)
    cross = np.cross(da, db)
    ncross = np.linalg.norm(cross)
    w = pa - pb
    if ncross < 1e-9:
        # Parallel: distance from pa to the line through pb.
        return float(np.linalg.norm(w - np.dot(w, db) * db))
    return float(abs(np.dot(w, cross)) / ncross)
