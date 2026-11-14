"""Derive the scan-frame placement of a mating template from the mated feature (SPEC 3.1/6.4).

The CAD service never trusts a requested placement for a mating part: it builds the template in its
local frame (bore axis +Z opening on the -Z face, D-bore flat inward normal -Y, see
cad/templates/base.py) and computes the rigid transform that seats the bore on the shaft:

  1. rotate the template bore axis (+Z) onto the feature's axis direction,
  2. roll about that axis so the bore's flat inward normal matches the shaft's flat_normal,
  3. translate so the bore opening sits `engagement` mm back from the shaft's free end.

The result is a unit quaternion [x, y, z, w] and a translation in scan-frame millimetres. The service
applies this to the solid before export (STL/STEP are written already placed, so the viewer overlays
them directly) and reports it in CadResult.placement.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

_LOCAL_AXIS = np.array([0.0, 0.0, 1.0])       # template bore axis (into the knob from the opening)
_LOCAL_FLAT_INWARD = np.array([0.0, -1.0, 0.0])  # template D-bore flat inward normal


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix taking unit vector a onto unit vector b."""
    a, b = _unit(a), _unit(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        # 180 deg: rotate about any axis perpendicular to a.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, perp))
        return _axis_angle_matrix(axis, math.pi)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _matrix_to_quat(m: np.ndarray) -> Quat:
    """Rotation matrix -> unit quaternion [x, y, z, w]."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def derive_mate_placement(*, feature_axis_point: Vec3, feature_axis_dir: Vec3,
                          shaft_free_end_mm: Vec3, engagement_mm: float,
                          has_flat: bool, feature_flat_normal: Vec3 | None,
                          local_flat_inward: Vec3 = (0.0, -1.0, 0.0)) -> tuple[Quat, Vec3, np.ndarray]:
    """Return (quat [x,y,z,w], translate_mm, rotation_matrix) seating the bore on the shaft.

    `shaft_free_end_mm` is the scan-frame point of the shaft tip the knob slides onto; the bore opening
    is placed `engagement_mm` back from it along the axis (toward the shaft base).
    """
    axis_dir = _unit(np.asarray(feature_axis_dir, dtype=float))

    # 1. Rotate template bore axis (+Z) onto the feature axis.
    r1 = _rotation_between(_LOCAL_AXIS, axis_dir)

    # 2. Roll about the axis so the flat inward normal lines up (only matters for a D-bore).
    r = r1
    if has_flat and feature_flat_normal is not None:
        local_flat = np.asarray(local_flat_inward, dtype=float)
        placed_flat = r1 @ local_flat
        target_flat = _unit(np.asarray(feature_flat_normal, dtype=float))
        # Project both onto the plane perpendicular to the axis, then find the roll angle.
        placed_perp = _unit(placed_flat - np.dot(placed_flat, axis_dir) * axis_dir)
        target_perp = _unit(target_flat - np.dot(target_flat, axis_dir) * axis_dir)
        if np.linalg.norm(placed_perp) > 1e-6 and np.linalg.norm(target_perp) > 1e-6:
            cos = float(np.clip(np.dot(placed_perp, target_perp), -1.0, 1.0))
            cross = np.dot(np.cross(placed_perp, target_perp), axis_dir)
            angle = math.atan2(cross, cos)
            r = _axis_angle_matrix(axis_dir, angle) @ r1

    # 3. Translate so the bore opening (local origin) sits engagement back from the free end.
    tip = np.asarray(shaft_free_end_mm, dtype=float)
    opening = tip - engagement_mm * axis_dir  # move from the tip toward the shaft base
    translate = opening  # local origin maps to the opening point

    quat = _matrix_to_quat(r)
    return quat, (float(translate[0]), float(translate[1]), float(translate[2])), r


def shaft_free_end(feature_axis_point: Vec3, feature_axis_dir: Vec3, part_mesh,
                   fallback_length_mm: float) -> Vec3:
    """Find the shaft's free end: the extreme point of the part mesh along +axis_dir from the axis
    point. Falls back to axis_point + length*axis_dir when no mesh is available."""
    axis_point = np.asarray(feature_axis_point, dtype=float)
    axis_dir = _unit(np.asarray(feature_axis_dir, dtype=float))
    if part_mesh is not None and len(getattr(part_mesh, "vertices", [])) > 0:
        v = np.asarray(part_mesh.vertices, dtype=float)
        t = (v - axis_point) @ axis_dir
        t_max = float(t.max())
        end = axis_point + t_max * axis_dir
        return (float(end[0]), float(end[1]), float(end[2]))
    end = axis_point + fallback_length_mm * axis_dir
    return (float(end[0]), float(end[1]), float(end[2]))
