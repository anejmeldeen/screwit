"""Model-free main-object mask from depth and cameras (the CPU live path).

SPEC 2.2 asks the user to put the object on a flat surface with the board lying flat beside it, so
the scene has a dominant support plane with the object standing on it. This masker uses only that:

1. Back-project every view's depth into the scan frame and find the support plane with RANSAC. A
   candidate plane is accepted only if almost nothing lies below it and the cameras are above it
   (a face of the object is not a support).
2. Points higher than the plane noise are object candidates (the board and the table are not).
3. Cluster the candidates. The cameras orbit the object, so keep the clusters close to the point
   the cameras look at, and drop clutter at the edges of the frames.
4. A pixel is in the mask if its 3D point is near a kept cluster.

Nothing here knows what the object is. When the scene has no usable support plane (for example a
synthetic test frame with depth only on the object) the masker returns "all valid depth" and says so
in ``last_info``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from screwit.interfaces import FrameSet


@dataclass
class MaskInfo:
    method: str
    plane_threshold_mm: float | None = None
    clusters_kept: int = 0
    clusters_total: int = 0
    mask_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)


def _backproject(depth: np.ndarray, k: np.ndarray, w2c: np.ndarray, stride: int = 1):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs].astype(np.float64)
    valid = z > 0
    ys, xs, z = ys[valid], xs[valid], z[valid]
    x = (xs - k[0, 2]) / k[0, 0] * z
    y = (ys - k[1, 2]) / k[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    r, t = w2c[:3, :3], w2c[:3, 3]
    world = (cam - t) @ r  # x = R^T (p_cam - t)
    return world, ys, xs


def _camera_centres(cams) -> np.ndarray:
    out = []
    for c in cams:
        m = np.asarray(c.world_to_camera, dtype=np.float64).reshape(4, 4)
        out.append(-m[:3, :3].T @ m[:3, 3])
    return np.asarray(out)


def _look_at_point(cams) -> np.ndarray:
    """Least-squares point closest to every camera's optical axis."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        m = np.asarray(c.world_to_camera, dtype=np.float64).reshape(4, 4)
        centre = -m[:3, :3].T @ m[:3, 3]
        d = m[2, :3]  # camera +Z (forward) in world = third row of R
        d = d / np.linalg.norm(d)
        p = np.eye(3) - np.outer(d, d)
        a += p
        b += p @ centre
    return np.linalg.lstsq(a, b, rcond=None)[0]


class GeometricObjectMasker:
    """Support-plane + clustering object mask (see module docstring)."""

    def __init__(self, sample_points: int = 300_000, max_planes: int = 4) -> None:
        self.sample_points = sample_points
        self.max_planes = max_planes
        self.last_info = MaskInfo(method="none")

    def masks(self, frames: FrameSet) -> list[np.ndarray]:
        import open3d as o3d
        from scipy.spatial import cKDTree

        cams = frames.cameras.frames
        all_pts = []
        for depth, cam in zip(frames.depth_mm, cams, strict=True):
            k = np.asarray(cam.K, dtype=np.float64).reshape(3, 3)
            w2c = np.asarray(cam.world_to_camera, dtype=np.float64).reshape(4, 4)
            pts, _, _ = _backproject(depth, k, w2c, stride=2)
            all_pts.append(pts)
        pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
        fallback = [d > 0 for d in frames.depth_mm]
        if len(pts) < 1000:
            self.last_info = MaskInfo(method="all_valid_depth", notes=["too few depth points"])
            return fallback
        rng = np.random.default_rng(0)
        if len(pts) > self.sample_points:
            pts = pts[rng.choice(len(pts), self.sample_points, replace=False)]

        centres = _camera_centres(cams)
        look = _look_at_point(cams)
        cam_dist = float(np.median(np.linalg.norm(centres - look, axis=1)))
        # Plane noise grows with distance; 0.5% of the viewing distance, at least 2 mm.
        thr = max(2.0, 0.005 * cam_dist)

        self._rejected: list[str] = []
        plane = self._support_plane(o3d, pts, centres, thr)
        if plane is None:
            self.last_info = MaskInfo(method="all_valid_depth", plane_threshold_mm=thr,
                                      notes=["no support plane with the cameras above it",
                                             *self._rejected])
            return fallback
        n, d = plane

        height = pts @ n + d
        cand = pts[height > 3.0 * thr]
        if len(cand) < 200:
            self.last_info = MaskInfo(method="all_valid_depth", plane_threshold_mm=thr,
                                      notes=["nothing stands on the support plane"])
            return fallback

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand))
        # Neighbourhood radius: above the plane noise and the back-projection spacing (stride-2
        # pixels at the viewing distance), so one object's points join into one cluster.
        fx = float(np.median([np.asarray(c.K, dtype=np.float64).reshape(3, 3)[0, 0] for c in cams]))
        spacing = 2.0 * cam_dist / max(fx, 1e-6)
        eps = max(4.0 * thr, 0.02 * cam_dist, 3.0 * spacing)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=20))
        ids = [i for i in np.unique(labels) if i >= 0]
        if not ids:
            self.last_info = MaskInfo(method="all_valid_depth", plane_threshold_mm=thr,
                                      notes=["no point clusters above the plane"])
            return fallback
        # Distance from the look-at point to each cluster, measured in the plane.
        look_in_plane = look - (look @ n + d) * n
        reach = 0.35 * cam_dist
        kept = []
        dists = {}
        for i in ids:
            c = cand[labels == i]
            flat = c - np.outer(c @ n + d, n)
            dists[i] = float(np.min(np.linalg.norm(flat - look_in_plane, axis=1)))
            if dists[i] <= reach:
                kept.append(i)
        if not kept:
            kept = [min(dists, key=dists.get)]
        obj = cand[np.isin(labels, kept)]
        tree = cKDTree(obj)
        # The plane band (within 3x noise of the table) was excluded to drop the table, which also
        # cuts the bottom few millimetres off the object. Put back band pixels that sit directly
        # under the object's lowest points (its base outline), within one sample spacing.
        u_axis = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(u_axis) < 1e-6:
            u_axis = np.cross(n, [0.0, 1.0, 0.0])
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(n, u_axis)
        obj_h = obj @ n + d
        base = obj[obj_h <= np.percentile(obj_h, 10)]
        base_tree = cKDTree(np.stack([base @ u_axis, base @ v_axis], axis=1)) if len(base) else None

        out = []
        total = 0
        for depth, cam in zip(frames.depth_mm, cams, strict=True):
            k = np.asarray(cam.K, dtype=np.float64).reshape(3, 3)
            w2c = np.asarray(cam.world_to_camera, dtype=np.float64).reshape(4, 4)
            p, ys, xs = _backproject(depth, k, w2c, stride=1)
            mask = np.zeros(depth.shape, dtype=bool)
            if len(p):
                h_all = p @ n + d
                above = h_all > 1.5 * thr
                near, _ = tree.query(p[above], k=1, distance_upper_bound=eps)
                hit = np.isfinite(near)
                sel = np.flatnonzero(above)[hit]
                mask[ys[sel], xs[sel]] = True
                band = np.flatnonzero((h_all > -thr) & ~above)
                if base_tree is not None and len(band):
                    q = p[band]
                    dist, _ = base_tree.query(np.stack([q @ u_axis, q @ v_axis], axis=1), k=1,
                                              distance_upper_bound=spacing)
                    keep = band[np.isfinite(dist)]
                    mask[ys[keep], xs[keep]] = True
            out.append(mask)
            total += int(mask.sum())
        frac = total / max(1, sum(m.size for m in out))
        self.last_info = MaskInfo(method="support_plane_clusters", plane_threshold_mm=thr,
                                  clusters_kept=len(kept), clusters_total=len(ids),
                                  mask_fraction=frac)
        return out

    def _support_plane(self, o3d, pts: np.ndarray, centres: np.ndarray, thr: float):
        """(unit normal pointing at the cameras, d) of the first RANSAC plane that supports the
        scene, or None."""
        rest = pts
        for _ in range(self.max_planes):
            if len(rest) < 500:
                return None
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rest))
            (a, b, c, dd), inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=3,
                                                      num_iterations=1000)
            n = np.array([a, b, c], dtype=np.float64)
            norm = np.linalg.norm(n)
            n, dd = n / norm, dd / norm
            if np.median(centres @ n + dd) < 0:
                n, dd = -n, -dd
            cams_above = float(np.mean(centres @ n + dd > 0))
            below = float(np.mean(pts @ n + dd < -3.0 * thr))
            # Allow some points below (a floor seen past the table edge), not a large share.
            if cams_above >= 0.9 and below <= 0.15:
                return n, dd
            self._rejected.append(f"plane {len(inliers) / len(pts):.0%} of points: "
                                  f"{cams_above:.0%} cameras above, {below:.0%} points below")
            mask = np.ones(len(rest), dtype=bool)
            mask[np.asarray(inliers, dtype=int)] = False
            rest = rest[mask]
        return None
