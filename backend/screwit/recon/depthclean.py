"""Per-view depth cleanup before masking and fusion (SPEC 6.1; P2).

Feed-forward depth on the CPU path is noisy in ways TSDF fusion cannot average away:

1. Flying pixels at silhouettes: a pixel on an object edge gets a depth interpolated between the
   object and whatever is behind it. Fused, they form a skirt around the object and spikes on its
   rims, and the skirt welds the object to the table.
2. Disagreement between views: on the headphones clip two views disagree about the same surface by
   0.8% of depth at the median (about 6 mm at 0.7 m) and 1.7% at p75, more than the 4 mm truncation
   band, so fusion stacks the disagreeing surfaces into bumps and holes.

``clean`` removes (1) with a depth-discontinuity test, removes points that other views look straight
through (free-space violations, the signature of floaters) with a multi-view consistency test, and
reduces the remaining per-pixel noise with an edge-preserving bilateral filter that never mixes
depths across a real jump. Removed pixels get depth 0, which every consumer already treats as "no
measurement". Nothing here is object specific: every threshold is relative to depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CleanInfo:
    flying_fraction: float = 0.0
    inconsistent_fraction: float = 0.0
    flattened: bool = False

    def note(self) -> str:
        flat = "views flattened to the support plane, " if self.flattened else ""
        return (f"depth cleanup: {flat}{self.flying_fraction:.1%} edge pixels, "
                f"{self.inconsistent_fraction:.1%} multi-view outliers removed")


def _fill_from_nearest(depth: np.ndarray) -> np.ndarray:
    """Copy of ``depth`` with every missing pixel (0) set to its nearest valid pixel's depth, so a
    filter never mixes a real depth with a placeholder value."""
    from scipy.ndimage import distance_transform_edt

    valid = depth > 0
    if valid.all() or not valid.any():
        return depth.astype(np.float32)
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return depth[iy, ix].astype(np.float32)


def smooth(depth: np.ndarray, sigma_space_px: float = 2.0, sigma_rel: float = 0.008) -> np.ndarray:
    """Edge-preserving bilateral smoothing of one depth map (mm). The range sigma is relative to the
    median depth, so depths are only averaged with neighbours within about one noise level; a real
    step (an object edge, tens of mm) is left sharp. Missing pixels (0) stay missing and never pull
    their neighbours (they are filled from the nearest valid pixel for the filter only)."""
    import cv2

    valid = depth > 0
    if not valid.any():
        return depth
    med = float(np.median(depth[valid]))
    fill = _fill_from_nearest(depth)
    k = int(2 * round(2.5 * sigma_space_px) + 1)
    out = cv2.bilateralFilter(fill, k, sigmaColor=sigma_rel * med, sigmaSpace=sigma_space_px)
    return np.where(valid, out, 0.0).astype(depth.dtype)


def flying_pixels(depth: np.ndarray, rel_jump: float = 0.03, dilate_px: int = 0) -> np.ndarray:
    """Bool mask of pixels next to a depth discontinuity (a jump larger than ``rel_jump`` of the
    depth to any 4-neighbour), grown by ``dilate_px``. These are the least trustworthy pixels: their
    depth is a blend of the foreground and the background. 3% with no growth: 2% grown by a pixel
    also took the steep lower walls an orbit sees at grazing angles (see-through 16% -> 20%)."""
    from scipy.ndimage import binary_dilation

    z = depth.astype(np.float64)
    zp = np.pad(z, 1, mode="edge")
    nb = np.stack([zp[1:-1, 2:], zp[1:-1, :-2], zp[2:, 1:-1], zp[:-2, 1:-1]])
    nb_valid = nb > 0
    jump = np.where(nb_valid, np.abs(nb - z), 0.0).max(axis=0)
    bad = (z > 0) & (jump > rel_jump * np.maximum(z, 1e-6))
    if dilate_px > 0 and bad.any():
        bad = binary_dilation(bad, iterations=dilate_px) & (z > 0)
    return bad


def _backproject(depth, k, w2c):
    ys, xs = np.nonzero(depth > 0)
    z = depth[ys, xs].astype(np.float64)
    cam = np.stack([(xs - k[0, 2]) / k[0, 0] * z, (ys - k[1, 2]) / k[1, 1] * z, z], axis=1)
    r, t = w2c[:3, :3], w2c[:3, 3]
    return (cam - t) @ r, ys, xs


def support_plane(depths, intrinsics, extrinsics, *, max_points: int = 200_000,
                  rel_thr: float = 0.01):
    """(unit normal pointing at the cameras, d) of the scene's support surface, or None.

    RANSAC over every view's back-projected points; a plane is accepted only when at least 90% of
    the cameras are above it and at most 15% of the points are below it (a face of the object, a
    wall or a ceiling is not a support). Deterministic (seeded).
    """
    import open3d as o3d

    pts = [_backproject(d, np.asarray(k), np.asarray(e))[0] for d, k, e in
           zip(depths, intrinsics, extrinsics, strict=True)]
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(pts) < 2000:
        return None
    rng = np.random.default_rng(0)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    centres = np.array([-np.asarray(e)[:3, :3].T @ np.asarray(e)[:3, 3] for e in extrinsics])
    look = np.median(pts, axis=0)
    thr = rel_thr * float(np.median(np.linalg.norm(centres - look, axis=1)))
    try:
        o3d.utility.random.seed(0)
    except Exception:  # noqa: BLE001
        pass
    rest = pts
    for _ in range(3):
        if len(rest) < 1000:
            return None
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rest))
        (a, b, c, dd), inl = pcd.segment_plane(distance_threshold=thr, ransac_n=3,
                                              num_iterations=1000)
        n = np.array([a, b, c], float)
        s = float(np.linalg.norm(n)) or 1.0
        n, dd = n / s, dd / s
        if np.median(centres @ n + dd) < 0:
            n, dd = -n, -dd
        if np.mean(centres @ n + dd > 0) >= 0.9 and np.mean(pts @ n + dd < -3 * thr) <= 0.15:
            # Refine by least squares on the inliers (RANSAC's plane is only as good as its 3 points).
            inl_pts = rest[np.asarray(inl, int)]
            c = inl_pts.mean(axis=0)
            _, _, vt = np.linalg.svd(inl_pts - c, full_matrices=False)
            n2 = vt[-1] if float(vt[-1] @ n) >= 0 else -vt[-1]
            return n2, float(-n2 @ c)
        keep = np.ones(len(rest), bool)
        keep[np.asarray(inl, int)] = False
        rest = rest[keep]
    return None


def flatten_to_plane(depths, intrinsics, extrinsics, plane, *, band_rel: float = 0.03,
                     max_corr: float = 0.03) -> list[np.ndarray]:
    """Per-view smooth depth correction that puts each view's own view of the support surface onto
    the consensus plane.

    A monocular-style depth map carries a smooth, view-dependent error: one view's table is tilted or
    bowed by a few mm relative to another's (measured on a real clip: 0.4-2 mm residual against a
    per-view quadratic, 0.5-10 mm against the shared plane). The same error field sits on the
    object, which is why the views disagree about its surface. For each view, fit a quadratic
    multiplicative correction ``1 + c(x, y)`` that maps its near-plane pixels exactly onto the plane
    (robust, iteratively re-weighted), and apply it to the whole map. The object pixels inherit the
    correction from the table around them. Views with too little visible support surface, or a fit
    that wants more than ``max_corr`` (3%) anywhere, are left unchanged.
    """
    n, d = plane
    out = []
    for depth, k, e in zip(depths, intrinsics, extrinsics, strict=True):
        depth = np.asarray(depth)
        k = np.asarray(k, float)
        e = np.asarray(e, float)
        r, t = e[:3, :3], e[:3, 3]
        centre = -r.T @ t
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w]
        ray = np.stack([(xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones((h, w))],
                       axis=-1) @ r                    # world direction per unit camera depth
        rn = ray @ n
        c0 = float(centre @ n + d)                    # camera height above the plane
        with np.errstate(divide="ignore", invalid="ignore"):
            z_plane = -c0 / rn
        height = c0 + depth * rn                       # pixel's point height above the plane
        ok = (depth > 0) & np.isfinite(z_plane) & (z_plane > 0)
        near = ok & (np.abs(height) < band_rel * np.maximum(depth, 1e-6))
        # Only flatten against a real, large support surface: at least a fifth of the frame must lie
        # near the plane (a table filling the view), or the fit is dominated by the object's own
        # lower walls and would distort the object. A view that is already flat (within 0.1% of
        # depth) is left alone.
        if near.sum() < max(2000, 0.2 * h * w):
            out.append(depth.copy())
            continue
        xn = xs / w - 0.5
        yn = ys / h - 0.5
        basis = np.stack([np.ones((h, w)), xn, yn, xn * xn, xn * yn, yn * yn], axis=-1)
        a = basis[near]
        target = z_plane[near] / depth[near] - 1.0
        before = 1.4826 * float(np.median(np.abs(target - np.median(target))))
        if before < 1e-3:
            out.append(depth.copy())
            continue
        keep = np.ones(len(target), bool)
        coef = np.zeros(6)
        mad = before
        for _ in range(4):
            coef = np.linalg.lstsq(a[keep], target[keep], rcond=None)[0]
            res = target - a @ coef
            mad = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep]))))
            keep = np.abs(res) < 3.0 * max(mad, 2e-4)
        corr = basis @ coef
        # The correction must explain the error (residual noise drops clearly) and stay small.
        if mad > 0.7 * before or float(np.abs(corr[depth > 0]).max(initial=0.0)) > max_corr:
            out.append(depth.copy())
            continue
        out.append(np.where(depth > 0, depth * (1.0 + corr), 0.0).astype(depth.dtype))
    return out


def multiview_outliers(depths: list[np.ndarray], intrinsics, extrinsics, tau: float = 0.02,
                       min_votes: int = 3) -> list[np.ndarray]:
    """Bool mask per view of pixels that the other views contradict.

    Each pixel's 3D point is projected into every other view and compared with that view's depth:
    agreement within ``tau`` (relative) is a supporting vote; the other view seeing FARTHER than the
    point means it looked through empty space where the point should be (a free-space violation);
    seeing nearer is an occlusion and carries no information. A pixel is an outlier when at least
    ``min_votes`` views report a violation and violations outnumber supports. Legit surface points
    with ordinary noise keep a majority of supports; floaters do not.
    """
    n = len(depths)
    out = []
    for i in range(n):
        h, w = depths[i].shape
        bad = np.zeros((h, w), dtype=bool)
        if n < 3:
            out.append(bad)
            continue
        X, ys, xs = _backproject(depths[i], np.asarray(intrinsics[i]), np.asarray(extrinsics[i]))
        if len(X) == 0:
            out.append(bad)
            continue
        support = np.zeros(len(X), np.int32)
        violate = np.zeros(len(X), np.int32)
        for j in range(n):
            if j == i:
                continue
            kj = np.asarray(intrinsics[j])
            ej = np.asarray(extrinsics[j])
            hj, wj = depths[j].shape
            pc = X @ ej[:3, :3].T + ej[:3, 3]
            zj = pc[:, 2]
            front = zj > 1e-6
            u = np.full(len(X), -1)
            v = np.full(len(X), -1)
            u[front] = np.round(kj[0, 0] * pc[front, 0] / zj[front] + kj[0, 2]).astype(int)
            v[front] = np.round(kj[1, 1] * pc[front, 1] / zj[front] + kj[1, 2]).astype(int)
            inb = front & (u >= 0) & (u < wj) & (v >= 0) & (v < hj)
            dj = np.zeros(len(X))
            dj[inb] = depths[j][v[inb], u[inb]]
            seen = inb & (dj > 0)
            rel = np.zeros(len(X))
            rel[seen] = (dj[seen] - zj[seen]) / zj[seen]
            support += (seen & (np.abs(rel) <= tau)).astype(np.int32)
            violate += (seen & (rel > tau)).astype(np.int32)
        outlier = (violate >= min_votes) & (violate > support)
        bad[ys[outlier], xs[outlier]] = True
        out.append(bad)
    return out


def clean(depths: list[np.ndarray], intrinsics, extrinsics, *, smooth_depth: bool = True,
          flatten: bool = True, info: CleanInfo | None = None) -> list[np.ndarray]:
    """Return cleaned copies of ``depths`` (mm). Steps, in order:

    1. Flatten each view onto the support plane (when there is one), which removes most of the
       smooth per-view depth error before anything compares views.
    2. Drop flying pixels, found on the unsmoothed depth (smoothing would soften the blend pixels
       into plausible values).
    3. Drop multi-view outliers, judged against all of the other views' depth (so the table and
       background count as evidence of free space).
    4. Bilateral-smooth what survived.
    """
    depths = [np.asarray(d) for d in depths]
    if flatten:
        plane = support_plane(depths, intrinsics, extrinsics)
        if plane is not None:
            depths = flatten_to_plane(depths, intrinsics, extrinsics, plane)
            if info is not None:
                info.flattened = True
    total = sum(int((d > 0).sum()) for d in depths) or 1
    fly = [flying_pixels(d) for d in depths]
    mv = multiview_outliers(depths, intrinsics, extrinsics)
    out = []
    n_fly = n_mv = 0
    for d, f, m in zip(depths, fly, mv, strict=True):
        n_fly += int(f.sum())
        n_mv += int((m & ~f).sum())
        c = np.where(f | m, 0.0, d).astype(d.dtype)
        out.append(smooth(c) if smooth_depth else c)
    if info is not None:
        info.flying_fraction = n_fly / total
        info.inconsistent_fraction = n_mv / total
    return out
