"""Scale stage (SPEC 3.1, 6.1 step 3): recover the single similarity transform from the model frame
to the scan frame.

With a printed ChArUco board: detect it in the full-resolution keyframes, map corners to model-grid
pixels, triangulate each corner across views with ``P = K[R|t]`` (pixel coordinates, no undistortion
-- the model returns a distortion-free pinhole ``K``), and fit a robust similarity (RANSAC over
corner subsets, Umeyama inside) to the board's known coordinates. The support plane (Open3D
``segment_plane`` on the triangulated/point cloud) defines Z; a board tilt above 10 degrees flags the
scan and falls back to ``object_bottom_center``. Fallbacks in order: the model's learned metric
scale, then the manual two-point ``/scale`` route (scale only).

Everything is analytic; no training and no per-scene optimization (SPEC 1.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from screwit.contracts import ScaleInfo

from . import board as board_mod

RESIDUAL_FLAG_MM = 2.0
MIN_CORNERS_IN_5_VIEWS = 8
BOARD_TILT_FLAG_DEG = 10.0


@dataclass
class ScaleResult:
    """The recovered model->scan similarity plus the ScaleInfo contract and flags."""

    r: np.ndarray            # (3, 3) rotation
    t: np.ndarray            # (3,) translation (mm)
    s: float                 # uniform scale
    info: ScaleInfo
    frame_origin: str        # "board" | "object_bottom_center"
    flags: list[str]
    support_plane: tuple[np.ndarray, float] | None = None  # (normal, d) in the scan frame


# --------------------------------------------------------------------------- Umeyama

def umeyama(src: np.ndarray, dst: np.ndarray, *, force_reflection: bool = False
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity (R, t, s) with s*R*src + t ~= dst (Umeyama 1991).

    ``force_reflection`` flips the sign choice on the smallest singular direction. For coplanar
    correspondences (a flat board) the out-of-plane axis is unconstrained, so the two proper
    rotations that fit equally differ by this flip; the caller picks the one that puts the cameras
    on the +Z side of the board (SPEC 3.1 "rotation comes from the support plane").
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / n
    u, d, vt = np.linalg.svd(cov)
    s_diag = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_diag[-1] = -1
    if force_reflection:
        s_diag[-1] *= -1
    r = u @ np.diag(s_diag) @ vt
    var_s = (sc ** 2).sum() / n
    scale = float((d * s_diag).sum() / var_s)
    t = mu_d - scale * r @ mu_s
    return r, t, scale


def ransac_similarity(src: np.ndarray, dst: np.ndarray, *, iters: int = 200,
                      threshold_mm: float = 2.0, rng: np.random.Generator | None = None):
    """RANSAC over corner subsets with Umeyama inside; returns (R, t, s, inlier_mask, rmse)."""
    rng = rng or np.random.default_rng(0)
    n = src.shape[0]
    if n < 4:
        r, t, s = umeyama(src, dst)
        res = np.linalg.norm(s * (src @ r.T) + t - dst, axis=1)
        return r, t, s, np.ones(n, bool), float(np.sqrt((res ** 2).mean()))
    best_inliers = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=4, replace=False)
        try:
            r, t, s = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm(s * (src @ r.T) + t - dst, axis=1)
        inliers = res < threshold_mm
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers
    if best_inliers is None or best_count < 4:
        best_inliers = np.ones(n, bool)
    r, t, s = umeyama(src[best_inliers], dst[best_inliers])
    res = np.linalg.norm(s * (src @ r.T) + t - dst, axis=1)
    rmse = float(np.sqrt((res[best_inliers] ** 2).mean()))
    return r, t, s, best_inliers, rmse


# --------------------------------------------------------------------------- triangulation

def triangulate_point(projs: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    """Direct multi-view DLT for one 3D point from P = K[R|t] matrices and pixel observations."""
    rows = []
    for p, (u, v) in zip(projs, pixels, strict=False):
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    a = np.stack(rows, axis=0)
    _, _, vt = np.linalg.svd(a)
    x = vt[-1]
    return x[:3] / x[3]


def projection_matrices(intrinsics: np.ndarray, extrinsics: np.ndarray) -> list[np.ndarray]:
    """P_i = K_i [R_i | t_i] from world-to-camera extrinsics (model frame)."""
    projs = []
    for i in range(extrinsics.shape[0]):
        k = intrinsics[i]
        rt = extrinsics[i, :3, :4]
        projs.append(k @ rt)
    return projs


# --------------------------------------------------------------------------- board detection

def detect_board_corners(full_res_image: np.ndarray):
    """Return (charuco_ids, pixel_corners) for the ChArUco board in a full-res frame, or (None,None)."""
    import cv2

    board = board_mod.charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    gray = full_res_image if full_res_image.ndim == 2 else cv2.cvtColor(full_res_image, cv2.COLOR_RGB2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) == 0:
        return None, None
    return charuco_ids.reshape(-1), charuco_corners.reshape(-1, 2)


def board_object_points() -> np.ndarray:
    """Known 3D positions (mm) of the inner ChArUco corners in OpenCV's board frame: origin at the
    top-left of the printed sheet, X right, Y DOWN the page, Z into the paper."""
    board = board_mod.charuco_board()
    return np.asarray(board.getChessboardCorners(), dtype=float)


def board_scan_points() -> np.ndarray:
    """The same corners in the scan frame the board defines (SPEC 3.1): origin at the printed origin
    corner (bottom-left, where the arrow starts), X along the arrow, Y up the page, Z out of the
    paper towards the cameras. Right-handed, so the board fit is a proper rotation and nothing is
    mirrored. This is OpenCV's frame rotated 180 degrees about X and shifted by the board height."""
    pts = board_object_points().copy()
    pts[:, 1] = board_mod.SQUARES_Y * board_mod.SQUARE_MM - pts[:, 1]
    pts[:, 2] = -pts[:, 2]
    return pts


# --------------------------------------------------------------------------- support plane

def support_plane(points_mm: np.ndarray):
    """Dominant plane under the scene via Open3D segment_plane; returns (normal(unit), d) with the
    plane n·x + d = 0 and the normal pointing up (+ half-space holding most points)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_mm, float))
    model, _ = pcd.segment_plane(distance_threshold=2.0, ransac_n=3, num_iterations=200)
    a, b, c, d = model
    n = np.array([a, b, c], float)
    norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    # Orient the normal so most points are on the positive side (object sits above the plane).
    signed = np.asarray(points_mm, float) @ n + d
    if np.mean(signed) < 0:
        n, d = -n, -d
    return n, float(d)


# --------------------------------------------------------------------------- levelling & support

def level_to_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform that makes a support plane exactly Z-up at z = 0 (SPEC 3.1).

    Given a plane ``normal . x + d = 0`` with ``normal`` pointing towards the cameras (the object
    side), returns ``(R, t)`` such that ``x' = R x + t`` rotates ``normal`` onto +Z and shifts the
    plane to z = 0. Applying it to the whole scan corrects any residual tilt left by an unreliable
    board rotation, so the object sits flat on z = 0 instead of slanted and floating. Object-agnostic
    and category-free: it only aligns the support surface.
    """
    n = np.asarray(normal, float)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return np.eye(3), np.zeros(3)
    n = n / nn
    d = float(d) / nn
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = float(np.clip(n @ z, -1.0, 1.0))
    sv = float(np.linalg.norm(v))
    if sv < 1e-9:
        # Already parallel to Z (up) or anti-parallel (flip about X).
        r = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        r = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (sv * sv))
    # After rotation the plane normal is +Z; a point on the plane maps to z = d (since R n = z and
    # the plane passes through -d n, which maps to -d z). Shift by +d so the plane lands on z = 0.
    t = np.array([0.0, 0.0, d])
    return r, t


def support_band_mask(points_mm: np.ndarray, normal: np.ndarray, d: float,
                      *, band_mm: float = 5.0) -> np.ndarray:
    """Boolean mask of points that are ON or BELOW the support plane within ``band_mm``.

    The support surface (a table, the floor, the board's sheet) sits on this plane; a round table
    top recorded from an elevation lands here too. ``True`` marks points to DROP so only the object
    above the surface is kept. The plane normal points at the object side. No object knowledge: it
    removes whatever lies in the support surface, not a named category.
    """
    p = np.asarray(points_mm, float)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    n = np.asarray(normal, float)
    nn = float(np.linalg.norm(n)) or 1.0
    signed = (p @ n + float(d)) / nn      # height above the plane (object side positive)
    return signed <= band_mm


# --------------------------------------------------------------------------- scale from board

def scale_from_board(intrinsics: np.ndarray, extrinsics: np.ndarray,
                     per_view_corners: dict[int, dict[int, np.ndarray]],
                     grid_mappings: dict[int, dict]) -> ScaleResult:
    """Recover the model->scan similarity from ChArUco corners.

    ``per_view_corners`` maps view index -> {charuco_id -> full-res pixel (u, v)}.
    ``grid_mappings`` maps view index -> the per-axis mapping from frames.json (scale_x, scale_y).
    """
    projs = projection_matrices(intrinsics, extrinsics)
    obj_pts = board_object_points()

    # Triangulate each corner seen in >= 2 views, counting the 5-view coverage flag.
    seen_counts: dict[int, int] = {}
    tri_ids: list[int] = []
    tri_pts: list[np.ndarray] = []
    for cid in range(obj_pts.shape[0]):
        views = []
        pixels = []
        for vi, corners in per_view_corners.items():
            if cid in corners:
                u, v = corners[cid]
                m = grid_mappings[vi]
                # Map full-res pixels into model-grid pixels (per-axis scale).
                pixels.append(np.array([u * m["scale_x"], v * m["scale_y"]]))
                views.append(vi)
        seen_counts[cid] = len(views)
        if len(views) >= 2:
            pts3 = triangulate_point([projs[v] for v in views], pixels)
            tri_ids.append(cid)
            tri_pts.append(pts3)

    if len(tri_pts) < 4:
        raise ValueError("too few triangulated board corners for a similarity fit")

    src = np.asarray(tri_pts)                 # model frame
    dst = board_scan_points()[tri_ids]        # scan frame, Z=0, right-handed, Z towards cameras
    r, t, s, inliers, rmse = ransac_similarity(src, dst, threshold_mm=RESIDUAL_FLAG_MM)
    if np.linalg.det(r) < 0:  # umeyama never returns one; guard the invariant explicitly
        raise ValueError("board fit produced a reflection")

    # With right-handed board coordinates the cameras must land on +Z. If they do not, the video
    # is mirrored (for example a front camera with mirroring on); reflecting the fit would mirror
    # the whole scan, so refuse and let the caller fall back.
    cam_centres_model = np.array([-extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3]
                                  for i in range(extrinsics.shape[0])])
    cam_z = (s * (cam_centres_model @ r.T) + t)[:, 2]
    if np.mean(cam_z) < 0:
        raise ValueError("cameras are below the board after the fit: mirrored video?")

    corners_in_5 = sum(1 for c in seen_counts.values() if c >= 5)
    flags: list[str] = []
    if rmse > RESIDUAL_FLAG_MM:
        flags.append(f"scale_residual_{rmse:.2f}mm_above_{RESIDUAL_FLAG_MM}mm")
    if corners_in_5 < MIN_CORNERS_IN_5_VIEWS:
        flags.append(f"only_{corners_in_5}_corners_in_5plus_views")

    # Board tilt: compare the board normal (scan-frame +Z of the board, mapped to scan) with the
    # support plane normal of the triangulated corners mapped to the scan frame.
    scan_pts = s * (src @ r.T) + t
    board_normal_scan = np.array([0, 0, 1.0])     # board defines Z=0 plane in scan frame
    try:
        sp_n, _ = support_plane(scan_pts)
    except Exception:
        sp_n = board_normal_scan
    tilt = float(np.degrees(np.arccos(np.clip(abs(sp_n @ board_normal_scan), -1, 1))))

    frame_origin = "board"
    if tilt > BOARD_TILT_FLAG_DEG:
        flags.append(f"board_tilt_{tilt:.1f}deg_above_{BOARD_TILT_FLAG_DEG}deg_lay_board_flat")
        frame_origin = "object_bottom_center"

    info = ScaleInfo(method="aruco", board=board_mod.BOARD_NAME, corner_views=corners_in_5,
                     residual_mm=round(rmse, 3), board_tilt_deg=round(tilt, 2))
    return ScaleResult(r=r, t=t, s=s, info=info, frame_origin=frame_origin, flags=flags,
                       support_plane=(board_normal_scan, 0.0))


# --------------------------------------------------------------------------- learned / manual

def scale_from_learned(learned_scale: float, board_residual_ratio: float | None = None,
                       *, metric: bool = False) -> ScaleResult:
    """Fallback when no scale board was seen.

    ``metric=True`` (DA3-nested: is_metric=1, depth already in METRES per cpu-verify FINDINGS item
    5/note 90) means the only conversion needed is metres -> millimetres, since the pipeline works in
    mm (TSDF divides depth by 1000, mask/bbox are mm). In that case the applied scale is the fixed
    1000x bridge -- NOT the model's scale_factor, which is already baked into the metric depth.

    ``metric=False`` (default; non-metric models, and the historical behaviour) applies
    ``learned_scale`` directly as the scan-frame scale.

    Rotation is deferred to the support plane; origin to the object-mask stage. Marked approximate
    because, unlike the board, this trusts the model's own estimate."""
    s = 1000.0 if metric else float(learned_scale)
    info = ScaleInfo(method="learned", learned_scale_ratio=board_residual_ratio or float(learned_scale))
    return ScaleResult(r=np.eye(3), t=np.zeros(3), s=s, info=info,
                       frame_origin="object_bottom_center", flags=["learned_scale_approximate"])


def rescale_factor_from_two_points(point_a_mm, point_b_mm, known_length_mm: float) -> float:
    """Manual /scale: the uniform factor that makes |a-b| equal the known length (SPEC 5)."""
    a = np.asarray(point_a_mm, float)
    b = np.asarray(point_b_mm, float)
    current = float(np.linalg.norm(a - b))
    if current <= 1e-9:
        raise ValueError("the two points coincide")
    return known_length_mm / current


# --------------------------------------------------------------------------- per-view board poses

MIN_PNP_CORNERS = 6        # a view needs this many board corners to be posed from the board alone
MAX_PNP_REPROJ_PX = 6.0    # full-res RMS reprojection error above which a view's board pose is dropped
BOARD_FIT_REJECT_MM = 10.0  # a multi-view board fit this far off is never used as the scale


def board_fit_failed(res: ScaleResult) -> bool:
    """The multi-view board fit is not usable as the scan frame: residual or tilt over the flag."""
    return (res.info.residual_mm or 0.0) > RESIDUAL_FLAG_MM or \
        (res.info.board_tilt_deg or 0.0) > BOARD_TILT_FLAG_DEG


def repose_from_board(result, per_view_corners: dict[int, dict[int, np.ndarray]],
                      grid_mappings: dict[int, dict]):
    """Pose every view that sees the board from the board itself, for when the model's own poses do
    not agree across views.

    On a real phone clip (8 views around black headphones on a plain floor) the same board corner,
    located from each view's depth map with the model's poses, landed up to about 0.75 m apart, so
    the multi-view board fit came out 48 mm off and 4x too small and the fused mesh was torn.
    Within any single view the model's depth is fine: the board corners fit their printed layout to
    1.4 mm. So each view gets its own camera pose from PnP on its board corners (full-resolution
    pixels; already in the scan frame, millimetres), and its depth is rescaled so the corners sit at
    the PnP depth. Views with too few corners or a poor PnP fit are left out.

    Returns (GeometryResult, ScaleResult) for the kept views, with the result's extrinsics and depth
    in metres in the scan frame and a ScaleResult of r = I, t = 0, s = 1000 (metres to millimetres),
    so the rest of the pipeline runs unchanged; or None when fewer than two views can be posed.
    """
    import dataclasses

    import cv2

    scan_pts = board_scan_points()
    kept, extr, depth, corner_err = [], [], [], []
    seen: dict[int, int] = {}
    for vi, corners in sorted(per_view_corners.items()):
        for cid in corners:
            seen[cid] = seen.get(cid, 0) + 1
        if len(corners) < MIN_PNP_CORNERS:
            continue
        ids = sorted(corners)
        obj = scan_pts[ids].astype(np.float64)
        img = np.array([corners[c] for c in ids], dtype=np.float64)
        m = grid_mappings[vi]
        sx, sy = float(m["scale_x"]), float(m["scale_y"])
        K = np.asarray(result.intrinsics[vi], dtype=np.float64)
        k_full = np.array([[K[0, 0] / sx, 0, (K[0, 2] + 0.5) / sx - 0.5],
                           [0, K[1, 1] / sy, (K[1, 2] + 0.5) / sy - 0.5], [0, 0, 1.0]])
        ok, rvec, tvec = cv2.solvePnP(obj, img, k_full, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, k_full, None, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, k_full, None)
        if float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1)))) > MAX_PNP_REPROJ_PX:
            continue
        r_wc, _ = cv2.Rodrigues(rvec)
        t_wc = tvec.reshape(3)
        if (-r_wc.T @ t_wc)[2] <= 0:      # the camera must be above the board (no mirrored fit)
            continue
        cam_pts = obj @ r_wc.T + t_wc       # corners in this camera's frame, mm
        d = np.asarray(result.depth[vi], dtype=np.float64)
        ratios = []
        for (u, v), z_mm in zip(img, cam_pts[:, 2], strict=True):
            x, y = int(round(u * sx)), int(round(v * sy))
            if 0 <= y < d.shape[0] and 0 <= x < d.shape[1] and d[y, x] > 0:
                ratios.append(z_mm / d[y, x])
        if len(ratios) < 4:
            continue
        mm_per_unit = float(np.median(ratios))
        # How far the corners land from their printed positions with this view's depth and pose.
        backproj = []
        for (u, v), cid in zip(img, ids, strict=True):
            x, y = u * sx, v * sy
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < d.shape[0] and 0 <= xi < d.shape[1] and d[yi, xi] > 0:
                ray = np.linalg.inv(K) @ np.array([x, y, 1.0])
                backproj.append(np.linalg.norm(r_wc.T @ (ray * d[yi, xi] * mm_per_unit - t_wc) - scan_pts[cid]))
        corner_err.extend(backproj)
        e = np.eye(4)
        e[:3, :3], e[:3, 3] = r_wc, t_wc / 1000.0
        kept.append(vi)
        extr.append(e)
        depth.append((d * mm_per_unit / 1000.0).astype(np.float32))
    if len(kept) < 2:
        return None
    new = dataclasses.replace(
        result, frames=[result.frames[i] for i in kept],
        intrinsics=np.asarray(result.intrinsics)[kept], extrinsics=np.asarray(extr),
        depth=depth, confidence=[result.confidence[i] for i in kept])
    rmse = float(np.sqrt(np.mean(np.square(corner_err)))) if corner_err else 0.0
    corners_in_5 = sum(1 for c in seen.values() if c >= 5)
    flags = [f"board_poses_per_view_{len(kept)}_of_{len(result.frames)}_views"]
    if rmse > RESIDUAL_FLAG_MM:
        flags.append(f"scale_residual_{rmse:.2f}mm_above_{RESIDUAL_FLAG_MM}mm")
    info = ScaleInfo(method="aruco", board=board_mod.BOARD_NAME, corner_views=corners_in_5,
                     residual_mm=round(rmse, 3), board_tilt_deg=0.0)
    res = ScaleResult(r=np.eye(3), t=np.zeros(3), s=1000.0, info=info, frame_origin="board",
                      flags=flags, support_plane=(np.array([0, 0, 1.0]), 0.0))
    return new, res
