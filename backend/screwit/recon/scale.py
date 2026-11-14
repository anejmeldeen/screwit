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
