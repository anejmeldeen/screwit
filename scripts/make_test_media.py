"""Deterministic realistic test-media generator for the Screw It pipeline.

Builds ``fixtures/media/`` from a synthetic CPU-rendered orbit scene (a flat
ChArUco board plus a matte test cube and a cylinder) and encodes it into a set
of files that mimic what real phones upload, together with a ``manifest.json``
describing every file and the scene ground truth.

The scene is rendered with a small numpy rasterizer (no GPU, no OpenGL). The
ChArUco board geometry/ids come from OpenCV so P2's board detector can run on
the real frames. Ground-truth cameras are written in the scan frame
(right-handed, millimetres, Z up) with OpenCV axes, matching SPEC 3.1 / the
camera contract.

Encoding uses the ``imageio-ffmpeg`` binary only. Properties are *never*
assumed: every encoded file is re-probed with ``ffmpeg -i`` afterwards and the
parsed probe output is stored in the manifest. Run with ``--help`` for options.

Usage:
    python scripts/make_test_media.py                 # write fixtures/media/
    python scripts/make_test_media.py --out /tmp/x    # write elsewhere (tests)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import piexif
from PIL import Image

# --------------------------------------------------------------------------
# Constants describing the physical scene (all lengths in millimetres).
# --------------------------------------------------------------------------

SEED = 20261114

# ChArUco board: the single definition from P2's backend/screwit/recon/board.py
# (AGENTS.md "The scale board (one definition)"). Constants copied, not imported
# across worktrees, so the detector P2 builds sees exactly this board.
BOARD_NAME = "charuco_5x7_30mm"
BOARD_SQUARES_X = 5
BOARD_SQUARES_Y = 7
BOARD_SQUARE_MM = 30.0
BOARD_MARKER_RATIO = 0.75
BOARD_MARKER_MM = BOARD_SQUARE_MM * BOARD_MARKER_RATIO  # 22.5 mm
BOARD_DICT_NAME = "DICT_4X4_50"
BOARD_DICT = cv2.aruco.DICT_4X4_50

# Matte test cube: 100 mm edge, sitting on the board plane next to the board.
CUBE_MM = 100.0
# Cylinder: 38.1 mm OD (1-1/2" tubular drain), 120 mm tall.
CYL_OD_MM = 38.1
CYL_HEIGHT_MM = 120.0

# Render grid for the ground-truth scene (kept modest for CPU speed; the
# encoded videos are scaled up to the phone-like resolutions by ffmpeg).
RENDER_W = 1280
RENDER_H = 720

# Camera orbit.
ORBIT_RADIUS_MM = 520.0
ORBIT_ELEV_DEG_MIN = 30.0
ORBIT_ELEV_DEG_MAX = 40.0
CAM_FOCAL_PX = 1100.0  # focal length in render-grid pixels

# Allowlisted codecs per SPEC 6.1 step 1 (MOV/MP4/WebM with H.264/HEVC/VP8/VP9).
ALLOWLISTED_CODECS = frozenset({"h264", "hevc", "vp8", "vp9"})
ALLOWLISTED_CONTAINERS = frozenset({"mov", "mp4", "webm"})
DURATION_MIN_S = 10.0
DURATION_MAX_S = 90.0

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------
# Small linear-algebra / rasterizer helpers.
# --------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def look_at_world_to_camera(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a 4x4 row-major world_to_camera matrix, OpenCV axes.

    OpenCV camera: X right, Y down, Z forward (into the scene).
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # OpenCV: Y is down, so camera-right = forward x up(world) then recompute.
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    # Rotation rows map world vectors into camera axes.
    rot = np.stack([right, down, forward], axis=0)  # 3x3
    t = -rot @ eye
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = t
    return m


def intrinsics(width: int, height: int, focal_px: float) -> np.ndarray:
    return np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project(points_world: np.ndarray, w2c: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixels (OpenCV pinhole). Returns NaN behind camera."""
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = cam[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = (k @ (cam / z).T).T[:, :2]
    pix[np.squeeze(z, 1) <= 1e-6] = np.nan
    return pix


# --------------------------------------------------------------------------
# ChArUco board texture and 3D corner geometry.
# --------------------------------------------------------------------------


def build_board():
    """Return (board, charuco_corner_ids, charuco_corners_mm) for the scene.

    Board lies on the Z=0 plane. Origin is the board's origin corner; X along
    the board's printed-arrow axis, Y the other in-plane axis, Z up.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(BOARD_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        BOARD_SQUARE_MM,
        BOARD_MARKER_MM,
        dictionary,
    )
    # Interior chessboard corners in board coordinates (mm), Z=0. The texture
    # is mirrored in X (render_board_texture), so the physical world X of each
    # corner is mirrored to match: world_x = board_width - cb_x.
    cb = board.getChessboardCorners().astype(np.float64)  # (N,3)
    board_width = BOARD_SQUARES_X * BOARD_SQUARE_MM
    corners_mm = cb.copy()
    corners_mm[:, 0] = board_width - cb[:, 0]
    ids = np.arange(corners_mm.shape[0])
    return board, ids, corners_mm


def render_board_texture(board, px_per_mm: float = 8.0) -> np.ndarray:
    """Render the board to a grayscale image used as a planar texture."""
    w_mm = BOARD_SQUARES_X * BOARD_SQUARE_MM
    h_mm = BOARD_SQUARES_Y * BOARD_SQUARE_MM
    w_px = int(round(w_mm * px_per_mm))
    h_px = int(round(h_mm * px_per_mm))
    img = board.generateImage((w_px, h_px), marginSize=0)
    # The board is textured onto a Z-up world plane and viewed from above. That
    # mapping is orientation-reversing relative to the printed sheet, which
    # mirrors the (chiral) ArUco markers so they never decode. Mirror the
    # texture horizontally to undo it; build_scene() mirrors the stored corner
    # X coordinates to match, so detections reproject onto the ground-truth
    # cameras to sub-pixel accuracy (verified in the tests).
    img = np.fliplr(img).copy()
    return img  # uint8 grayscale


# --------------------------------------------------------------------------
# Scene rasterizer (painter's algorithm with a z-buffer for the solids).
# --------------------------------------------------------------------------


@dataclass
class Scene:
    board: object
    board_tex: np.ndarray
    board_origin_mm: np.ndarray  # world position of board (0,0) corner
    cube_center_xy: np.ndarray   # world XY of cube centre (on Z=0)
    cyl_center_xy: np.ndarray    # world XY of cylinder axis (on Z=0)
    charuco_ids: np.ndarray
    charuco_corners_mm: np.ndarray  # in board coords (Z=0)

    def charuco_world(self) -> np.ndarray:
        c = self.charuco_corners_mm.copy()
        return c + self.board_origin_mm


def build_scene() -> Scene:
    board, ids, corners = build_board()
    tex = render_board_texture(board)
    # Board occupies x in [0,150], y in [0,210]. Put the cube and cylinder
    # beyond the board in +x and spread in y so they sit beside it and do not
    # occlude the markers from the orbit's elevated viewpoints.
    board_origin = np.array([0.0, 0.0, 0.0])
    cube_center_xy = np.array([150.0 + 95.0, 55.0])  # centre of 100mm cube
    cyl_center_xy = np.array([150.0 + 85.0, 165.0])
    return Scene(
        board=board,
        board_tex=tex,
        board_origin_mm=board_origin,
        cube_center_xy=cube_center_xy,
        cyl_center_xy=cyl_center_xy,
        charuco_ids=ids,
        charuco_corners_mm=corners,
    )


def _rasterize_tri(
    img: np.ndarray,
    zbuf: np.ndarray,
    pts: np.ndarray,
    zs: np.ndarray,
    color: np.ndarray,
) -> None:
    """Fill one triangle with a flat colour using a z-buffer (zs = camera depth)."""
    h, w = img.shape[:2]
    xs = pts[:, 0]
    ys = pts[:, 1]
    min_x = max(int(math.floor(xs.min())), 0)
    max_x = min(int(math.ceil(xs.max())), w - 1)
    min_y = max(int(math.floor(ys.min())), 0)
    max_y = min(int(math.ceil(ys.max())), h - 1)
    if max_x < min_x or max_y < min_y:
        return
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    xf = xx + 0.5
    yf = yy + 0.5
    a = ((y1 - y2) * (xf - x2) + (x2 - x1) * (yf - y2)) / denom
    b = ((y2 - y0) * (xf - x2) + (x0 - x2) * (yf - y2)) / denom
    c = 1.0 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    if not inside.any():
        return
    depth = a * zs[0] + b * zs[1] + c * zs[2]
    region_z = zbuf[min_y : max_y + 1, min_x : max_x + 1]
    region_i = img[min_y : max_y + 1, min_x : max_x + 1]
    closer = inside & (depth < region_z)
    region_z[closer] = depth[closer]
    region_i[closer] = color


def _rasterize_tri_tex(
    img: np.ndarray,
    zbuf: np.ndarray,
    pts: np.ndarray,
    zs: np.ndarray,
    uvs: np.ndarray,
    tex: np.ndarray,
) -> None:
    """Fill a triangle sampling a grayscale texture by barycentric uv."""
    h, w = img.shape[:2]
    th, tw = tex.shape[:2]
    xs = pts[:, 0]
    ys = pts[:, 1]
    min_x = max(int(math.floor(xs.min())), 0)
    max_x = min(int(math.ceil(xs.max())), w - 1)
    min_y = max(int(math.floor(ys.min())), 0)
    max_y = min(int(math.ceil(ys.max())), h - 1)
    if max_x < min_x or max_y < min_y:
        return
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    xf = xx + 0.5
    yf = yy + 0.5
    a = ((y1 - y2) * (xf - x2) + (x2 - x1) * (yf - y2)) / denom
    b = ((y2 - y0) * (xf - x2) + (x0 - x2) * (yf - y2)) / denom
    c = 1.0 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    if not inside.any():
        return
    depth = a * zs[0] + b * zs[1] + c * zs[2]
    region_z = zbuf[min_y : max_y + 1, min_x : max_x + 1]
    region_i = img[min_y : max_y + 1, min_x : max_x + 1]
    closer = inside & (depth < region_z)
    if not closer.any():
        return
    u = a * uvs[0, 0] + b * uvs[1, 0] + c * uvs[2, 0]
    v = a * uvs[0, 1] + b * uvs[1, 1] + c * uvs[2, 1]
    # Perspective-correct the UV: screen-space barycentric weights interpolate
    # uv/z and 1/z linearly, so divide to recover true uv. Affine interpolation
    # otherwise warps the ArUco grid under oblique views and shifts corners by
    # pixels, which breaks the "ground truth" reprojection.
    inv_z = a / zs[0] + b / zs[1] + c / zs[2]
    u = (a * uvs[0, 0] / zs[0] + b * uvs[1, 0] / zs[1] + c * uvs[2, 0] / zs[2]) / inv_z
    v = (a * uvs[0, 1] / zs[0] + b * uvs[1, 1] / zs[1] + c * uvs[2, 1] / zs[2]) / inv_z
    # Bilinear sample keeps the ArUco marker edges crisp under perspective.
    fu = np.clip(u * (tw - 1), 0, tw - 1)
    fv = np.clip(v * (th - 1), 0, th - 1)
    x0 = np.floor(fu).astype(np.int32)
    y0 = np.floor(fv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tw - 1)
    y1 = np.clip(y0 + 1, 0, th - 1)
    wx = fu - x0
    wy = fv - y0
    s00 = tex[y0, x0].astype(np.float64)
    s01 = tex[y0, x1].astype(np.float64)
    s10 = tex[y1, x0].astype(np.float64)
    s11 = tex[y1, x1].astype(np.float64)
    sampled = (
        s00 * (1 - wx) * (1 - wy)
        + s01 * wx * (1 - wy)
        + s10 * (1 - wx) * wy
        + s11 * wx * wy
    )
    gray = np.stack([sampled, sampled, sampled], axis=-1).astype(np.uint8)
    region_i[closer] = gray[closer]
    region_z[closer] = depth[closer]


def _cube_mesh(center_xy: np.ndarray, size: float):
    cx, cy = center_xy
    h = size / 2.0
    # 8 vertices, cube sitting on Z=0.
    v = np.array(
        [
            [cx - h, cy - h, 0.0],
            [cx + h, cy - h, 0.0],
            [cx + h, cy + h, 0.0],
            [cx - h, cy + h, 0.0],
            [cx - h, cy - h, size],
            [cx + h, cy - h, size],
            [cx + h, cy + h, size],
            [cx - h, cy + h, size],
        ]
    )
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 5, 1), (0, 4, 5),  # side y-
        (1, 5, 6), (1, 6, 2),  # side x+
        (2, 6, 7), (2, 7, 3),  # side y+
        (3, 7, 4), (3, 4, 0),  # side x-
    ]
    return v, faces


def _cyl_mesh(center_xy: np.ndarray, radius: float, height: float, segments: int = 32):
    cx, cy = center_xy
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    bottom = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang), np.zeros(segments)], axis=1)
    top = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang), np.full(segments, height)], axis=1)
    v = np.concatenate([bottom, top], axis=0)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((i, j, segments + j))
        faces.append((i, segments + j, segments + i))
    # top cap fan
    for i in range(1, segments - 1):
        faces.append((segments, segments + i, segments + i + 1))
    return v, faces


def render_frame(
    scene: Scene,
    w2c: np.ndarray,
    k: np.ndarray,
    blur: bool,
    rng: np.random.Generator,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Render one RGB frame. Defaults to the landscape RENDER_W x RENDER_H grid;
    pass width/height for a portrait grid (the iPhone clip)."""
    w = RENDER_W if width is None else width
    h = RENDER_H if height is None else height
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (60, 70, 80)  # neutral background
    zbuf = np.full((h, w), np.inf)

    def cam_depth(points):
        homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        return (w2c @ homo.T).T[:, 2]

    # Board: two textured triangles over its rectangle.
    w_mm = BOARD_SQUARES_X * BOARD_SQUARE_MM
    h_mm = BOARD_SQUARES_Y * BOARD_SQUARE_MM
    o = scene.board_origin_mm
    corners3d = np.array(
        [
            [o[0], o[1], 0.0],
            [o[0] + w_mm, o[1], 0.0],
            [o[0] + w_mm, o[1] + h_mm, 0.0],
            [o[0], o[1] + h_mm, 0.0],
        ]
    )
    board_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    bpix = project(corners3d, w2c, k)
    bdepth = cam_depth(corners3d)
    if not np.isnan(bpix).any():
        _rasterize_tri_tex(img, zbuf, bpix[[0, 1, 2]], bdepth[[0, 1, 2]], board_uv[[0, 1, 2]], scene.board_tex)
        _rasterize_tri_tex(img, zbuf, bpix[[0, 2, 3]], bdepth[[0, 2, 3]], board_uv[[0, 2, 3]], scene.board_tex)

    # Cube (matte light gray) with per-face shading.
    cv_, cf = _cube_mesh(scene.cube_center_xy, CUBE_MM)
    cpix = project(cv_, w2c, k)
    cdepth = cam_depth(cv_)
    base_cube = np.array([200, 200, 195], dtype=np.float64)
    for fi, f in enumerate(cf):
        if np.isnan(cpix[list(f)]).any():
            continue
        shade = 0.6 + 0.4 * ((fi // 2) / 5.0)
        _rasterize_tri(img, zbuf, cpix[list(f)], cdepth[list(f)], (base_cube * shade).astype(np.uint8))

    # Cylinder (matte warm gray).
    yv, yf = _cyl_mesh(scene.cyl_center_xy, CYL_OD_MM / 2.0, CYL_HEIGHT_MM)
    ypix = project(yv, w2c, k)
    ydepth = cam_depth(yv)
    base_cyl = np.array([170, 160, 150], dtype=np.float64)
    for fi, f in enumerate(yf):
        if np.isnan(ypix[list(f)]).any():
            continue
        shade = 0.55 + 0.45 * ((fi % 8) / 8.0)
        _rasterize_tri(img, zbuf, ypix[list(f)], ydepth[list(f)], (base_cyl * shade).astype(np.uint8))

    if blur:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=2.5)
    return img


# --------------------------------------------------------------------------
# Camera orbit and full scene render.
# --------------------------------------------------------------------------


def orbit_cameras(n: int):
    """Yield (world_to_camera 4x4, elevation_deg, azimuth_deg) for n views."""
    target = np.array([150.0, 105.0, 30.0])  # centre of board+objects cluster
    up = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        az = 2 * math.pi * i / n
        # Elevation sweeps smoothly between min and max within the orbit.
        elev = math.radians(
            ORBIT_ELEV_DEG_MIN
            + (ORBIT_ELEV_DEG_MAX - ORBIT_ELEV_DEG_MIN) * 0.5 * (1 - math.cos(2 * az))
        )
        eye = target + ORBIT_RADIUS_MM * np.array(
            [math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)]
        )
        w2c = look_at_world_to_camera(eye, target, up)
        yield w2c, math.degrees(elev), math.degrees(az)


@dataclass
class RenderedScene:
    frames: list[np.ndarray]
    cameras: list[np.ndarray]  # world_to_camera 4x4
    blurred: list[bool]
    k: np.ndarray
    scene: Scene


def render_orbit(n_frames: int, blur_fraction: float = 0.2) -> RenderedScene:
    rng = _rng()
    scene = build_scene()
    k = intrinsics(RENDER_W, RENDER_H, CAM_FOCAL_PX)
    # Deterministic choice of which frames get motion blur.
    blur_flags = [((i * 7 + 3) % 10) < int(round(blur_fraction * 10)) for i in range(n_frames)]
    frames = []
    cams = []
    for i, (w2c, _elev, _az) in enumerate(orbit_cameras(n_frames)):
        frames.append(render_frame(scene, w2c, k, blur_flags[i], rng))
        cams.append(w2c)
    return RenderedScene(frames=frames, cameras=cams, blurred=blur_flags, k=k, scene=scene)


# --------------------------------------------------------------------------
# Portrait (iPhone) render path.
#
# A real iPhone portrait clip stores landscape 1920x1080 samples plus a 90 deg
# display-matrix tag; a decoder that honours the tag shows upright 1080x1920
# frames. We reproduce that exactly: render the scene UPRIGHT in a portrait
# 1080x1920 display frame (camera rolled about its optical axis + portrait K),
# then store each frame rotated into the 1920x1080 orientation that a 90 deg
# display rotation turns back upright, and stamp the tag. The displayed frames
# a decoder (and P2's frames.py) produces are the upright portrait ones.
# --------------------------------------------------------------------------

PORTRAIT_DISPLAY_W = 1080
PORTRAIT_DISPLAY_H = 1920
PORTRAIT_RENDER_W = 720   # same 9:16 aspect as the display; scaled up at encode
PORTRAIT_RENDER_H = 1280
PORTRAIT_FOCAL_PX = CAM_FOCAL_PX * (PORTRAIT_RENDER_H / RENDER_H)  # keep the same world FoV on the long axis


def portrait_intrinsics_render() -> np.ndarray:
    return intrinsics(PORTRAIT_RENDER_W, PORTRAIT_RENDER_H, PORTRAIT_FOCAL_PX)


def portrait_intrinsics_display() -> np.ndarray:
    """K in the DISPLAYED (1080x1920) portrait frame: the render K scaled up."""
    scale = PORTRAIT_DISPLAY_W / PORTRAIT_RENDER_W
    return intrinsics(PORTRAIT_DISPLAY_W, PORTRAIT_DISPLAY_H, PORTRAIT_FOCAL_PX * scale)


def render_orbit_portrait(n_frames: int, blur_fraction: float = 0.2) -> RenderedScene:
    """Render the same orbit as render_orbit but into upright portrait display frames.

    The camera up-vector is world +Z (as in the landscape orbit), so the scene is
    upright and world +Z projects UP the portrait frame; only the image aspect and
    K change (taller 9:16 grid, portrait K). The stored `k` is the DISPLAYED-frame
    (1080x1920) portrait K and the stored `cameras` are that orbit's world_to_camera
    (the ground truth for the displayed portrait frames)."""
    rng = _rng()
    scene = build_scene()
    k_render = portrait_intrinsics_render()
    k_display = portrait_intrinsics_display()
    blur_flags = [((i * 7 + 3) % 10) < int(round(blur_fraction * 10)) for i in range(n_frames)]
    frames = []
    cams = []
    for i, (w2c, _elev, _az) in enumerate(orbit_cameras(n_frames)):
        frames.append(render_frame(scene, w2c, k_render, blur_flags[i], rng,
                                   width=PORTRAIT_RENDER_W, height=PORTRAIT_RENDER_H))
        cams.append(w2c)
    return RenderedScene(frames=frames, cameras=cams, blurred=blur_flags, k=k_display, scene=scene)


def display_to_storage(display_frame: np.ndarray) -> np.ndarray:
    """Rotate an upright portrait display frame into the landscape STORAGE frame.

    A 90 deg display rotation turns storage back into display as
    `display = np.rot90(storage, 1)` (verified against ffmpeg autorotate), so the
    inverse is `storage = np.rot90(display, -1)`. The stored frame is 1920 wide x
    1080 tall; the decoded (displayed) frame is 1080 wide x 1920 tall."""
    return np.ascontiguousarray(np.rot90(display_frame, -1))


def board_pose_per_frame(rs: RenderedScene) -> list[dict]:
    """Ground-truth board pose in each camera frame: rvec/tvec (OpenCV, mm).

    The board frame == scan frame here (board origin at world origin, Z up), so
    the board-to-camera transform is exactly world_to_camera.
    """
    poses = []
    for w2c in rs.cameras:
        rvec, _ = cv2.Rodrigues(w2c[:3, :3])
        poses.append(
            {
                "rvec": rvec.reshape(3).tolist(),
                "tvec": w2c[:3, 3].tolist(),
                "board_to_camera": w2c.tolist(),
            }
        )
    return poses


def board_detection_stats(rs: RenderedScene) -> dict:
    """Detect the ChArUco board in every base frame and report accuracy.

    This is ground truth about the board's realism: how many frames a detector
    actually sees it, and the sub-pixel reprojection of the detected corners
    against the stored cameras. Computed here, not assumed.
    """
    board = rs.scene.board
    detector = cv2.aruco.CharucoDetector(board)
    corners_world = rs.scene.charuco_world()
    per_frame = []
    reproj_all = []
    detected = 0
    for i, frame in enumerate(rs.frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cc, cids, _mc, _mids = detector.detectBoard(gray)
        n = 0 if cc is None else len(cc)
        rp = None
        if cc is not None and n >= 1:
            obj = corners_world[cids.flatten()]
            proj = project(obj, rs.cameras[i], rs.k)
            rp = float(np.nanmean(np.linalg.norm(proj - cc.reshape(-1, 2), axis=1)))
            reproj_all.append(rp)
        if n >= 8:
            detected += 1
        per_frame.append({"index": i, "corners": int(n), "reproj_px": rp,
                          "blurred": bool(rs.blurred[i])})
    return {
        "frames_with_8plus_corners": detected,
        "total_frames": len(rs.frames),
        "reproj_px_mean": round(float(np.mean(reproj_all)), 3) if reproj_all else None,
        "reproj_px_max": round(float(np.max(reproj_all)), 3) if reproj_all else None,
        "per_frame": per_frame,
    }


# --------------------------------------------------------------------------
# ffmpeg encode + probe.
# --------------------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def _write_raw_rgb(frames: list[np.ndarray], path: Path) -> None:
    with open(path, "wb") as f:
        for fr in frames:
            f.write(np.ascontiguousarray(fr).tobytes())


def _raw_input_args(width: int, height: int, fps: float) -> list[str]:
    return [
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{fps}",
        "-i", "-",
    ]


def encode_from_frames(
    frames: list[np.ndarray],
    fps: float,
    out_path: Path,
    vcodec_args: list[str],
    out_w: int,
    out_h: int,
    container_args: list[str] | None = None,
    scale_to: tuple[int, int] | None = None,
    in_w: int | None = None,
    in_h: int | None = None,
) -> None:
    """Pipe raw RGB frames into ffmpeg and encode. Input frames are in_w x in_h
    (defaulting to the landscape RENDER grid); ffmpeg scales to (out_w, out_h)."""
    iw = RENDER_W if in_w is None else in_w
    ih = RENDER_H if in_h is None else in_h
    vf = f"scale={out_w}:{out_h}:flags=bicubic"
    cmd = [
        FFMPEG, "-hide_banner", "-y",
        # Single-threaded encode so the output is byte-reproducible run to run
        # (libvpx-vp9 in particular is not deterministic with multiple threads).
        "-threads", "1",
        *_raw_input_args(iw, ih, fps),
        "-vf", vf,
        *vcodec_args,
        *(container_args or []),
        str(out_path),
    ]
    raw = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
    proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encode failed for {out_path.name}:\n{proc.stdout.decode(errors='replace')[-2000:]}"
        )


def apply_display_rotation(src: Path, dst: Path, degrees: int) -> None:
    """Stream-copy src -> dst writing a display-matrix rotation (input-side opt)."""
    cmd = [
        FFMPEG, "-hide_banner", "-y",
        "-display_rotation:v:0", str(degrees),
        "-i", str(src),
        "-c", "copy",
        str(dst),
    ]
    out = _run(cmd)
    if not dst.exists():
        raise RuntimeError(f"display-rotation copy failed:\n{out[-1500:]}")


# ---- probe parsing ("ffmpeg -i", no ffprobe) ------------------------------

_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
_STREAM_RE = re.compile(r"Stream #\d+:\d+.*: Video:\s*(.*)")
_ROT_RE = re.compile(r"Display Matrix: rotation of ([-\d.]+) degrees")
_RES_RE = re.compile(r"(\d{2,5})x(\d{2,5})")


def decode_check(path: Path) -> dict:
    """Fully decode the video to the null muxer; report success and frame count.

    A readable container header does not mean the stream decodes: a truncated
    file still reports a duration from its moov box but errors partway through
    decoding. The server's validation job decodes, so we record the same.
    """
    out = _run([FFMPEG, "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"])
    errored = bool(out.strip())
    cnt = _run([FFMPEG, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"])
    fm = re.findall(r"frame=\s*(\d+)", cnt)
    frames = int(fm[-1]) if fm else None
    return {"decode_ok": not errored, "decoded_frames": frames,
            "decode_errors_tail": out.strip().splitlines()[-4:] if errored else []}


def probe(path: Path) -> dict:
    """Parse ``ffmpeg -i`` output into a structured probe dict (no ffprobe)."""
    out = _run([FFMPEG, "-hide_banner", "-i", str(path)])
    info: dict = {"raw_tail": out.strip().splitlines()[-12:]}

    dur = _DUR_RE.search(out)
    if dur:
        h, m, s = int(dur.group(1)), int(dur.group(2)), float(dur.group(3))
        info["duration_s"] = round(h * 3600 + m * 60 + s, 3)
    else:
        info["duration_s"] = None

    # Container: from "Input #0, <names>,".
    cont = re.search(r"Input #0,\s*([^,]+(?:,[^,]+)*?),\s*from", out)
    info["container_names"] = cont.group(1).strip() if cont else None

    sm = _STREAM_RE.search(out)
    if sm:
        vline = sm.group(1)
        info["video_line"] = vline.strip()
        # codec is first token.
        info["codec"] = vline.split("(")[0].split(",")[0].strip().split()[0]
        # pixel format: token after resolution or inside parens; grab known pix_fmts.
        pix = re.search(r"(yuv420p10le|yuv420p|yuvj420p|yuv422p|yuv444p|gbrp)", vline)
        info["pix_fmt"] = pix.group(1) if pix else None
        # colour transfer / primaries (bt2020, arib-std-b67/HLG, bt709 ...).
        col = re.search(r"\(([^)]*(?:bt2020|bt709|arib-std-b67|smpte2084|unknown)[^)]*)\)", vline)
        info["color_tags"] = col.group(1) if col else None
        res = _RES_RE.search(vline)
        if res:
            info["width"] = int(res.group(1))
            info["height"] = int(res.group(2))
        fps = re.search(r"([\d.]+)\s*fps", vline)
        info["fps"] = float(fps.group(1)) if fps else None
    else:
        info["video_line"] = None
        info["codec"] = None

    rot = _ROT_RE.search(out)
    info["rotation_deg"] = float(rot.group(1)) if rot else 0.0
    info.update(decode_check(path))
    return info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Validator verdict model (mirrors SPEC 2.2 / 3.3 server-side checks).
# --------------------------------------------------------------------------


def expected_verdict(
    container: str,
    codec: str,
    duration_s: float | None,
    truncated: bool,
    is_magic_video: bool,
) -> dict:
    """What a correct server-side media validator SHOULD decide for this file.

    This encodes the rules from SPEC 2.2 and 3.3 / 6.1 step 1:
      * magic bytes must be a real media container (a .mp4 that is text fails);
      * codec must be in the allowlist;
      * duration must be within 10-90 s;
      * a truncated/undecodable file is rejected.
    """
    reasons: list[str] = []
    if not is_magic_video:
        reasons.append("not_media_container")
    if container not in ALLOWLISTED_CONTAINERS:
        reasons.append("container_not_allowed")
    if codec not in ALLOWLISTED_CODECS:
        reasons.append("codec_not_allowed")
    if truncated:
        reasons.append("truncated_or_undecodable")
    if duration_s is None:
        reasons.append("duration_unknown")
    elif duration_s < DURATION_MIN_S:
        reasons.append("too_short")
    elif duration_s > DURATION_MAX_S:
        reasons.append("too_long")
    return {"verdict": "reject" if reasons else "accept", "reasons": reasons}


# --------------------------------------------------------------------------
# Build the full media set.
# --------------------------------------------------------------------------

# Video durations: keep well under 60 MB total via modest bitrates.
# (a) iPhone MOV HEVC: 20 s, (b) Android MP4 H264: 25 s, (c) WebM VP9: 15 s.
# Render one orbit of N frames and loop it to fill each duration to keep the
# render cheap and deterministic.
BASE_FRAMES = 64


def _fill_frames(base: list[np.ndarray], count: int) -> list[np.ndarray]:
    return [base[i % len(base)] for i in range(count)]


def build(out_dir: Path) -> dict:
    media_dir = out_dir
    media_dir.mkdir(parents=True, exist_ok=True)
    work = media_dir / "_work"
    work.mkdir(exist_ok=True)

    rs = render_orbit(BASE_FRAMES)

    manifest: dict = {
        "generator": "scripts/make_test_media.py",
        "seed": SEED,
        "scene": {
            "render_grid": [RENDER_W, RENDER_H],
            "camera_K_render_grid": rs.k.tolist(),
            "frame": "scan frame: right-handed, millimetres, Z up; OpenCV camera axes",
            "board": {
                "type": "charuco",
                "name": BOARD_NAME,
                "squares_x": BOARD_SQUARES_X,
                "squares_y": BOARD_SQUARES_Y,
                "square_mm": BOARD_SQUARE_MM,
                "marker_mm": BOARD_MARKER_MM,
                "marker_ratio": BOARD_MARKER_RATIO,
                "dictionary": BOARD_DICT_NAME,
                "note": "matches P2 backend/screwit/recon/board.py (the single scale-board definition)",
            },
            "objects": {
                "cube": {"edge_mm": CUBE_MM, "center_xy_mm": rs.scene.cube_center_xy.tolist()},
                "cylinder": {"outer_diameter_mm": CYL_OD_MM, "height_mm": CYL_HEIGHT_MM,
                             "center_xy_mm": rs.scene.cyl_center_xy.tolist()},
            },
            "known_object_size_mm": {"cube_edge": CUBE_MM, "cylinder_od": CYL_OD_MM},
        },
        "ground_truth": {
            "charuco_corner_ids": rs.scene.charuco_ids.tolist(),
            "charuco_corners_scan_mm": rs.scene.charuco_world().tolist(),
            "cameras_world_to_camera": [c.tolist() for c in rs.cameras],
            "blurred_frames": rs.blurred,
            "board_pose_per_frame": board_pose_per_frame(rs),
            "board_detection": board_detection_stats(rs),
        },
        "files": [],
    }

    # Save ground-truth cameras.json alongside the media (non-encoded artifact,
    # hashed for determinism checks).
    cameras_json = {
        "k_render_grid": rs.k.tolist(),
        "frames": [
            {"index": i, "world_to_camera": c.tolist(), "blurred": rs.blurred[i]}
            for i, c in enumerate(rs.cameras)
        ],
    }
    cam_path = media_dir / "cameras.json"
    cam_path.write_text(json.dumps(cameras_json, indent=2, sort_keys=True) + "\n")

    files_meta: list[dict] = []

    def record(name: str, declared: dict, substitute: bool = False, note: str = "") -> None:
        p = media_dir / name
        pr = probe(p) if p.suffix != ".txt" and _looks_media(p) else _probe_nonmedia(p)
        entry = {
            "name": name,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            **declared,
            "substitute": substitute,
            "probe": pr,
        }
        if note:
            entry["note"] = note
        files_meta.append(entry)

    # ---- (a) iPhone-like MOV: upright PORTRAIT scene stored as landscape samples
    # + a 90 deg display-matrix tag, exactly like a real iPhone portrait clip.
    # Render the orbit into upright 720x1280 portrait display frames, rotate each
    # into the 1280x720 landscape STORAGE orientation that a 90 deg display
    # rotation turns back upright, encode at 1920x1080 (HEVC Main10, BT.2020/HLG,
    # 20 s), then stamp the display rotation. A decoder (and P2's frames.py)
    # autorotates to upright 1080x1920 display frames.
    rs_portrait = render_orbit_portrait(BASE_FRAMES)
    display_frames_a = _fill_frames(rs_portrait.frames, int(20 * 30))  # 20 s at 30 fps
    storage_frames_a = [display_to_storage(f) for f in display_frames_a]
    store_h, store_w = storage_frames_a[0].shape[:2]  # 720 x 1280 (landscape)
    a_tmp = work / "a_hevc.mov"
    encode_from_frames(
        storage_frames_a, 30, a_tmp,
        vcodec_args=[
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-profile:v", "main10",
            "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
            "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:log-level=none",
            "-tag:v", "hvc1", "-crf", "30",
        ],
        out_w=1920, out_h=1080,
        container_args=["-movflags", "+faststart", "-f", "mov"],
        in_w=store_w, in_h=store_h,
    )
    a_final = media_dir / "iphone_hevc_hlg_portrait.mov"
    apply_display_rotation(a_tmp, a_final, 90)
    iphone_ground_truth = {
        "displayed_resolution": [PORTRAIT_DISPLAY_W, PORTRAIT_DISPLAY_H],
        "stored_resolution": [1920, 1080],
        "display_rotation_deg": 90,
        "note": (
            "upright portrait scene; samples stored landscape 1920x1080, a 90 deg display matrix "
            "turns them back upright to 1080x1920. world_to_camera and K are in the DISPLAYED "
            "(portrait 1080x1920) image frame; the orbit matches the top-level landscape ground truth."
        ),
        "camera_K_display": rs_portrait.k.tolist(),
        "charuco_corner_ids": rs_portrait.scene.charuco_ids.tolist(),
        "charuco_corners_scan_mm": rs_portrait.scene.charuco_world().tolist(),
        "cameras_world_to_camera_display": [c.tolist() for c in rs_portrait.cameras],
        "blurred_frames": rs_portrait.blurred,
    }
    record(
        "iphone_hevc_hlg_portrait.mov",
        declared={
            "description": (
                "iPhone-like MOV, HEVC Main10 10-bit, BT.2020/HLG, upright PORTRAIT, "
                "stored 1920x1080 + 90 deg display matrix -> displayed 1080x1920, 20s"
            ),
            "container": "mov", "codec": "hevc", "pix_fmt": "yuv420p10le",
            "color_transfer": "arib-std-b67 (HLG)", "color_primaries": "bt2020",
            "resolution": [1920, 1080], "displayed_resolution": [1080, 1920],
            "fps": 30, "duration_s": 20, "rotation_deg": 90,
            "hdr": True,
            "expected_validator": expected_verdict("mov", "hevc", 20.0, False, True),
            "expected_keyframe_count_range": [48, 64],
            "ground_truth": iphone_ground_truth,
        },
    )

    # ---- (b) Android-like MP4 H.264 yuv420p 1920x1080 30 fps 25 s.
    frames_b = _fill_frames(rs.frames, int(25 * 30))
    b_final = media_dir / "android_h264.mp4"
    encode_from_frames(
        frames_b, 30, b_final,
        vcodec_args=[
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-crf", "28",
        ],
        out_w=1920, out_h=1080,
        container_args=["-movflags", "+faststart", "-f", "mp4"],
    )
    record(
        "android_h264.mp4",
        declared={
            "description": "Android-like MP4, H.264 yuv420p BT.709, 1920x1080@30, 25s",
            "container": "mp4", "codec": "h264", "pix_fmt": "yuv420p",
            "color_transfer": "bt709", "color_primaries": "bt709",
            "resolution": [1920, 1080], "fps": 30, "duration_s": 25, "rotation_deg": 0,
            "hdr": False,
            "expected_validator": expected_verdict("mp4", "h264", 25.0, False, True),
            "expected_keyframe_count_range": [48, 64],
        },
    )

    # ---- (c) WebM VP9 1280x720 15 s.
    frames_c = _fill_frames(rs.frames, int(15 * 30))
    c_final = media_dir / "desktop_vp9.webm"
    encode_from_frames(
        frames_c, 30, c_final,
        vcodec_args=[
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "36",
            "-deadline", "good", "-cpu-used", "4",
        ],
        out_w=1280, out_h=720,
        container_args=["-f", "webm"],
    )
    record(
        "desktop_vp9.webm",
        declared={
            "description": "Desktop MediaRecorder-like WebM, VP9 yuv420p, 1280x720@30, 15s",
            "container": "webm", "codec": "vp9", "pix_fmt": "yuv420p",
            "color_transfer": "bt709", "color_primaries": "bt709",
            "resolution": [1280, 720], "fps": 30, "duration_s": 15, "rotation_deg": 0,
            "hdr": False,
            "expected_validator": expected_verdict("webm", "vp9", 15.0, False, True),
            "expected_keyframe_count_range": [30, 64],
        },
    )

    # ---- (d) 6 s clip (too short) - MP4 H.264.
    frames_d = _fill_frames(rs.frames, int(6 * 30))
    d_final = media_dir / "too_short_6s.mp4"
    encode_from_frames(
        frames_d, 30, d_final,
        vcodec_args=["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30"],
        out_w=1280, out_h=720,
        container_args=["-movflags", "+faststart", "-f", "mp4"],
    )
    record(
        "too_short_6s.mp4",
        declared={
            "description": "Valid MP4 H.264 but only 6s (below the 10s minimum)",
            "container": "mp4", "codec": "h264", "pix_fmt": "yuv420p",
            "resolution": [1280, 720], "fps": 30, "duration_s": 6, "rotation_deg": 0,
            "hdr": False,
            "expected_validator": expected_verdict("mp4", "h264", 6.0, False, True),
            "expected_keyframe_count_range": None,
        },
    )

    # ---- (e) truncated copy of (b).
    e_final = media_dir / "truncated_android.mp4"
    data = b_final.read_bytes()
    e_final.write_bytes(data[: max(1, len(data) // 3)])  # keep ~first third -> unplayable tail
    record(
        "truncated_android.mp4",
        declared={
            "description": "First ~1/3 of android_h264.mp4 (truncated; undecodable)",
            "container": "mp4", "codec": "h264", "pix_fmt": "yuv420p",
            "resolution": [1920, 1080], "fps": 30, "duration_s": None, "rotation_deg": 0,
            "hdr": False,
            "expected_validator": expected_verdict("mp4", "h264", None, True, True),
            "expected_keyframe_count_range": None,
        },
        note="bytes are a prefix of android_h264.mp4",
    )

    # ---- (f) MP4 with a non-allowlisted codec (MPEG-4 Part 2).
    # Only the codec matters here (it must be rejected), so keep it small.
    frames_f = _fill_frames(rs.frames, int(12 * 30))
    f_final = media_dir / "nonallowed_mpeg4.mp4"
    encode_from_frames(
        frames_f, 30, f_final,
        vcodec_args=["-c:v", "mpeg4", "-vtag", "mp4v", "-pix_fmt", "yuv420p", "-q:v", "12"],
        out_w=1280, out_h=720,
        container_args=["-f", "mp4"],
    )
    record(
        "nonallowed_mpeg4.mp4",
        declared={
            "description": "MP4 carrying MPEG-4 Part 2 video (codec not in the H.264/HEVC/VP8/VP9 allowlist)",
            "container": "mp4", "codec": "mpeg4", "pix_fmt": "yuv420p",
            "resolution": [1280, 720], "fps": 30, "duration_s": 12, "rotation_deg": 0,
            "hdr": False,
            "expected_validator": expected_verdict("mp4", "mpeg4", 12.0, False, True),
            "expected_keyframe_count_range": None,
        },
    )

    # ---- (g) text file renamed .mp4.
    g_final = media_dir / "fake_text.mp4"
    g_final.write_text(
        "This is not a video. It is a plain text file whose name ends in .mp4 to "
        "test that the server validates magic bytes, not the extension.\n"
    )
    record(
        "fake_text.mp4",
        declared={
            "description": "Plain text renamed to .mp4 (magic-byte check must reject it)",
            "container": "none", "codec": "none", "pix_fmt": None,
            "resolution": None, "fps": None, "duration_s": None, "rotation_deg": 0,
            "hdr": False,
            "expected_validator": expected_verdict("none", "none", None, False, False),
            "expected_keyframe_count_range": None,
        },
    )

    # ---- Photos set: 24 JPEGs of the scene, mixed orientation + EXIF 6/8.
    photos_meta = build_photos(rs, media_dir)
    manifest["photos"] = photos_meta

    manifest["files"] = files_meta

    # Clean up work dir.
    shutil.rmtree(work, ignore_errors=True)

    # Total size guard.
    total = sum(f["size_bytes"] for f in files_meta) + sum(p["size_bytes"] for p in photos_meta)
    manifest["total_bytes"] = total
    manifest["total_mb"] = round(total / (1024 * 1024), 2)

    (media_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _looks_media(path: Path) -> bool:
    """Heuristic: a real media container (not our text/truncated edge cases need probing too)."""
    return path.suffix.lower() in {".mov", ".mp4", ".webm", ".mkv", ".jpg", ".jpeg"}


def _probe_nonmedia(path: Path) -> dict:
    out = _run([FFMPEG, "-hide_banner", "-i", str(path)])
    return {"raw_tail": out.strip().splitlines()[-6:], "codec": None, "duration_s": None,
            "rotation_deg": 0.0, "invalid": True}


EXIF_ORIENTATIONS = {
    6: "rotate 90 CW (EXIF 6)",
    8: "rotate 90 CCW (EXIF 8)",
    1: "normal (EXIF 1)",
}


def build_photos(rs: RenderedScene, media_dir: Path) -> list[dict]:
    """Write 24 JPEGs sampled from the orbit, some with EXIF orientation 6/8."""
    photos_dir = media_dir / "photos"
    photos_dir.mkdir(exist_ok=True)
    meta: list[dict] = []
    # Pick 24 evenly-spread frames deterministically.
    idxs = [int(round(i * (len(rs.frames) - 1) / 23)) for i in range(24)]
    for n, fi in enumerate(idxs):
        rgb = rs.frames[fi]
        # Mix portrait/landscape by cropping some to portrait.
        if n % 3 == 0:
            # portrait crop (centre)
            w = rgb.shape[1]
            cw = int(rgb.shape[0] * 9 / 16)
            x0 = (w - cw) // 2
            img = rgb[:, x0 : x0 + cw]
            is_portrait = True
        else:
            img = rgb
            is_portrait = False
        # Assign EXIF orientation: some 6, some 8, rest normal.
        if n % 4 == 1:
            orient = 6
        elif n % 4 == 2:
            orient = 8
        else:
            orient = 1
        name = f"photos/photo_{n:02d}.jpg"
        p = media_dir / name
        pil = Image.fromarray(img)
        exif_dict = {"0th": {piexif.ImageIFD.Orientation: orient}}
        exif_bytes = piexif.dump(exif_dict)
        pil.save(p, format="JPEG", quality=88, exif=exif_bytes)
        meta.append(
            {
                "name": name,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "source_frame_index": fi,
                "stored_resolution": [img.shape[1], img.shape[0]],
                "orientation_is_portrait_after_exif": is_portrait or orient in (6, 8),
                "exif_orientation": orient,
                "exif_orientation_meaning": EXIF_ORIENTATIONS[orient],
                "probe_orientation": _jpeg_exif_orientation(p),
            }
        )
    return meta


def _jpeg_exif_orientation(path: Path) -> int | None:
    """Re-read the EXIF orientation from the written JPEG (verify, never assume)."""
    try:
        ex = piexif.load(str(path))
        return ex["0th"].get(piexif.ImageIFD.Orientation)
    except Exception:
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parents[1] / "fixtures" / "media"
    ap.add_argument("--out", type=Path, default=default_out, help="output directory (fixtures/media/)")
    args = ap.parse_args(argv)
    manifest = build(args.out)
    print(f"Wrote {len(manifest['files'])} media files + {len(manifest['photos'])} photos "
          f"to {args.out} ({manifest['total_mb']} MB total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
