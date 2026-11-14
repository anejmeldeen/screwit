"""Render hard test scenes with EXACT ground truth, for X8 (does the real pipeline work on real objects?).

Every scene is built from triangle meshes with known dimensions, rendered by a small numpy software
rasterizer (no GPU, no display, no OpenGL) that does textured, diffuse and specular shading -- shiny
surfaces are a known depth-model weakness, so chrome-like materials with moving highlights are a
first-class case here. The ChArUco board (P2's single definition, copied constant-for-constant) is
laid flat in every scene so the real scale stage runs. Cameras follow an orbit or a hand-held arc.

For each scene we write, under an output root OUTSIDE git (default /local/home/brancli/screwit-hard):
  <scene>/video.mp4                 H.264 1080p30, 20-30 s, encoded with the imageio-ffmpeg binary
  <scene>/truth/meshes/*.ply        every object mesh in the SCAN frame (mm), one file per object
  <scene>/truth/cameras.json        per-frame world_to_camera (OpenCV, scan frame, mm) and K, in the
                                    DISPLAYED image frame (portrait rotation already applied)
  <scene>/truth/masks/<obj>/*.png   per-frame object label mask (255=object) in the displayed frame
  <scene>/truth/parts/<part>/*.png  per-frame part label mask (255=part)
  <scene>/truth/scene.json          key dimensions, materials, part list, frame/display metadata
and a small summary JSON is copied into hard/ by hard_eval.py (never the big outputs).

Ground-truth frame (SPEC 3.1, screwit.recon.scale.board_scan_points): origin at the board's printed
origin corner, X along the arrow, Y up the page, Z up out of the paper towards the cameras. The board
lies on Z=0; every object sits on it or stands beside it, all in millimetres.

Usage:
    python scripts/make_hard_scenes.py --scene gso_mug --out /local/home/brancli/screwit-hard
    python scripts/make_hard_scenes.py --scene under_sink --frames 40
    python scripts/make_hard_scenes.py --list
GSO models must be downloaded first with --fetch-gso (writes under <out>/_gso_cache/); see hard/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

# ChArUco board: P2's single definition (backend/screwit/recon/board.py). Copied as constants so this
# generator does not import across worktrees; the detector in scale.py sees exactly this board.
BOARD_NAME = "charuco_5x7_30mm"
BOARD_SQUARES_X = 5
BOARD_SQUARES_Y = 7
BOARD_SQUARE_MM = 30.0
BOARD_MARKER_RATIO = 0.75
BOARD_MARKER_MM = BOARD_SQUARE_MM * BOARD_MARKER_RATIO
BOARD_DICT_NAME = "DICT_4X4_50"
BOARD_W_MM = BOARD_SQUARES_X * BOARD_SQUARE_MM   # 150
BOARD_H_MM = BOARD_SQUARES_Y * BOARD_SQUARE_MM   # 210

DEFAULT_OUT = Path("/local/home/brancli/screwit-hard")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SEED = 20261114

# Render grid. 1080p is the encode target; render at a modest grid and let ffmpeg scale up (keeps
# the CPU rasterizer fast while the real H.264 decode path still runs on true 1080p30).
RENDER_W = 1280
RENDER_H = 720
ENCODE_W = 1920
ENCODE_H = 1080
FPS = 30


# =========================================================================== linear algebra


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def look_at_world_to_camera(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4x4 row-major world_to_camera, OpenCV axes (X right, Y down, Z forward into the scene)."""
    forward = _unit(target - eye)
    right = _unit(np.cross(forward, up))
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward], axis=0)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = -rot @ eye
    return m


def intrinsics(width: int, height: int, focal_px: float) -> np.ndarray:
    return np.array([[focal_px, 0.0, width / 2.0],
                     [0.0, focal_px, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# =========================================================================== mesh primitives
# Every primitive returns (vertices Nx3 mm, faces Mx3 int, uv Nx2 or None). Meshes live in the scan
# frame already (the builder places them). Normals are computed per face at render time.


@dataclass
class Mesh:
    name: str
    verts: np.ndarray
    faces: np.ndarray
    material: str                 # key into MATERIALS
    uv: np.ndarray | None = None  # per-vertex uv for a textured mesh (board only)
    part: str | None = None       # part label (defaults to name)
    is_board: bool = False

    def part_label(self) -> str:
        return self.part or self.name


def box(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = size / 2.0
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return v, f


def cylinder(center_xy: np.ndarray, z0: float, z1: float, radius: float, segments: int = 48,
             caps: bool = True) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = center_xy
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)
    bottom = np.column_stack([ring, np.full(segments, z0)])
    top = np.column_stack([ring, np.full(segments, z1)])
    v = np.concatenate([bottom, top], axis=0)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
    if caps:
        bc = len(v)
        tc = len(v) + 1
        v = np.concatenate([v, [[cx, cy, z0]], [[cx, cy, z1]]], axis=0)
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([bc, j, i])
            faces.append([tc, segments + i, segments + j])
    return v, np.array(faces, dtype=np.int64)


def tube(center_xy: np.ndarray, z0: float, z1: float, r_out: float, r_in: float,
         segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Hollow tube (annular cross section) with flat annular end rings -- a tailpiece/nut body."""
    cx, cy = center_xy
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    co, si = np.cos(ang), np.sin(ang)
    outer_b = np.column_stack([cx + r_out * co, cy + r_out * si, np.full(segments, z0)])
    outer_t = np.column_stack([cx + r_out * co, cy + r_out * si, np.full(segments, z1)])
    inner_b = np.column_stack([cx + r_in * co, cy + r_in * si, np.full(segments, z0)])
    inner_t = np.column_stack([cx + r_in * co, cy + r_in * si, np.full(segments, z1)])
    v = np.concatenate([outer_b, outer_t, inner_b, inner_t], axis=0)
    ob, ot, ib, it = 0, segments, 2 * segments, 3 * segments
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[ob + i, ob + j, ot + j], [ob + i, ot + j, ot + i]]          # outer wall
        faces += [[ib + i, it + j, ib + j], [ib + i, it + i, it + j]]          # inner wall
        faces += [[ot + i, ot + j, it + j], [ot + i, it + j, it + i]]          # top ring
        faces += [[ob + i, ib + j, ob + j], [ob + i, ib + i, ib + j]]          # bottom ring
    return v, np.array(faces, dtype=np.int64)


def torus(center: np.ndarray, major_r: float, minor_r: float, axis: str = "y",
          seg_u: int = 48, seg_v: int = 20, u_span: float = 2 * math.pi) -> tuple[np.ndarray, np.ndarray]:
    """Torus / bend. ``u_span`` < 2pi makes an arc (a P-trap J-bend). Axis is the symmetry axis."""
    cx, cy, cz = center
    us = np.linspace(0, u_span, seg_u, endpoint=(u_span < 2 * math.pi - 1e-6))
    vs = np.linspace(0, 2 * math.pi, seg_v, endpoint=False)
    verts = []
    for u in us:
        for vv in vs:
            rr = major_r + minor_r * math.cos(vv)
            if axis == "y":
                p = [rr * math.cos(u), minor_r * math.sin(vv), rr * math.sin(u)]
            elif axis == "x":
                p = [minor_r * math.sin(vv), rr * math.cos(u), rr * math.sin(u)]
            else:
                p = [rr * math.cos(u), rr * math.sin(u), minor_r * math.sin(vv)]
            verts.append([cx + p[0], cy + p[1], cz + p[2]])
    v = np.array(verts, dtype=np.float64)
    faces = []
    nu = len(us)
    for iu in range(nu - (1 if u_span < 2 * math.pi - 1e-6 else 0)):
        for iv in range(seg_v):
            a = (iu % nu) * seg_v + iv
            b = (iu % nu) * seg_v + (iv + 1) % seg_v
            c = ((iu + 1) % nu) * seg_v + iv
            d = ((iu + 1) % nu) * seg_v + (iv + 1) % seg_v
            faces += [[a, b, d], [a, d, c]]
    return v, np.array(faces, dtype=np.int64)


def plane_quad(origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = origin
    p = np.array([o, o + u, o + u + v, o + v], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return p, f


# =========================================================================== materials


@dataclass
class Material:
    albedo: tuple[float, float, float]
    diffuse: float = 0.8
    specular: float = 0.25
    shininess: float = 24.0


MATERIALS: dict[str, Material] = {
    "board": Material((1.0, 1.0, 1.0), diffuse=0.9, specular=0.05, shininess=8.0),
    "matte_gray": Material((200, 200, 195), diffuse=0.85, specular=0.1, shininess=12.0),
    "white_plastic": Material((235, 235, 232), diffuse=0.8, specular=0.35, shininess=30.0),
    "chrome": Material((205, 208, 212), diffuse=0.25, specular=0.9, shininess=120.0),
    "glossy_white": Material((240, 240, 240), diffuse=0.55, specular=0.7, shininess=90.0),
    "ceramic_mug": Material((210, 90, 70), diffuse=0.6, specular=0.5, shininess=60.0),
    "rubber_dark": Material((60, 60, 65), diffuse=0.85, specular=0.15, shininess=10.0),
    "wood_table": Material((150, 110, 70), diffuse=0.9, specular=0.1, shininess=10.0),
    "cardboard": Material((190, 165, 125), diffuse=0.9, specular=0.05, shininess=6.0),
    "brass": Material((190, 160, 90), diffuse=0.4, specular=0.7, shininess=80.0),
    "steel": Material((150, 155, 160), diffuse=0.4, specular=0.6, shininess=70.0),
}


# =========================================================================== board texture


def charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT_NAME))
    return cv2.aruco.CharucoBoard((BOARD_SQUARES_X, BOARD_SQUARES_Y), BOARD_SQUARE_MM,
                                  BOARD_MARKER_MM, dictionary)


def board_texture(px_per_mm: float = 16.0) -> np.ndarray:
    board = charuco_board()
    w_px = int(round(BOARD_W_MM * px_per_mm))
    h_px = int(round(BOARD_H_MM * px_per_mm))
    return board.generateImage((w_px, h_px), marginSize=0)


def board_scan_corners() -> tuple[np.ndarray, np.ndarray]:
    """ChArUco inner-corner ids and their positions in the SCAN frame (mm), matching
    screwit.recon.scale.board_scan_points: OpenCV board frame mapped so origin is the printed
    origin corner, X along the arrow, Y up the page, Z up. The texture is mapped to agree (no
    mirror), so a real detector decodes the markers and the corners reproject on the GT cameras."""
    board = charuco_board()
    cb = board.getChessboardCorners().astype(np.float64)  # OpenCV board frame: X right, Y down, Z 0
    scan = cb.copy()
    scan[:, 1] = BOARD_H_MM - cb[:, 1]  # Y up the page
    ids = np.arange(len(scan))
    return ids, scan


def board_mesh() -> Mesh:
    """Board on Z=0: origin at the printed origin corner, X along +X (arrow), Y up the page.

    UV maps the (un-mirrored) texture so markers decode: texture row 0 is the top of the printed
    sheet (OpenCV board y=0); the scan origin corner (world (0,0)) is the sheet's bottom-left, so it
    samples texture (u=0, v=1). world Y up the page = toward the top of the sheet = decreasing v."""
    o = np.array([0.0, 0.0, 0.0])
    u = np.array([BOARD_W_MM, 0.0, 0.0])
    v = np.array([0.0, BOARD_H_MM, 0.0])
    verts, faces = plane_quad(o, u, v)                 # [(0,0),(W,0),(W,H),(0,H)]
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    return Mesh("board", verts, faces, "board", uv=uv, is_board=True)


# =========================================================================== rasterizer


@dataclass
class Scene:
    meshes: list[Mesh]
    board_tex: np.ndarray
    ambient: float = 0.18
    lights: list[tuple[np.ndarray, float]] = field(default_factory=list)  # (world pos, intensity)
    background: tuple[int, int, int] = (40, 42, 48)

    def object_meshes(self) -> list[Mesh]:
        return [m for m in self.meshes if not m.is_board]


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.clip(ln, 1e-12, None)


def _shade_face(centroid: np.ndarray, normal: np.ndarray, mat: Material, eye: np.ndarray,
                scene: Scene, albedo: np.ndarray) -> np.ndarray:
    """Phong shading for one flat face: ambient + sum of diffuse+specular over the point lights."""
    view_dir = _unit(eye - centroid)
    # Two-sided: flip the normal toward the viewer so back-facing-but-visible faces still light.
    if np.dot(normal, view_dir) < 0:
        normal = -normal
    color = scene.ambient * albedo
    for lpos, lint in scene.lights:
        ldir = _unit(lpos - centroid)
        ndotl = max(0.0, float(np.dot(normal, ldir)))
        color = color + lint * mat.diffuse * ndotl * albedo
        if ndotl > 0:
            half = _unit(ldir + view_dir)
            spec = max(0.0, float(np.dot(normal, half))) ** mat.shininess
            color = color + lint * mat.specular * spec * np.array([255.0, 255.0, 255.0])
    return np.clip(color, 0, 255)


def _project(verts: np.ndarray, w2c: np.ndarray, k: np.ndarray):
    homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = (k @ (cam / z[:, None]).T).T[:, :2]
    return pix, z, cam


@dataclass
class RenderOut:
    rgb: np.ndarray            # (H,W,3) uint8
    depth: np.ndarray          # (H,W) float64 camera z, inf where empty
    obj_id: np.ndarray         # (H,W) int32, -1 empty, else index into id_names
    part_id: np.ndarray        # (H,W) int32
    id_names: list[str]
    part_names: list[str]


def render(scene: Scene, w2c: np.ndarray, k: np.ndarray, eye: np.ndarray,
           w: int = RENDER_W, h: int = RENDER_H) -> RenderOut:
    img = np.empty((h, w, 3), np.float64)
    img[:] = scene.background
    zbuf = np.full((h, w), np.inf)
    obj_buf = np.full((h, w), -1, np.int32)
    part_buf = np.full((h, w), -1, np.int32)

    obj_names = [m.name for m in scene.object_meshes()]
    part_names: list[str] = []
    obj_index = {m.name: i for i, m in enumerate(scene.object_meshes())}

    for mesh in scene.meshes:
        pix, z, _cam = _project(mesh.verts, w2c, k)
        normals = _face_normals(mesh.verts, mesh.faces)
        centroids = mesh.verts[mesh.faces].mean(axis=1)
        oid = obj_index.get(mesh.name, -1)
        if mesh.part_label() not in part_names and not mesh.is_board:
            part_names.append(mesh.part_label())
        pid = part_names.index(mesh.part_label()) if not mesh.is_board else -1
        mat = MATERIALS[mesh.material]
        for fi, face in enumerate(mesh.faces):
            tri_pix = pix[face]
            tri_z = z[face]
            if not np.all(np.isfinite(tri_pix)) or np.any(tri_z <= 1e-6):
                continue
            if mesh.is_board:
                _raster_tex(img, zbuf, obj_buf, part_buf, tri_pix, tri_z, mesh.uv[face],
                            scene.board_tex, oid, pid)
            else:
                albedo = np.array(mat.albedo, dtype=np.float64)
                col = _shade_face(centroids[fi], normals[fi], mat, eye, scene, albedo)
                _raster_flat(img, zbuf, obj_buf, part_buf, tri_pix, tri_z, col, oid, pid)

    return RenderOut(np.clip(img, 0, 255).astype(np.uint8), zbuf, obj_buf, part_buf,
                     obj_names, part_names)


def _bbox(pts: np.ndarray, w: int, h: int):
    min_x = max(int(math.floor(pts[:, 0].min())), 0)
    max_x = min(int(math.ceil(pts[:, 0].max())), w - 1)
    min_y = max(int(math.floor(pts[:, 1].min())), 0)
    max_y = min(int(math.ceil(pts[:, 1].max())), h - 1)
    return min_x, max_x, min_y, max_y


def _bary(pts: np.ndarray, min_x, max_x, min_y, max_y):
    yy, xx = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
    xf, yf = xx + 0.5, yy + 0.5
    (x0, y0), (x1, y1), (x2, y2) = pts
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return None
    a = ((y1 - y2) * (xf - x2) + (x2 - x1) * (yf - y2)) / denom
    b = ((y2 - y0) * (xf - x2) + (x0 - x2) * (yf - y2)) / denom
    c = 1.0 - a - b
    return a, b, c


def _raster_flat(img, zbuf, obj_buf, part_buf, pts, zs, color, oid, pid):
    h, w = img.shape[:2]
    min_x, max_x, min_y, max_y = _bbox(pts, w, h)
    if max_x < min_x or max_y < min_y:
        return
    bc = _bary(pts, min_x, max_x, min_y, max_y)
    if bc is None:
        return
    a, b, c = bc
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    if not inside.any():
        return
    depth = a * zs[0] + b * zs[1] + c * zs[2]
    rz = zbuf[min_y:max_y + 1, min_x:max_x + 1]
    closer = inside & (depth < rz)
    if not closer.any():
        return
    img[min_y:max_y + 1, min_x:max_x + 1][closer] = color
    rz[closer] = depth[closer]
    obj_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = oid
    part_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = pid


def _raster_tex(img, zbuf, obj_buf, part_buf, pts, zs, uvs, tex, oid, pid):
    h, w = img.shape[:2]
    th, tw = tex.shape[:2]
    min_x, max_x, min_y, max_y = _bbox(pts, w, h)
    if max_x < min_x or max_y < min_y:
        return
    bc = _bary(pts, min_x, max_x, min_y, max_y)
    if bc is None:
        return
    a, b, c = bc
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    if not inside.any():
        return
    depth = a * zs[0] + b * zs[1] + c * zs[2]
    rz = zbuf[min_y:max_y + 1, min_x:max_x + 1]
    closer = inside & (depth < rz)
    if not closer.any():
        return
    inv_z = a / zs[0] + b / zs[1] + c / zs[2]
    u = (a * uvs[0, 0] / zs[0] + b * uvs[1, 0] / zs[1] + c * uvs[2, 0] / zs[2]) / inv_z
    v = (a * uvs[0, 1] / zs[0] + b * uvs[1, 1] / zs[1] + c * uvs[2, 1] / zs[2]) / inv_z
    fu = np.clip(u * (tw - 1), 0, tw - 1)
    fv = np.clip(v * (th - 1), 0, th - 1)
    x0 = np.floor(fu).astype(np.int32)
    y0 = np.floor(fv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tw - 1)
    y1 = np.clip(y0 + 1, 0, th - 1)
    wx, wy = fu - x0, fv - y0
    s = (tex[y0, x0] * (1 - wx) * (1 - wy) + tex[y0, x1] * wx * (1 - wy)
         + tex[y1, x0] * (1 - wx) * wy + tex[y1, x1] * wx * wy)
    gray = np.stack([s, s, s], axis=-1)
    reg = img[min_y:max_y + 1, min_x:max_x + 1]
    reg[closer] = gray[closer]
    rz[closer] = depth[closer]
    obj_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = oid
    part_buf[min_y:max_y + 1, min_x:max_x + 1][closer] = pid


# =========================================================================== cameras


def orbit_cameras(n: int, target: np.ndarray, radius: float, elev_min: float, elev_max: float):
    for i in range(n):
        az = 2 * math.pi * i / n
        elev = math.radians(elev_min + (elev_max - elev_min) * 0.5 * (1 - math.cos(2 * az)))
        eye = target + radius * np.array([math.cos(elev) * math.cos(az),
                                          math.cos(elev) * math.sin(az),
                                          math.sin(elev)])
        yield eye, look_at_world_to_camera(eye, target, np.array([0.0, 0.0, 1.0]))


def arc_cameras(n: int, target: np.ndarray, radius: float, az_min: float, az_max: float,
                elev_min: float, elev_max: float):
    """Hand-held arc in FRONT of the scene (what a person can reach under a cabinet). Azimuth 0 is
    straight in front (-Y of the target); +/- az sweeps left/right around that front hemisphere."""
    for i in range(n):
        f = i / max(1, n - 1)
        az = math.radians(az_min + (az_max - az_min) * f)
        elev = math.radians(elev_min + (elev_max - elev_min) * 0.5 * (1 - math.cos(2 * math.pi * f)))
        eye = target + radius * np.array([math.cos(elev) * math.sin(az),
                                          -math.cos(elev) * math.cos(az),
                                          math.sin(elev)])
        yield eye, look_at_world_to_camera(eye, target, np.array([0.0, 0.0, 1.0]))


# =========================================================================== scene builders


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _tabletop(size_mm: float = 500.0) -> Mesh:
    """A textured-looking wood tabletop as a big quad just below Z=0 (board sits on it)."""
    half = size_mm / 2.0
    o = np.array([BOARD_W_MM / 2 - half, BOARD_H_MM / 2 - half, -0.5])
    u = np.array([size_mm, 0.0, 0.0])
    v = np.array([0.0, size_mm, 0.0])
    verts, faces = plane_quad(o, u, v)
    return Mesh("tabletop", verts, faces, "wood_table", part="tabletop")


def build_gso_primitive(kind: str) -> tuple[Scene, dict]:
    """A GSO-like tabletop scene built from primitives standing in for a downloaded model, when the
    real GSO mesh is not provided. The dimensions are still exact ground truth."""
    meshes = [board_mesh(), _tabletop()]
    tex = board_texture()
    dims: dict[str, float] = {}
    parts: list[dict] = []
    cx = BOARD_W_MM + 90.0
    cy = BOARD_H_MM / 2.0
    if kind == "mug":
        h, r_out, r_in, wall_top = 95.0, 40.0, 34.0, 90.0
        bv, bf = cylinder((cx, cy), 0.0, 6.0, r_out)                    # base
        wv, wf = tube((cx, cy), 6.0, h, r_out, r_in)                    # body wall
        meshes.append(Mesh("mug_body", np.concatenate([bv, wv]),
                           np.concatenate([bf, wf + len(bv)]), "ceramic_mug", part="body"))
        hv, hf = torus(np.array([cx + r_out, cy, wall_top / 2 + 10]), 26.0, 6.0, axis="y",
                       u_span=math.pi)
        meshes.append(Mesh("mug_handle", hv, hf, "ceramic_mug", part="handle"))
        dims = {"mug_height_mm": h, "mug_outer_diameter_mm": 2 * r_out, "handle_tube_diameter_mm": 12.0}
        parts = [{"part": "body", "feature": "cylinder", "outer_diameter_mm": 2 * r_out,
                  "height_mm": h}, {"part": "handle", "feature": "cylinder",
                                    "tube_diameter_mm": 12.0}]
    elif kind == "shoe":
        v, f = box(np.array([cx, cy, 45.0]), np.array([110.0, 260.0, 90.0]))
        meshes.append(Mesh("shoe", v, f, "rubber_dark", part="shoe"))
        dims = {"shoe_length_mm": 260.0, "shoe_width_mm": 110.0, "shoe_height_mm": 90.0}
        parts = [{"part": "shoe", "feature": "box", "length_mm": 260.0, "width_mm": 110.0}]
    elif kind == "thin_tool":  # a kitchen whisk-like thin-handled tool
        hv, hf = cylinder((cx, cy), 0.0, 10.0, 12.0)                    # handle lying flat? stand it
        sv, sf = cylinder((cx, cy), 10.0, 210.0, 7.0)                   # thin shaft
        meshes.append(Mesh("tool_handle", hv, hf, "steel", part="handle"))
        meshes.append(Mesh("tool_shaft", sv, sf, "steel", part="shaft"))
        dims = {"tool_shaft_diameter_mm": 14.0, "tool_height_mm": 210.0, "handle_diameter_mm": 24.0}
        parts = [{"part": "shaft", "feature": "cylinder", "diameter_mm": 14.0},
                 {"part": "handle", "feature": "cylinder", "diameter_mm": 24.0}]
    elif kind == "glossy_bottle":
        bv, bf = cylinder((cx, cy), 0.0, 150.0, 35.0)
        nv, nf = cylinder((cx, cy), 150.0, 190.0, 14.0)
        meshes.append(Mesh("bottle_body", bv, bf, "glossy_white", part="body"))
        meshes.append(Mesh("bottle_neck", nv, nf, "glossy_white", part="neck"))
        dims = {"bottle_diameter_mm": 70.0, "bottle_height_mm": 190.0, "neck_diameter_mm": 28.0}
        parts = [{"part": "body", "feature": "cylinder", "diameter_mm": 70.0},
                 {"part": "neck", "feature": "cylinder", "diameter_mm": 28.0}]
    else:
        raise ValueError(f"unknown gso primitive {kind}")
    scene = Scene(meshes, tex, lights=[(np.array([-200.0, -150.0, 400.0]), 0.7),
                                       (np.array([300.0, 250.0, 350.0]), 0.5)])
    return scene, {"kind": f"gso_primitive_{kind}", "dimensions_mm": dims, "parts": parts,
                   "note": "primitive stand-in for a Google Scanned Object (exact dims); replace "
                           "with a real GSO mesh via --fetch-gso for a true-geometry test"}


def build_under_sink(n_clutter: bool = True) -> tuple[Scene, dict]:
    """The user's main case: a cabinet interior with plumbing, chrome + white plastic, dim light.

    Layout: the board lies on the cabinet floor (Z=0) centred under the drain. The tailpiece,
    slip nuts, P-trap J-bend, basin underside, shutoff valve and supply line stand above and behind
    it; a cardboard box and a bottle are clutter. The camera arcs in front at roughly board height,
    so the board and the lower plumbing share the frame -- what a person crouching with a phone sees.
    """
    meshes = [board_mesh()]
    tex = board_texture()
    bx = BOARD_W_MM / 2          # drain axis over the board centre in X
    by = BOARD_H_MM / 2          # and in Y
    depth = 320.0
    cab_w = 460.0
    cab_h = 360.0
    x0 = bx - cab_w / 2
    # Cabinet: floor (just below Z=0 so the board sits on it), back wall, two sides.
    floorv, floorf = plane_quad(np.array([x0, by - 170, -1.0]),
                                np.array([cab_w, 0, 0]), np.array([0, depth, 0]))
    meshes.append(Mesh("cabinet_floor", floorv, floorf, "cardboard", part="cabinet"))
    backv, backf = plane_quad(np.array([x0, by + 150, -1.0]),
                              np.array([cab_w, 0, 0]), np.array([0, 0, cab_h]))
    meshes.append(Mesh("cabinet_back", backv, backf, "cardboard", part="cabinet"))

    # Basin underside high up.
    basinv, basinf = plane_quad(np.array([bx - 150, by - 120, 300.0]),
                                np.array([300, 0, 0]), np.array([0, 260, 0]))
    meshes.append(Mesh("basin_underside", basinv, basinf, "white_plastic", part="basin"))

    tp_od, tp_id = 38.1, 34.0        # 1-1/2" tubular
    # Tailpiece hangs from the basin drain down to the P-trap.
    tpv, tpf = tube((bx, by), 140.0, 300.0, tp_od / 2, tp_id / 2)
    meshes.append(Mesh("tailpiece", tpv, tpf, "chrome", part="tailpiece"))
    for kk, z in enumerate((140.0, 108.0)):
        nv, nf = tube((bx, by), z, z + 22.0, 24.0, tp_od / 2)
        meshes.append(Mesh(f"slip_nut_{kk+1}", nv, nf, "chrome", part=f"slip_nut_{kk+1}"))
    wv, wf = tube((bx, by), 104.0, 107.0, tp_od / 2 + 2, tp_od / 2)
    meshes.append(Mesh("slip_washer", wv, wf, "white_plastic", part="slip_washer"))
    # P-trap J-bend: chrome arc in the Y-Z plane joining the tailpiece bottom to a short arm.
    jv, jf = torus(np.array([bx, by, 55.0]), 42.0, tp_od / 2, axis="x", u_span=math.pi)
    meshes.append(Mesh("p_trap", jv, jf, "chrome", part="p_trap"))

    # Shutoff valve + supply line against the back wall.
    valv, valf = box(np.array([bx + 150, by + 130, 80.0]), np.array([40.0, 40.0, 40.0]))
    meshes.append(Mesh("shutoff_valve", valv, valf, "brass", part="valve"))
    supv, supf = cylinder((bx + 150, by + 130), 80.0, 280.0, 6.0)
    meshes.append(Mesh("supply_line", supv, supf, "steel", part="supply_line"))

    if n_clutter:
        cbx, cbf = box(np.array([x0 + 80, by - 120, 55.0]), np.array([110.0, 90.0, 110.0]))
        meshes.append(Mesh("clutter_box", cbx, cbf, "cardboard", part="clutter"))
        bov, bof = cylinder((x0 + cab_w - 70, by - 110), 0.0, 150.0, 30.0)
        meshes.append(Mesh("clutter_bottle", bov, bof, "glossy_white", part="clutter"))

    scene = Scene(meshes, tex, ambient=0.22,
                  lights=[(np.array([bx - 60, by - 280, 230.0]), 0.6),    # phone light from front
                          (np.array([bx + 120, by - 180, 260.0]), 0.3),
                          (np.array([bx, by - 120, 90.0]), 0.5)],         # fill that reaches the board
                  background=(14, 14, 18))
    dims = {"tailpiece_outer_diameter_mm": tp_od, "slip_nut_width_mm": 48.0,
            "p_trap_tube_diameter_mm": tp_od, "p_trap_bend_radius_mm": 42.0,
            "supply_line_diameter_mm": 12.0, "valve_width_mm": 40.0}
    parts = [{"part": "tailpiece", "feature": "cylinder", "outer_diameter_mm": tp_od},
             {"part": "slip_nut_1", "feature": "cylinder", "outer_diameter_mm": 48.0},
             {"part": "slip_nut_2", "feature": "cylinder", "outer_diameter_mm": 48.0},
             {"part": "p_trap", "feature": "cylinder", "tube_diameter_mm": tp_od},
             {"part": "supply_line", "feature": "cylinder", "diameter_mm": 12.0}]
    return scene, {"kind": "under_sink", "dimensions_mm": dims, "parts": parts,
                   "materials": ["chrome", "white_plastic"], "camera": "arc"}


def build_knob_panel() -> tuple[Scene, dict]:
    """A knob on a panel with a D-shaft behind it: a thin feature for measurement."""
    meshes = [board_mesh()]
    tex = board_texture()
    px = BOARD_W_MM + 70.0
    py = BOARD_H_MM / 2.0
    # Panel standing vertically behind the knob (a thin plate).
    panelv, panelf = box(np.array([px, py + 20, 60.0]), np.array([160.0, 6.0, 120.0]))
    meshes.append(Mesh("panel", panelv, panelf, "matte_gray", part="panel"))
    # D-shaft: a cylinder with one flat, poking forward from the panel (-y toward the camera),
    # built as a cylinder whose +x side is clipped to a flat.
    shaft_r = 6.0
    seg = 48
    ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    co, si = np.cos(ang), np.sin(ang)
    flat_x = shaft_r * 0.5  # flat cut at x = +flat_x
    xs = px + shaft_r * co
    ys_shaft = py + shaft_r * si
    xs = np.minimum(xs, px + flat_x)  # cut one flat
    y0s, y1s = py + 17.0, py - 5.0     # shaft extends forward (toward -y), behind the knob
    base = np.column_stack([xs, np.full(seg, y0s), ys_shaft])
    tip = np.column_stack([xs, np.full(seg, y1s), ys_shaft])
    sv = np.concatenate([base, tip])
    sf = []
    for i in range(seg):
        j = (i + 1) % seg
        sf += [[i, j, seg + j], [i, seg + j, seg + i]]
    meshes.append(Mesh("d_shaft", sv, np.array(sf, np.int64), "steel", part="d_shaft"))
    # Knob: a short wide cylinder on the front of the shaft (its axis along -y).
    kseg = 48
    kr, kdepth = 22.0, 24.0
    ka = np.linspace(0, 2 * math.pi, kseg, endpoint=False)
    kring_x = px + kr * np.cos(ka)
    kring_z = 60.0 + kr * np.sin(ka)
    kb = np.column_stack([kring_x, np.full(kseg, py - 5.0), kring_z])
    kt = np.column_stack([kring_x, np.full(kseg, py - 5.0 - kdepth), kring_z])
    kv = np.concatenate([kb, kt, [[px, py - 5.0, 60.0]], [[px, py - 5.0 - kdepth, 60.0]]])
    kf = []
    for i in range(kseg):
        j = (i + 1) % kseg
        kf += [[i, j, kseg + j], [i, kseg + j, kseg + i]]
        kf += [[2 * kseg, j, i], [2 * kseg + 1, kseg + i, kseg + j]]
    meshes.append(Mesh("knob", kv, np.array(kf, np.int64), "rubber_dark", part="knob"))
    scene = Scene(meshes, tex, lights=[(np.array([-100.0, -200.0, 300.0]), 0.7),
                                       (np.array([250.0, -50.0, 250.0]), 0.4)])
    dims = {"knob_diameter_mm": 2 * kr, "knob_depth_mm": kdepth, "shaft_diameter_mm": 2 * shaft_r,
            "shaft_flat_depth_mm": shaft_r - flat_x}
    parts = [{"part": "knob", "feature": "cylinder", "diameter_mm": 2 * kr},
             {"part": "d_shaft", "feature": "d_shaft", "diameter_mm": 2 * shaft_r,
              "flat_depth_mm": shaft_r - flat_x}]
    return scene, {"kind": "knob_panel", "dimensions_mm": dims, "parts": parts}


SCENES = {
    "gso_mug": lambda: build_gso_primitive("mug"),
    "gso_shoe": lambda: build_gso_primitive("shoe"),
    "gso_thin_tool": lambda: build_gso_primitive("thin_tool"),
    "gso_glossy_bottle": lambda: build_gso_primitive("glossy_bottle"),
    "under_sink": build_under_sink,
    "knob_panel": build_knob_panel,
    # Portrait variant of the under-sink case: same scene, rendered upright in portrait and stored
    # the way phones store portrait video (sideways samples + a 90deg display matrix).
    "under_sink_portrait": build_under_sink,
}
PORTRAIT_SCENES = frozenset({"under_sink_portrait"})


# --------------------------------------------------------------------------- Google Scanned Objects
# Real GSO meshes (CC BY 4.0) downloaded from Gazebo Fuel, loaded as exact geometry. A picked model
# is scaled so its given real dimension matches, sat on the board, and rendered with a representative
# material. The geometry is real (which is what a reconstruction test needs); the per-model material
# is a stand-in because the flat rasterizer does not sample the OBJ texture.

GSO_MODELS: dict[str, dict] = {
    # name on Fuel -> how to place it. real_mm sets the scale (longest horizontal extent -> real_mm),
    # upright rotates the model so its natural up is +Z, material picks a shading response.
    "gso_real_mug": {"fuel": "Cole_Hardware_Mug_Classic_Blue", "real_mm": 95.0,
                     "material": "ceramic_mug", "label": "mug with a handle"},
    "gso_real_shoe": {"fuel": "Womens_Teva_Capistrano_Bootie", "real_mm": 260.0,
                      "material": "rubber_dark", "label": "shoe"},
    "gso_real_glass": {"fuel": "Ecoforms_Plant_Saucer_SQ8COR", "real_mm": 205.0,
                       "material": "glossy_white", "label": "glossy saucer"},
    "gso_real_toy": {"fuel": "Vtech_Roll_Learn_Turtle", "real_mm": 230.0,
                     "material": "white_plastic", "label": "kitchen-tool-like toy"},
}
FUEL_BASE = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"


def fetch_gso(name: str, cache: Path) -> Path | None:
    """Download one GSO model's OBJ (and texture) from Gazebo Fuel into ``cache/<fuel>/``. Returns
    the OBJ path, or None on any failure (the caller falls back to a primitive)."""
    import urllib.request

    spec = GSO_MODELS[name]
    fuel = spec["fuel"]
    dst = cache / fuel
    obj = dst / "meshes" / "model.obj"
    if obj.exists():
        return obj
    (dst / "meshes").mkdir(parents=True, exist_ok=True)
    (dst / "materials" / "textures").mkdir(parents=True, exist_ok=True)
    files = ["meshes/model.obj", "meshes/model.mtl", "materials/textures/texture.png"]
    for rel in files:
        url = f"{FUEL_BASE}/{fuel}/1/files/{rel}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https Fuel host
                (dst / rel).write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            if rel.endswith("model.obj"):
                print(f"fetch_gso {name}: {e}", file=__import__("sys").stderr)
                return None
    return obj if obj.exists() else None


def build_gso_real(name: str, obj_path: Path) -> tuple[Scene, dict]:
    """Build a tabletop scene from a REAL downloaded GSO mesh (exact geometry), scaled to real_mm."""
    import trimesh

    spec = GSO_MODELS[name]
    tm = trimesh.load(obj_path, force="mesh")
    v = np.asarray(tm.vertices, dtype=np.float64)
    # Fuel GSO meshes are in metres; convert to mm and set the model's natural up (+Y in many GSO
    # OBJs) to +Z by swapping axes, then scale so the longest horizontal extent is real_mm.
    v = v[:, [0, 2, 1]] * 1000.0            # metres -> mm, Y-up OBJ -> Z-up scan frame
    v -= v.min(axis=0)                       # sit base at Z=0, corner at origin-ish
    ext = v.max(axis=0) - v.min(axis=0)
    horiz = max(ext[0], ext[1])
    s = spec["real_mm"] / horiz if horiz > 1e-6 else 1.0
    v = v * s
    ext = v.max(axis=0) - v.min(axis=0)
    cx = BOARD_W_MM + 40.0 + ext[0] / 2.0
    cy = BOARD_H_MM / 2.0
    v[:, 0] += cx - (v[:, 0].min() + v[:, 0].max()) / 2.0
    v[:, 1] += cy - (v[:, 1].min() + v[:, 1].max()) / 2.0
    v[:, 2] -= v[:, 2].min()
    meshes = [board_mesh(), _tabletop()]
    meshes.append(Mesh(name, v, np.asarray(tm.faces, dtype=np.int64), spec["material"],
                       part=spec["label"].split()[0]))
    scene = Scene(meshes, board_texture(),
                  lights=[(np.array([-200.0, -150.0, 400.0]), 0.7),
                          (np.array([300.0, 250.0, 350.0]), 0.5)])
    dims = {f"{name}_length_mm": float(ext[0]), f"{name}_width_mm": float(ext[1]),
            f"{name}_height_mm": float(v[:, 2].max()), "scaled_longest_horizontal_mm": spec["real_mm"]}
    parts = [{"part": spec["label"].split()[0], "feature": "box",
              "length_mm": float(max(ext[0], ext[1])), "height_mm": float(v[:, 2].max())}]
    return scene, {"kind": f"gso_real_{name}", "dimensions_mm": dims, "parts": parts,
                   "gso_fuel_model": spec["fuel"], "license": "CC BY 4.0",
                   "attribution": f"Google Research, '{spec['fuel']}', Gazebo Fuel, CC BY 4.0",
                   "triangles": int(len(tm.faces)),
                   "note": "real GSO geometry (metres->mm, scaled to real size); material is a "
                           "stand-in (flat rasterizer does not sample the OBJ texture)"}


def scene_camera_path(kind: str, n: int, portrait: bool = False):
    if kind == "under_sink":
        if portrait:
            # Tall frame: raise the target up the plumbing column and pull back a little so the
            # drain, tailpiece, P-trap and the board below all fit the vertical frame.
            target = np.array([BOARD_W_MM / 2, BOARD_H_MM / 2, 140.0])
            return list(arc_cameras(n, target, 470.0, -40.0, 40.0, 10.0, 34.0))
        target = np.array([BOARD_W_MM / 2, BOARD_H_MM / 2, 70.0])
        return list(arc_cameras(n, target, 420.0, -42.0, 42.0, 14.0, 38.0))
    if kind == "knob_panel":
        target = np.array([BOARD_W_MM + 70.0, BOARD_H_MM / 2 - 5.0, 70.0])
        return list(arc_cameras(n, target, 380.0, -40.0, 40.0, 10.0, 35.0))
    target = np.array([BOARD_W_MM / 2 + 40, BOARD_H_MM / 2, 60.0])
    return list(orbit_cameras(n, target, 520.0, 28.0, 42.0))


# =========================================================================== ground truth + encode


def _export_mesh_ply(mesh: Mesh, path: Path) -> None:
    import trimesh

    tm = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces, process=False)
    tm.export(path)


def board_detection_stats(frames: list[np.ndarray]) -> dict:
    """How many rendered frames a real ChArUco detector actually sees the board in (realism check).
    Runs on the already-rendered frames so the scene is not rasterized twice."""
    detector = cv2.aruco.CharucoDetector(charuco_board())
    detected = 0
    per = []
    for i, rgb in enumerate(frames):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cc, cids, _, _ = detector.detectBoard(gray)
        n = 0 if cc is None else len(cc)
        if n >= 8:
            detected += 1
        per.append({"index": i, "corners": int(n)})
    return {"frames_with_8plus_corners": detected, "total_frames": len(frames), "per_frame": per}


def _encode(frames: list[np.ndarray], out_path: Path, portrait: bool) -> None:
    """Pipe raw RGB frames to the imageio-ffmpeg binary and encode H.264 1080p30.

    Portrait: store the upright frames as LANDSCAPE samples (rotate -90), encode, then add a +90
    display matrix in a stream-copy pass -- exactly how a phone stores portrait video. The decoder
    (screwit.recon.frames.extract_video_frames) applies the display matrix and recovers the upright
    frame; the generate() caller verifies that before trusting the file.
    """
    h, w = frames[0].shape[:2]
    scale = f"scale={ENCODE_W}:{ENCODE_H}:flags=bicubic"
    store = frames
    if portrait:
        # The upright frame is portrait (tall). Store it as LANDSCAPE samples: rotate -90 so samples
        # are wide, scale to 1920x1080 landscape, then a +90 display matrix rotates it back to a tall
        # portrait display -- exactly how a phone stores portrait video.
        store = [np.rot90(f, k=-1).copy() for f in frames]
        h, w = store[0].shape[:2]
        scale = f"scale={ENCODE_W}:{ENCODE_H}:flags=bicubic"   # landscape samples
    tmp_out = out_path.with_name(out_path.stem + ".samples.mp4") if portrait else out_path
    cmd = [FFMPEG, "-hide_banner", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{w}x{h}", "-framerate", str(FPS), "-i", "-", "-vf", scale,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-g", "30",
           str(tmp_out)]
    raw = b"".join(np.ascontiguousarray(f).tobytes() for f in store)
    proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{proc.stdout.decode('utf-8', 'replace')[-1500:]}")
    if portrait:
        # Stream-copy, writing a +90 display matrix as an INPUT option (before -i), as phones do.
        copy_cmd = [FFMPEG, "-hide_banner", "-y", "-display_rotation:v:0", "90",
                    "-i", str(tmp_out), "-c", "copy", str(out_path)]
        cp = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if cp.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"display-rotation copy failed:\n"
                               f"{cp.stdout.decode('utf-8', 'replace')[-1500:]}")
        tmp_out.unlink(missing_ok=True)


def _verify_portrait_decode(video_path: Path, upright_frame: np.ndarray) -> dict:
    """Decode the stored video through the REAL pipeline frame extractor and confirm it comes back
    upright (portrait aspect, same orientation as the rendered frame), matching the brief's check."""
    import sys as _sys
    import tempfile

    be = str(Path(__file__).resolve().parents[1] / "backend")
    if be not in _sys.path:
        _sys.path.insert(0, be)
    from screwit.recon.frames import extract_video_frames, probe_video

    pr = probe_video(video_path)
    with tempfile.TemporaryDirectory() as td:
        frames_dir = Path(td) / "frames"
        idx = extract_video_frames(video_path, frames_dir)
        first = np.asarray(Image.open(frames_dir / idx.frames[0].frame).convert("RGB"))
    up_h, up_w = upright_frame.shape[:2]
    decoded_portrait = first.shape[0] > first.shape[1]
    want_portrait = up_h > up_w
    return {"probe_display_wh": [pr.width, pr.height], "probe_rotation_handled": pr.height > pr.width,
            "decoded_frame_wh": [int(first.shape[1]), int(first.shape[0])],
            "decoded_upright": bool(decoded_portrait == want_portrait),
            "kept": idx.kept}


def generate(scene_name: str, out_root: Path, n_frames: int, portrait: bool = False,
             scene_override: tuple[Scene, dict] | None = None) -> dict:
    scene, meta = scene_override if scene_override is not None else SCENES[scene_name]()
    kind = meta["kind"]
    cams = scene_camera_path(kind, n_frames, portrait=portrait)
    # Portrait: render a tall (portrait) grid, as a phone held vertically captures. The displayed
    # frame the pipeline decodes is this upright portrait frame; the file stores it as landscape
    # samples + a 90deg display matrix (see _encode).
    rw, rh = (RENDER_H, RENDER_W) if portrait else (RENDER_W, RENDER_H)
    k = intrinsics(rw, rh, 1150.0)

    scene_dir = out_root / scene_name
    truth = scene_dir / "truth"
    (truth / "meshes").mkdir(parents=True, exist_ok=True)
    (truth / "masks").mkdir(parents=True, exist_ok=True)
    (truth / "parts").mkdir(parents=True, exist_ok=True)

    for mesh in scene.object_meshes():
        _export_mesh_ply(mesh, truth / "meshes" / f"{mesh.name}.ply")

    frames: list[np.ndarray] = []
    cam_entries: list[dict] = []
    obj_names = [m.name for m in scene.object_meshes()]
    part_names_all: list[str] = []
    for i, (eye, w2c) in enumerate(cams):
        out = render(scene, w2c, k, eye, w=rw, h=rh)
        for p in out.part_names:
            if p not in part_names_all:
                part_names_all.append(p)
        rgb = out.rgb
        obj_buf, part_buf = out.obj_id, out.part_id
        disp_w2c = w2c
        disp_k = k.copy()
        name = f"{i:04d}.png"
        frames.append(rgb)
        for oi, oname in enumerate(obj_names):
            mdir = truth / "masks" / oname
            mdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(((obj_buf == oi) * 255).astype(np.uint8)).save(mdir / name)
        for pi, pname in enumerate(out.part_names):
            pdir = truth / "parts" / pname
            pdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(((part_buf == pi) * 255).astype(np.uint8)).save(pdir / name)
        cam_entries.append({"index": i, "frame": name,
                            "world_to_camera": [[float(x) for x in row] for row in disp_w2c],
                            "K": [[float(x) for x in row] for row in disp_k],
                            "width": rw, "height": rh})

    duration = n_frames / FPS
    if duration < 20:
        frames = [frames[i % len(frames)] for i in range(int(20 * FPS))]
    video_path = scene_dir / "video.mp4"
    _encode(frames, video_path, portrait)

    portrait_check = None
    if portrait:
        portrait_check = _verify_portrait_decode(video_path, frames[0])

    det = board_detection_stats(frames[:n_frames])
    scene_json = {
        "scene": scene_name, "kind": kind, "portrait": portrait,
        "render_grid": [rw, rh], "encode": [ENCODE_W, ENCODE_H], "fps": FPS,
        "n_frames_rendered": n_frames, "video_duration_s": round(len(frames) / FPS, 2),
        "frame_note": ("displayed frame = upright; stored as landscape samples + 90deg display "
                       "matrix (phone portrait)" if portrait else "landscape, no rotation"),
        "portrait_decode_check": portrait_check,
        "scan_frame": "board origin corner, X=arrow, Y=up page, Z=up (board_scan_points)",
        "board": {"name": BOARD_NAME, "squares_x": BOARD_SQUARES_X, "squares_y": BOARD_SQUARES_Y,
                  "square_mm": BOARD_SQUARE_MM},
        "objects": obj_names, "parts": part_names_all,
        "board_detection": det, **meta,
    }
    (truth / "cameras.json").write_text(json.dumps(
        {"scan_frame": scene_json["scan_frame"], "frames": cam_entries}, indent=2))
    (truth / "scene.json").write_text(json.dumps(scene_json, indent=2))
    print(f"{scene_name}: wrote {video_path} ({len(frames)} frames, "
          f"{len(frames) / FPS:.1f}s), board seen in {det['frames_with_8plus_corners']}/{len(cams)} "
          f"render frames")
    return scene_json


# =========================================================================== CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="scene name (see --list)")
    ap.add_argument("--all", action="store_true", help="render every scene")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root (outside git)")
    ap.add_argument("--frames", type=int, default=90, help="frames to render (video loops to >=20s)")
    ap.add_argument("--portrait", action="store_true", help="store as phone portrait video")
    ap.add_argument("--fetch-gso", action="store_true",
                    help="download real Google Scanned Objects from Gazebo Fuel and render them "
                         "(falls back to the primitive gso_* scenes on any download failure)")
    ap.add_argument("--list", action="store_true", help="list scenes and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name in SCENES:
            print(name)
        for name in GSO_MODELS:
            print(f"{name}  (needs --fetch-gso)")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)

    if args.fetch_gso:
        cache = args.out / "_gso_cache"
        rendered = 0
        for name in GSO_MODELS:
            obj = fetch_gso(name, cache)
            if obj is None:
                print(f"{name}: download failed, skipping (use the primitive gso_* scene instead)")
                continue
            generate(name, args.out, args.frames, scene_override=build_gso_real(name, obj))
            rendered += 1
        print(f"fetched and rendered {rendered}/{len(GSO_MODELS)} real GSO scenes")
        if not args.all and not args.scene:
            return 0

    if args.all:
        for name in SCENES:
            generate(name, args.out, args.frames, portrait=(name in PORTRAIT_SCENES))
        return 0
    if not args.scene:
        if args.fetch_gso:
            return 0
        ap.error("pass --scene NAME, --all, or --fetch-gso (or --list)")
    generate(args.scene, args.out, args.frames,
             portrait=(args.portrait or args.scene in PORTRAIT_SCENES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
