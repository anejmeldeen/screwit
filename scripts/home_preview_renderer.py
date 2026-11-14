"""CPU rasterizer for static homepage model previews."""
from __future__ import annotations

import io

import numpy as np

BG_COLOR = (24, 26, 30)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->camera (OpenGL-style: camera looks down -Z, Y up)."""
    f = _unit(target - eye)
    s = _unit(np.cross(f, up))
    u = np.cross(s, f)
    R = np.stack([s, u, -f], axis=0)
    t = -R @ eye
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def render_mesh_png(vertices_mm: np.ndarray, faces: np.ndarray,
                    vertex_colors: np.ndarray | None = None, *,
                    angle_deg: float = 30.0, width: int = 512, height: int = 512) -> bytes:
    """Rasterise a mesh to a PNG (bytes) with a simple z-buffer and Lambert shading.

    The camera orbits the mesh centroid at ``angle_deg`` about the scan-frame Z axis, looking
    slightly down, framed so the whole mesh fits. Colours come from ``vertex_colors`` (RGB or
    RGBA, 0-255) when present, else a neutral grey.
    """
    from PIL import Image

    verts = np.asarray(vertices_mm, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    if len(verts) == 0 or len(faces) == 0:
        return _solid_png(width, height)

    center = verts.mean(axis=0)
    radius = float(np.linalg.norm(verts - center, axis=1).max()) or 1.0

    a = np.radians(angle_deg)
    # Z up in the scan frame; place the eye on a ring, lifted above the plane.
    dir_world = np.array([np.cos(a), np.sin(a), 0.6])
    eye = center + _unit(dir_world) * radius * 3.0
    up = np.array([0.0, 0.0, 1.0])
    view = _look_at(eye, center, up)

    cam = (np.c_[verts, np.ones(len(verts))] @ view.T)[:, :3]
    # Simple perspective: focal set so the bounding sphere fills ~80% of the frame.
    f = 0.8 * min(width, height) * (np.linalg.norm(eye - center) / radius) / 2.0
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, -1e-6, z)
    sx = (-cam[:, 0] / safe) * f + width / 2.0
    sy = (cam[:, 1] / safe) * f + height / 2.0
    screen = np.stack([sx, sy], axis=1)
    depth = -z  # positive distance in front of camera

    if vertex_colors is not None and len(vertex_colors):
        vc = np.asarray(vertex_colors, dtype=float)[:, :3]
    else:
        vc = np.full((len(verts), 3), 180.0)

    # Face normals for flat Lambert shading.
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.where(np.linalg.norm(fn, axis=1, keepdims=True) == 0, 1,
                       np.linalg.norm(fn, axis=1, keepdims=True))
    light = _unit(eye - center)
    shade = 0.55 + 0.45 * np.abs(fn @ light)

    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:] = BG_COLOR
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    for fi, (ia, ib, ic) in enumerate(faces):
        pa, pb, pc = screen[ia], screen[ib], screen[ic]
        za, zb, zc = depth[ia], depth[ib], depth[ic]
        if za <= 0 and zb <= 0 and zc <= 0:
            continue
        minx = max(int(np.floor(min(pa[0], pb[0], pc[0]))), 0)
        maxx = min(int(np.ceil(max(pa[0], pb[0], pc[0]))), width - 1)
        miny = max(int(np.floor(min(pa[1], pb[1], pc[1]))), 0)
        maxy = min(int(np.ceil(max(pa[1], pb[1], pc[1]))), height - 1)
        if minx > maxx or miny > maxy:
            continue
        area = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        if abs(area) < 1e-6:
            continue
        xs = np.arange(minx, maxx + 1)
        ys = np.arange(miny, maxy + 1)
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((pb[0] - pa[0]) * (gy - pa[1]) - (pb[1] - pa[1]) * (gx - pa[0])) / area
        w1 = ((pc[0] - pb[0]) * (gy - pb[1]) - (pc[1] - pb[1]) * (gx - pb[0])) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        # Edge AB weights C; edge BC weights A; remaining weight belongs to B.
        zz = w1 * za + w2 * zb + w0 * zc
        col = (w1[..., None] * vc[ia] + w2[..., None] * vc[ib] + w0[..., None] * vc[ic]) * shade[fi]
        sub_z = zbuf[miny:maxy + 1, minx:maxx + 1]
        closer = inside & (zz < sub_z) & (zz > 0)
        sub_z[closer] = zz[closer]
        sub_img = img[miny:maxy + 1, minx:maxx + 1]
        sub_img[closer] = np.clip(col[closer], 0, 255)

    out = Image.fromarray(img.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _solid_png(width: int, height: int) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), BG_COLOR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
