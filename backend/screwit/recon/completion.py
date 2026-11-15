"""Close the walls the cameras never saw (SPEC 6.1 step 6, after the TSDF repair).

A phone orbit looks down at the object, and people rarely walk all the way round: on the headphones
clip the eight solved views span 206 degrees of azimuth, so the far side of both ear cups, the
lower walls, and the crevice where the cups touch were never observed. TSDF fusion only builds
surface where depth exists, so those walls came out as holes that show the inside of the object.

``complete_surface`` closes them with screened Poisson reconstruction of the fused surface, then
keeps only what the evidence allows:

- surface within ``max_gap`` of the fused surface (a hole is bridged; nothing grows far from data);
- nothing below the support (the object stands on it);
- nothing two or more cameras contradict: a point a camera saw through (it lies in front of that
  view's depth) or saw as background (it lies on a pixel outside the object mask, at that pixel's
  depth) is not object. This is what stops the reconstruction from capping the table inside a ring
  or growing a flared skirt at the base.

Walls nobody saw are therefore filled smoothly, while everything a camera did see stays as fused.

Open3D's Poisson solver (PoissonRecon) calls ``exit()`` on some inputs ("Failed to close loop" in
its multi-threaded surface extraction at depth 9), which would take the API process down with it,
so it runs in a child process (``python -m screwit.recon.completion``), single-threaded and at
depth 8 or less, under a timeout. Any failure returns the input mesh unchanged.

Measured on the headphones scan (M3 Pro): see-through share of the lower half in low side views
17.2% -> 10.6%, extents unchanged to 2 mm, about 4 s. ``SCREWIT_COMPLETE=0`` turns it off.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .tsdf import MeshResult, _compact, _decimate

MAX_GAP_REL = 0.06          # bridge holes up to about 2 x 6% of the object's diagonal
BOTTOM_BAND_MM = 2.0        # the cover under the object: surface this close to the support
MAX_POISSON_DEPTH = 8       # depth 9 hits the solver's exit() on real scans
CONTRADICT_VOTES = 2        # views that must contradict a completed point before it is dropped
POISSON_TIMEOUT_S = 90.0


def enabled() -> bool:
    return os.environ.get("SCREWIT_COMPLETE", "1").strip().lower() not in ("0", "false", "off", "no")


def complete_surface(mesh: MeshResult, depths_mm: list[np.ndarray], masks: list[np.ndarray],
                     intrinsics, extrinsics_mm, *, voxel_mm: float = 1.0,
                     tol_mm: float = 6.0) -> MeshResult:
    """Fill unobserved walls of ``mesh`` (scan frame, mm) without contradicting any view.

    ``depths_mm`` are the cleaned, UNmasked depth maps (the background is evidence too), ``masks``
    the object masks on the same grids, ``intrinsics``/``extrinsics_mm`` the matching cameras
    (world-to-camera, translation in mm). Returns the input unchanged when disabled, on any
    failure, or when the result fails a sanity check."""
    if not enabled() or mesh.triangles.shape[0] < 500 or not depths_mm:
        return mesh
    try:
        out = _complete(mesh, depths_mm, masks, intrinsics, extrinsics_mm, voxel_mm, tol_mm)
    except Exception:  # noqa: BLE001 - completion is cosmetic; keep the fused mesh on any error
        return mesh
    return out if out is not None and _plausible(mesh, out) else mesh


def _complete(mesh, depths_mm, masks, intrinsics, extrinsics_mm, voxel_mm, tol_mm):
    import open3d as o3d

    v = mesh.vertices_mm.astype(np.float64)
    f = mesh.triangles.astype(np.int32)
    diag = float(np.linalg.norm(np.ptp(v, axis=0)))
    if diag <= 0:
        return None
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    om.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh.vertex_colors, 0, 1).astype(np.float64))
    om.compute_triangle_normals()
    n_samples = int(np.clip(2 * len(f), 50_000, 400_000))
    # Seeded: the samples come from Open3D's process-wide generator, whose state otherwise depends
    # on whatever ran before in the API process, so a second scan of the same video got a
    # different mesh (and missed the reply cache downstream).
    o3d.utility.random.seed(0)
    pcd = om.sample_points_uniformly(number_of_points=n_samples, use_triangle_normal=True)
    pts, nrm, col = np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors)
    # The bottom cover: down-facing samples on the support under the object, so Poisson closes the
    # underside instead of leaking out below the table (the cups' bottoms stayed open otherwise).
    z0 = float(v[:, 2].min())
    tri = v[f]
    area = 0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    foot = footprint_points(v, f, z0, masks, intrinsics, extrinsics_mm,
                            spacing=max(float(np.sqrt(area / n_samples)), 0.3))
    if len(foot):
        pts = np.concatenate([pts, foot])
        nrm = np.concatenate([nrm, np.tile([0.0, 0.0, -1.0], (len(foot), 1))])
        if len(col) == n_samples:
            col = np.concatenate([col, _nearest(v, np.clip(mesh.vertex_colors, 0, 1), foot)])
    # Octree cells about the size of the fusion voxel.
    depth = int(np.clip(np.ceil(np.log2(1.1 * diag / max(voxel_mm, 0.25))), 6, MAX_POISSON_DEPTH))
    res = _poisson_in_child(pts, nrm, col if len(col) == len(pts) else np.zeros((0, 3)), depth)
    if res is None:
        return None
    pv, pf, pc = res
    if len(pf) == 0:
        return None

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    gap = scene.compute_distance(o3d.core.Tensor(pv.astype(np.float32))).numpy()
    # The object stands on a flat support: what Poisson bulged below it is squashed onto it. The
    # bottom cover is kept however far it is from the fused walls (no camera can see under the
    # object), unless a view contradicts it: the table seen inside a ring or around the footprint
    # is background, so the cover never spans it.
    pv = pv.copy()
    pv[:, 2] = np.maximum(pv[:, 2], z0)
    bottom = pv[:, 2] <= z0 + BOTTOM_BAND_MM
    keep = (gap <= MAX_GAP_REL * diag) | bottom
    keep &= contradicting_views(pv, depths_mm, masks, intrinsics, extrinsics_mm, tol_mm) < CONTRADICT_VOTES
    faces = pf[keep[pf].all(axis=1)]
    if len(faces) == 0:
        return None
    cols = pc if len(pc) == len(pv) else np.full((len(pv), 3), 0.6)
    out = _compact(MeshResult(pv.astype(np.float32), faces.astype(np.int32), cols),
                   np.ones(len(faces), bool), min_component_frac=0.01)
    # Poisson's output is denser than the fused mesh; bring it back to about the same size.
    target = int(1.15 * max(len(f), 1000))
    if len(out.triangles) > target:
        out = _decimate(out, target)
    return out


def footprint_points(v: np.ndarray, f: np.ndarray, z0: float, masks, intrinsics, extrinsics_mm, *,
                     spacing: float) -> np.ndarray:
    """Points on the support plane (z = ``z0``) under the object, on a grid ``spacing`` mm apart.

    A point is under the object when a vertical ray from it hits the fused surface ``v``/``f`` (so
    nothing is added beside the object, inside a ring, or past an unseen wall) and every view that
    sees its place puts it inside the object mask, eroded by 2 pixels (so nothing is added under an
    overhang the cameras saw the table beneath, such as a mug's handle)."""
    import open3d as o3d
    from scipy import ndimage

    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + spacing, spacing),
                         np.arange(lo[1], hi[1] + spacing, spacing))
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)], axis=1)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    rays = np.concatenate([pts + [0.0, 0.0, 0.5], np.tile([0.0, 0.0, 1.0], (len(pts), 1))], axis=1)
    under = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy())
    inside = under.copy()
    seen = np.zeros(len(pts), np.int32)
    for m, k, e in zip(masks, intrinsics, extrinsics_mm, strict=False):
        mask = ndimage.binary_erosion(np.asarray(m, bool), iterations=2)
        k = np.asarray(k, np.float64)
        e = np.asarray(e, np.float64)
        h, w = mask.shape
        pc = pts @ e[:3, :3].T + e[:3, 3]
        z = pc[:, 2]
        front = z > 1.0
        zs = np.where(front, z, 1.0)
        u = np.round(k[0, 0] * pc[:, 0] / zs + k[0, 2]).astype(np.int64)
        r = np.round(k[1, 1] * pc[:, 1] / zs + k[1, 2]).astype(np.int64)
        inb = front & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        seen += inb.astype(np.int32)
        idx = np.flatnonzero(inb)
        inside[idx] &= mask[r[idx], u[idx]]
    return pts[inside & (seen >= 2)]


def _nearest(src_v: np.ndarray, src_c: np.ndarray, dst: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(src_v).query(dst, k=1)
    return src_c[idx]


def contradicting_views(points: np.ndarray, depths_mm, masks, intrinsics, extrinsics_mm,
                        tol_mm: float) -> np.ndarray:
    """Per point, the number of views that saw it as empty space or as background.

    A view contradicts a point when the point projects inside the view and either lies more than
    ``tol_mm`` in front of what the view saw there (the camera looked through it; the depth is
    min-filtered over 5x5 pixels so a silhouette edge cannot fake this), or lies within ``tol_mm``
    of the view's depth on a pixel that is background even after growing the mask by 3 pixels."""
    from scipy import ndimage

    p = np.asarray(points, np.float64)
    votes = np.zeros(len(p), np.int32)
    for d, m, k, e in zip(depths_mm, masks, intrinsics, extrinsics_mm, strict=False):
        d = np.asarray(d, np.float64)
        k = np.asarray(k, np.float64)
        e = np.asarray(e, np.float64)
        h, w = d.shape
        valid = d > 0
        near = ndimage.minimum_filter(np.where(valid, d, np.inf), size=5)
        background = ~ndimage.binary_dilation(np.asarray(m, bool), iterations=3)
        pc = p @ e[:3, :3].T + e[:3, 3]
        z = pc[:, 2]
        front = z > 1.0
        zs = np.where(front, z, 1.0)
        u = np.round(k[0, 0] * pc[:, 0] / zs + k[0, 2]).astype(np.int64)
        r = np.round(k[1, 1] * pc[:, 1] / zs + k[1, 2]).astype(np.int64)
        inb = front & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        ui, ri = u[inb], r[inb]
        z_in = z[inb]
        seen_through = np.isfinite(near[ri, ui]) & (z_in < near[ri, ui] - tol_mm)
        on_background = valid[ri, ui] & background[ri, ui] & (np.abs(z_in - d[ri, ui]) <= tol_mm)
        votes[np.flatnonzero(inb)] += (seen_through | on_background).astype(np.int32)
    return votes


def _plausible(before: MeshResult, after: MeshResult) -> bool:
    """Completion fills holes; it must not shrink the surface or change the object's size."""
    def area(m):
        t = m.vertices_mm[m.triangles].astype(np.float64)
        return 0.5 * float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

    def span(m):
        lo, hi = np.percentile(m.vertices_mm, [1, 99], axis=0)
        return hi - lo

    a0, a1 = area(before), area(after)
    if not (0.9 * a0 <= a1 <= 1.6 * a0):
        return False
    s0, s1 = span(before), span(after)
    return bool(np.all(np.abs(s1 - s0) <= 0.05 * s0 + 3.0))


# ------------------------------------------------------------------------------- child process

def _poisson_in_child(points, normals, colors, depth: int):
    """Run screened Poisson in ``python -m screwit.recon.completion``; (V, F, C) or None."""
    with tempfile.TemporaryDirectory(prefix="screwit-poisson-") as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(src, points=np.asarray(points, np.float64), normals=np.asarray(normals, np.float64),
                 colors=np.asarray(colors, np.float64), depth=np.asarray(depth))
        try:
            proc = subprocess.run([sys.executable, "-m", "screwit.recon.completion", str(src), str(dst)],
                                  capture_output=True, timeout=POISSON_TIMEOUT_S, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0 or not dst.exists():
            return None   # the solver exited early (no output) or failed
        with np.load(dst) as z:
            return z["vertices"], z["triangles"].astype(np.int64), z["colors"]


def _child_main(src: str, dst: str) -> int:
    import open3d as o3d

    with np.load(src) as z:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(z["points"]))
        pcd.normals = o3d.utility.Vector3dVector(z["normals"])
        if len(z["colors"]) == len(z["points"]):
            pcd.colors = o3d.utility.Vector3dVector(z["colors"])
        depth = int(z["depth"])
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1,
                                                                        n_threads=1)
    cols = np.asarray(mesh.vertex_colors)
    tmp = dst + ".tmp.npz"
    np.savez(tmp, vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
             colors=cols if len(cols) == len(mesh.vertices) else np.zeros((0, 3)))
    os.replace(tmp, dst)   # written only on success: an early exit leaves no output
    return 0


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1], sys.argv[2]))
