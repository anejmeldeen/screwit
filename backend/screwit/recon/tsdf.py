"""TSDF meshing (SPEC 6.1 step 6).

Open3D works in METRES, full stop: its ``ScalableTSDFVolume`` allocates voxel blocks with a
hard-coded depth scale of 1000, so millimetre depth opens blocks at 1/1000 of the right positions
and yields a shredded mesh. We convert at the boundary -- depth mm->m and camera translations
mm->m (rotation unchanged) -- integrate with ``depth_scale=1.0``, ``depth_trunc=1.5``,
``convert_rgb_to_intensity=False``, ``voxel_length=0.001``, ``sdf_trunc=0.004``, RGB8 -- then
multiply the extracted vertices by 1000 to return to scan-frame millimetres.

Keep the largest connected component, close small holes with pymeshfix, keep the full mesh for
measurement, and decimate to under 200k triangles for the web GLB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOXEL_LENGTH_M = 0.001
# Truncation band around the surface. Kept at 4 mm: the cube golden (100 +/- 2 mm) regresses to
# ~107 mm at 10 mm trunc, so a wider band trades dimensional accuracy for hole-filling. Sparse-view
# fusion gaps are addressed by the mask/centering fixes and pymeshfix repair instead.
SDF_TRUNC_M = 0.004
DEPTH_TRUNC_M = 1.5
WEB_TRIANGLE_CAP = 200_000
REPAIR_TRIANGLE_CAP = 400_000
# Taubin smoothing of the fused surface (taubin_smooth): lambda/mu passes remove high-frequency
# bumps without the volume shrink of a plain Laplacian, with open-boundary vertices pinned so many
# passes cannot grow spikes. Measured on the headphones scan: 80 passes leave the p1-p99 extents
# unchanged to 0.1 mm and take 0.1 s. Applied in repair_mesh (the pipeline cleanup), NOT in
# fuse_tsdf, so the raw-fusion golden test is unaffected. Tunable via SCREWIT_SMOOTH_ITERS (0 off).
SMOOTH_ITERS = 80


@dataclass
class MeshResult:
    vertices_mm: np.ndarray    # (V, 3) float32, scan frame mm
    triangles: np.ndarray      # (T, 3) int32
    vertex_colors: np.ndarray  # (V, 3) float in [0, 1]


def fusion_resolution(object_size_mm: float | None) -> tuple[float, float]:
    """(voxel_length_m, sdf_trunc_m) for an object whose bounding-box diagonal is ``object_size_mm``.

    A fixed 1 mm voxel is finer than the depth noise on the CPU path (two views disagree by about
    4-6 mm about the same surface at 0.7 m), so it reconstructs the noise as bumps and costs time
    with no detail gain. Scale the voxel with the object (diagonal / 200, clamped to 1-3 mm) so the
    triangle count stays roughly constant across object sizes. The truncation band stays at 4 mm
    (2.5 voxels above 1.6 mm voxels): the band's silhouette bulge adds roughly its own width to
    measured sizes (the cube golden reads 107 mm at 10 mm), so it must not grow with the voxel. A
    100 mm object keeps the original 1 mm / 4 mm; the headphones (about 290 mm) get 1.45 mm / 4 mm,
    about 2x fewer triangles."""
    if not object_size_mm or not np.isfinite(object_size_mm) or object_size_mm <= 0:
        return VOXEL_LENGTH_M, SDF_TRUNC_M
    voxel_mm = float(np.clip(object_size_mm / 200.0, 1.0, 3.0))
    return voxel_mm / 1000.0, max(SDF_TRUNC_M, 2.5 * voxel_mm / 1000.0)


def _make_volume(voxel_length: float = VOXEL_LENGTH_M, sdf_trunc: float = SDF_TRUNC_M):
    import open3d as o3d

    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _remove_support_slab(mesh: MeshResult) -> MeshResult:
    """Cut away a fused support surface (the table the object lies on).

    Real CPU scans of an object lying on a table fuse the table in as one huge, thin, flat
    horizontal slab at the base: on the headphones clip 73% of the vertices sat in the bottom 15 mm,
    inflating the footprint to 218x236 mm (table, not object) and fusing the table into the nearest
    part. This finds the dominant horizontal plane within the bottom of the mesh via RANSAC and
    removes the thin slab of triangles lying in it, but ONLY when that slab is a table and not the
    object's own flat base: it must be nearly horizontal, thin (a table surface, not a tall object),
    and much WIDER than the object above it (a genuine object base is not). Best-effort and
    conservative -- on any doubt it keeps everything. Disable with SCREWIT_REMOVE_TABLE=0.
    """
    import os

    if os.environ.get("SCREWIT_REMOVE_TABLE", "1").strip().lower() in ("0", "false", "off", "no"):
        return mesh
    if mesh.triangles.shape[0] < 500:
        return mesh
    try:
        import open3d as o3d

        V = mesh.vertices_mm.astype(np.float64)
        F = mesh.triangles.astype(np.int64)
        ext = V.max(axis=0) - V.min(axis=0)
        # Scan frame is Z-up (AGENTS.md conventions); use Z as the support normal when it has a real
        # span, else fall back to the flattest axis.
        up = 2 if ext[2] > 1e-3 else int(np.argmin(ext))
        zc = V[:, up]
        z0, z1 = zc.min(), zc.max()
        height = z1 - z0
        if height < 1e-3:
            return mesh
        # Candidate table band: the bottom 25% of the height.
        band = zc <= z0 + 0.25 * height
        if band.sum() < 0.2 * len(V):  # no dominant low slab -> probably no table
            return mesh
        o3d.utility.random.seed(0)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[band]))
        thr = max(1.0, 0.01 * height)  # ~1 mm or 1% of height
        (a, b, c, dd), _ = pcd.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=400)
        normal = np.array([a, b, c], float)
        normal /= (np.linalg.norm(normal) or 1.0)
        # Must be nearly horizontal (plane normal within ~25 deg of the up axis).
        axis = np.zeros(3)
        axis[up] = 1.0
        if abs(float(normal @ axis)) < 0.9:
            return mesh
        # Signed distance of every vertex to the plane.
        dist = V @ normal + dd
        thin = np.abs(dist) < 2.0 * thr          # vertices lying IN the plane (a ~2 mm slab)
        med = np.median(dist[~band])
        above = dist > 3.0 * thr if med > 0 else dist < -3.0 * thr
        # A table is a THIN, DOMINANT horizontal plane: a large share of all vertices packed into a
        # few-mm band, with real object geometry standing clear of it. (Footprint width is not a
        # reliable discriminator -- a flat object and its table have similar spans.) An object's own
        # flat base is only a small fraction of its vertices, so it fails the dominance test and is
        # kept.
        if thin.mean() < 0.30 or above.sum() < 500:
            return mesh
        # Remove every triangle that lies within/below the plane (all three vertices not clearly
        # above it), keeping the object that stands above.
        above_v = dist > 1.5 * thr if med > 0 else dist < -1.5 * thr
        drop_face = (~above_v[F]).all(axis=1)
        if drop_face.all() or (~drop_face).sum() < 500:
            return mesh
        keep_v = np.zeros(len(V), bool)
        keep_v[np.unique(F[~drop_face])] = True
        remap = -np.ones(len(V), np.int64)
        remap[keep_v] = np.arange(int(keep_v.sum()))
        new_F = remap[F[~drop_face]]
        return MeshResult(vertices_mm=V[keep_v].astype(np.float32), triangles=new_F.astype(np.int32),
                          vertex_colors=mesh.vertex_colors[keep_v])
    except Exception:  # noqa: BLE001 - table removal is best-effort; keep the mesh on any error
        return mesh


def _denoise_mesh(mesh):
    """Drop stray floaters and TSDF speckle before smoothing. Two cheap passes:
    1. Remove tiny disconnected triangle clusters (specks under ~0.5% of the largest cluster) that
       survive even the keep-all component rule.
    2. Statistical outlier removal on the vertices: a vertex far from its neighbours (a floater
       spat out by a bad depth pixel) is removed and its triangles with it.
    No-op/best-effort: on any error or a tiny mesh, return the mesh unchanged. Disable with
    SCREWIT_DENOISE=0."""
    import os

    import numpy as np

    if os.environ.get("SCREWIT_DENOISE", "1").strip().lower() in ("0", "false", "off", "no"):
        return mesh
    if len(mesh.triangles) < 100:
        return mesh
    try:
        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if counts.size > 1:
            keep_min = max(25, int(0.005 * counts.max()))
            small = {i for i, c in enumerate(counts) if c < keep_min}
            if small:
                mesh.remove_triangles_by_mask(np.isin(np.asarray(labels), list(small)))
                mesh.remove_unreferenced_vertices()
        # Statistical outlier removal maps cleanly through the vertex set: build a cloud, find the
        # kept indices, and drop every triangle that references a removed vertex.
        if len(mesh.vertices) >= 50:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud(mesh.vertices)
            _, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            if 0 < len(keep_idx) < len(mesh.vertices):
                keep_mask = np.zeros(len(mesh.vertices), dtype=bool)
                keep_mask[np.asarray(keep_idx)] = True
                tris = np.asarray(mesh.triangles)
                tri_ok = keep_mask[tris].all(axis=1)
                mesh.remove_triangles_by_mask(~tri_ok)
                mesh.remove_unreferenced_vertices()
    except Exception:  # noqa: BLE001 - denoise is cosmetic; keep the mesh on any error
        pass
    return mesh


def fuse_tsdf(depth_mm_list: list[np.ndarray], color_list: list[np.ndarray],
              intrinsics: np.ndarray, extrinsics_mm: np.ndarray,
              *, keep_largest: bool = True, voxel_length: float = VOXEL_LENGTH_M,
              sdf_trunc: float = SDF_TRUNC_M) -> MeshResult:
    """Fuse masked depth maps (mm) with their world-to-camera poses (translation in mm) into a mesh.

    ``color_list`` are HxWx3 uint8 images matching the depth grids. Returns vertices in scan-frame
    millimetres. ``keep_largest`` keeps only the biggest connected component (the default, good when
    fusion is clean); set False on a sparse/fragmented scan to keep every component above a small
    size so the object is not dropped when fusion shatters it. ``voxel_length``/``sdf_trunc`` allow a
    looser retry that closes gaps across sparse views.
    """
    import open3d as o3d

    volume = _make_volume(voxel_length, sdf_trunc)
    h, w = depth_mm_list[0].shape
    for i, (depth_mm, color) in enumerate(zip(depth_mm_list, color_list, strict=False)):
        k = intrinsics[i]
        intr = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        )
        depth_m = (depth_mm.astype(np.float32) / 1000.0)
        color_img = o3d.geometry.Image(np.ascontiguousarray(color.astype(np.uint8)))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_m))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=DEPTH_TRUNC_M,
            convert_rgb_to_intensity=False,
        )
        extr_m = extrinsics_mm[i].copy().astype(np.float64)
        extr_m[:3, 3] /= 1000.0   # translation mm -> m; rotation unchanged
        volume.integrate(rgbd, intr, extr_m)

    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    # Connected-component filtering. keep_largest: one component (clean scans). Otherwise keep every
    # component above a small fraction of the biggest, so a fragmented sparse-view fuse does not lose
    # the object to the single-largest rule.
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if counts.size > 0:
        if keep_largest:
            keep_set = {int(np.argmax(counts))}
        else:
            thresh = max(50, int(0.05 * counts.max()))
            keep_set = {int(i) for i, c in enumerate(counts) if c >= thresh}
        remove = ~np.isin(labels, list(keep_set))
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()

    verts_m = np.asarray(mesh.vertices)
    verts_mm = (verts_m * 1000.0).astype(np.float32)   # back to scan-frame mm
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    cols = np.asarray(mesh.vertex_colors)
    if cols.size == 0:
        cols = np.full((verts_mm.shape[0], 3), 0.6)
    return MeshResult(vertices_mm=verts_mm, triangles=tris, vertex_colors=cols)


def _decimate(mesh: MeshResult, target_triangles: int) -> MeshResult:
    """Quadric decimation with Open3D (a declared dependency), keeping vertex colours."""
    import open3d as o3d

    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices_mm.astype(np.float64)),
                                   o3d.utility.Vector3iVector(mesh.triangles.astype(np.int32)))
    om.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh.vertex_colors, 0, 1).astype(np.float64))
    om = om.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles))
    om.remove_unreferenced_vertices()
    cols = np.asarray(om.vertex_colors)
    v = np.asarray(om.vertices, np.float32)
    if cols.shape[0] != v.shape[0]:
        cols = _nearest_colors(mesh.vertices_mm, mesh.vertex_colors, v)
    return MeshResult(vertices_mm=v, triangles=np.asarray(om.triangles, np.int32), vertex_colors=cols)


def repair_mesh(mesh: MeshResult, triangle_cap: int = REPAIR_TRIANGLE_CAP) -> MeshResult:
    """Clean the fused surface and close its small holes.

    Steps: cut a fused support slab if one survived, drop floaters, Taubin-smooth (shrink-free),
    decimate above ``triangle_cap``, then fill holes up to 5% of the object's size with Open3D's
    hole filler. The cleanup lives here, not in ``fuse_tsdf``, so the raw-fusion golden test still
    measures unsmoothed geometry.

    This used to be pymeshfix, which closes every hole however large: on the headphones scan it
    spent 158-280 s turning 394k triangles into 1.1M by capping the large openings (the open bottom
    of the cut-away table) with invented surface. Filling only small holes takes about 20 ms and
    leaves real openings open, which is what the viewer, the measurements and the part segmentation
    want."""
    if mesh.triangles.shape[0] == 0:
        return mesh
    mesh = _cleanup_surface(mesh)
    if mesh.triangles.shape[0] > triangle_cap:
        mesh = _decimate(mesh, triangle_cap)
    return _fill_small_holes(mesh)


def _fill_small_holes(mesh: MeshResult, rel_size: float = 0.05) -> MeshResult:
    """Fill holes whose size is under ``rel_size`` of the bounding-box diagonal (Open3D tensor API).
    Best-effort: on any failure the mesh is returned unchanged."""
    if mesh.triangles.shape[0] < 4:
        return mesh
    try:
        import open3d as o3d

        diag = float(np.linalg.norm(np.ptp(mesh.vertices_mm, axis=0)))
        tm = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(mesh.vertices_mm.astype(np.float32)),
            o3d.core.Tensor(mesh.triangles.astype(np.int32)))
        filled = tm.fill_holes(hole_size=max(1.0, rel_size * diag))
        v = filled.vertex.positions.numpy().astype(np.float32)
        f = filled.triangle.indices.numpy().astype(np.int32)
        if len(f) < len(mesh.triangles):
            return mesh
        colors = (mesh.vertex_colors if len(v) == len(mesh.vertices_mm)
                  else _nearest_colors(mesh.vertices_mm, mesh.vertex_colors, v))
        return MeshResult(vertices_mm=v, triangles=f, vertex_colors=colors)
    except Exception:  # noqa: BLE001 - hole filling is cosmetic
        return mesh


def taubin_smooth(vertices: np.ndarray, triangles: np.ndarray, iterations: int,
                  lam: float = 0.5, mu: float = -0.53, max_move_edges: float = 3.0) -> np.ndarray:
    """Taubin smoothing (lambda/mu passes of the uniform Laplacian) that removes high-frequency
    bumps without shrinking the volume, with two guards Open3D's filter lacks:

    - open-boundary vertices stay fixed (a boundary vertex has a one-sided neighbourhood, so the
      Laplacian pulls it inward every pass and, after many passes, out into needle spikes);
    - no vertex moves more than ``max_move_edges`` median edge lengths from where it started.
    """
    from scipy.sparse import coo_matrix, diags

    v0 = np.asarray(vertices, np.float64)
    f = np.asarray(triangles, np.int64)
    n = len(v0)
    if iterations <= 0 or len(f) < 50:
        return v0.astype(np.float32)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    a = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                   shape=(n, n)).tocsr()
    a.data[:] = 1.0                                      # duplicate edges count once
    deg = np.asarray(a.sum(axis=1)).ravel()
    lap = diags(1.0 / np.maximum(deg, 1.0)) @ a          # neighbour mean
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key[:, 0] * n + key[:, 1], return_inverse=True, return_counts=True)
    boundary = np.zeros(n, bool)
    be = key[cnt[inv] == 1]
    boundary[be.ravel()] = True
    free = (~boundary & (deg > 0))[:, None]
    lens = np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1)
    cap = max_move_edges * float(np.median(lens)) if len(lens) else 0.0
    v = v0.copy()
    for _ in range(iterations):
        for step in (lam, mu):
            v = np.where(free, v + step * (lap @ v - v), v)
    if cap > 0:
        d = v - v0
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        v = v0 + d * np.minimum(1.0, cap / np.maximum(norm, 1e-12))
    return v.astype(np.float32)


def _cleanup_surface(mesh: MeshResult) -> MeshResult:
    """Cut a fused support slab, drop floaters (Open3D), then boundary-safe Taubin smoothing,
    preserving vertex colours. Best-effort: returns the input unchanged on any error or a tiny
    mesh."""
    import os

    import numpy as _np

    if mesh.triangles.shape[0] < 100:
        return mesh
    mesh = _remove_support_slab(mesh)
    try:
        import open3d as o3d

        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices_mm.astype(_np.float64)),
            o3d.utility.Vector3iVector(mesh.triangles.astype(_np.int32)))
        om.vertex_colors = o3d.utility.Vector3dVector(
            _np.clip(mesh.vertex_colors, 0, 1).astype(_np.float64))
        om = _denoise_mesh(om)
        v = _np.asarray(om.vertices, _np.float32)
        t = _np.asarray(om.triangles, _np.int32)
        if v.shape[0] < 3 or t.shape[0] < 1:
            return mesh
        cols = _np.asarray(om.vertex_colors)
        if cols.shape[0] != v.shape[0]:
            cols = _nearest_colors(mesh.vertices_mm, mesh.vertex_colors, v)
        iters = int(os.environ.get("SCREWIT_SMOOTH_ITERS", SMOOTH_ITERS))
        v = taubin_smooth(v, t, iters)
        return MeshResult(vertices_mm=v, triangles=t, vertex_colors=cols)
    except Exception:  # noqa: BLE001 - cleanup is cosmetic; keep the raw mesh on error
        return mesh


def _nearest_colors(src_v: np.ndarray, src_c: np.ndarray, dst_v: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    if src_v.shape[0] == 0:
        return np.full((dst_v.shape[0], 3), 0.6)
    tree = cKDTree(src_v)
    _, idx = tree.query(dst_v, k=1)
    return src_c[idx]


def to_trimesh(mesh: MeshResult):
    import trimesh

    return trimesh.Trimesh(
        vertices=mesh.vertices_mm.astype(np.float64),
        faces=mesh.triangles.astype(np.int64),
        vertex_colors=(np.clip(mesh.vertex_colors, 0, 1) * 255).astype(np.uint8),
        process=False,
    )


def write_full_ply(mesh: MeshResult, path: Path) -> None:
    to_trimesh(mesh).export(Path(path))


def write_web_glb(mesh: MeshResult, path: Path, triangle_cap: int = WEB_TRIANGLE_CAP) -> int:
    """Decimate to under the triangle cap and write a web GLB (scan frame, mm). Returns the final
    triangle count."""
    if mesh.triangles.shape[0] > triangle_cap:
        mesh = _decimate(mesh, triangle_cap)
    tm = to_trimesh(mesh)
    tm.export(Path(path))
    return int(tm.faces.shape[0])


def drop_unsupported(mesh: MeshResult, depths_mm: list[np.ndarray], intrinsics, extrinsics_mm,
                     *, tol_mm: float, min_views: int = 2, rim_rings: int = 4,
                     rounds: int = 3) -> MeshResult:
    """Peel the ragged single-view flaps off the surface's open edges.

    A vertex counts as observed by a view when it projects inside that view and the view's depth
    there is within ``tol_mm`` of the vertex's own depth. TSDF keeps whatever a single view saw at
    its silhouette: thin flaps that end in a free edge along rims. Only faces within ``rim_rings``
    edge rings of an open boundary are candidates, peeled for ``rounds`` passes, so a flap is cut
    back to where two views agree but an interior surface that only one view saw (a thin band's far
    side) is kept: dropping those too punched holes that broke a headband into pieces and derailed
    the part segmentation. Specks under 1% of the mesh are removed. Disable with SCREWIT_MIN_VIEWS=1.
    """
    import os

    from scipy.sparse import coo_matrix

    min_views = int(os.environ.get("SCREWIT_MIN_VIEWS", min_views))
    if min_views <= 1 or mesh.triangles.shape[0] < 100 or len(depths_mm) < min_views:
        return mesh
    v = mesh.vertices_mm.astype(np.float64)
    support = np.zeros(len(v), np.int32)
    for depth, k, e in zip(depths_mm, intrinsics, extrinsics_mm, strict=False):
        k = np.asarray(k, float)
        e = np.asarray(e, float)
        h, w = depth.shape
        pc = v @ e[:3, :3].T + e[:3, 3]
        z = pc[:, 2]
        front = z > 1e-6
        u = np.full(len(v), -1)
        r = np.full(len(v), -1)
        u[front] = np.round(k[0, 0] * pc[front, 0] / z[front] + k[0, 2]).astype(int)
        r[front] = np.round(k[1, 1] * pc[front, 1] / z[front] + k[1, 2]).astype(int)
        inb = front & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        d = np.zeros(len(v))
        d[inb] = depth[r[inb], u[inb]]
        support += (inb & (d > 0) & (np.abs(d - z) <= tol_mm)).astype(np.int32)
    low_v = support < min_views
    tris = mesh.triangles.astype(np.int64)
    alive = np.ones(len(tris), bool)
    n = len(v)
    for _ in range(rounds):
        t = tris[alive]
        e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        key = e[:, 0] * n + e[:, 1]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        near = np.zeros(n, bool)
        near[e[cnt[inv] == 1].ravel()] = True                  # open-boundary vertices
        adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        adj = adj + adj.T
        for _ in range(rim_rings):
            near = near | (adj @ near.astype(np.float64) > 0)
        drop = alive.copy()
        drop[alive] = low_v[t].all(axis=1) & near[t].any(axis=1)
        if not drop.any():
            break
        alive &= ~drop
    if alive.sum() < 0.5 * len(alive):   # implausible: keep the mesh rather than gut it
        return mesh
    return _compact(mesh, alive, min_component_frac=0.01)


def _compact(mesh: MeshResult, keep_face: np.ndarray, min_component_frac: float = 0.0) -> MeshResult:
    """Keep the given faces (and drop connected pieces under ``min_component_frac`` of them)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tris = mesh.triangles[keep_face]
    if min_component_frac > 0 and len(tris):
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]]])
        n = len(mesh.vertices_mm)
        _, comp = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                                  shape=(n, n)), directed=False)
        fc = comp[tris[:, 0]]
        sizes = np.bincount(fc)
        tris = tris[sizes[fc] >= min_component_frac * len(tris)]
    used = np.unique(tris)
    remap = -np.ones(len(mesh.vertices_mm), np.int64)
    remap[used] = np.arange(len(used))
    return MeshResult(vertices_mm=mesh.vertices_mm[used], triangles=remap[tris].astype(np.int32),
                      vertex_colors=mesh.vertex_colors[used])
