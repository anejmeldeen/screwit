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
# Taubin smoothing of the fused surface. TSDF at 1 mm voxels leaves a stair-stepped (aliased)
# surface on real CPU scans; Taubin (lambda>0 then mu<0 passes) removes that high-frequency noise
# WITHOUT the volume shrink a plain Laplacian causes, so dimensions stay faithful. Applied in
# repair_mesh (the pipeline cleanup), NOT in fuse_tsdf, so the raw-fusion golden test is unaffected.
# Tunable via SCREWIT_SMOOTH_ITERS (0 disables).
SMOOTH_ITERS = 12


@dataclass
class MeshResult:
    vertices_mm: np.ndarray    # (V, 3) float32, scan frame mm
    triangles: np.ndarray      # (T, 3) int32
    vertex_colors: np.ndarray  # (V, 3) float in [0, 1]


def _make_volume(voxel_length: float = VOXEL_LENGTH_M, sdf_trunc: float = SDF_TRUNC_M):
    import open3d as o3d

    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _smooth_mesh(mesh, iterations: int | None = None):
    """Taubin-smooth an Open3D mesh in place of plain Laplacian: it removes TSDF staircase noise
    without shrinking the volume (dimensions preserved). No-op when disabled or the mesh is tiny.
    Vertex colours are preserved by Open3D's filter. Best-effort: never fail the fuse."""
    import os

    if iterations is None:
        iterations = int(os.environ.get("SCREWIT_SMOOTH_ITERS", SMOOTH_ITERS))
    if iterations <= 0 or len(mesh.triangles) < 50:
        return mesh
    try:
        sm = mesh.filter_smooth_taubin(number_of_iterations=iterations)
        if len(sm.triangles) >= 50:  # guard against a degenerate smoothing result
            sm.compute_vertex_normals()
            return sm
    except Exception:  # noqa: BLE001 - smoothing is cosmetic; keep the unsmoothed mesh on error
        pass
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
    """Close small holes with pymeshfix. pymeshfix time grows steeply with size on noisy real scans
    (about 11 s at 225k triangles, 11 minutes at 1M on 32 cores), so larger meshes are decimated to
    ``triangle_cap`` first; at 1 mm voxels that still keeps sub-millimetre surface detail.

    Before repair, the fused surface is denoised (stray floaters, TSDF speckle) and Taubin-smoothed
    (removes the 1 mm voxel staircase without shrinking the volume). This is the pipeline's cosmetic
    cleanup; it lives here, not in ``fuse_tsdf``, so the raw-fusion golden test measures unsmoothed
    geometry while every real scan gets the clean surface."""
    import pymeshfix

    if mesh.triangles.shape[0] == 0:
        return mesh
    mesh = _cleanup_surface(mesh)
    if mesh.triangles.shape[0] > triangle_cap:
        mesh = _decimate(mesh, triangle_cap)
    fixer = pymeshfix.MeshFix(mesh.vertices_mm.astype(np.float64), mesh.triangles.astype(np.int32))
    fixer.repair()
    v = np.asarray(fixer.points, np.float32)
    f = np.asarray(fixer.faces, np.int32)
    # pymeshfix drops colours; re-sample nearest original vertex colour.
    colors = _nearest_colors(mesh.vertices_mm, mesh.vertex_colors, v)
    return MeshResult(vertices_mm=v, triangles=f, vertex_colors=colors)


def _cleanup_surface(mesh: MeshResult) -> MeshResult:
    """Denoise + Taubin-smooth a MeshResult via an Open3D round-trip, preserving vertex colours.
    Best-effort: returns the input unchanged on any error or a tiny mesh."""
    import numpy as _np

    if mesh.triangles.shape[0] < 100:
        return mesh
    try:
        import open3d as o3d

        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices_mm.astype(_np.float64)),
            o3d.utility.Vector3iVector(mesh.triangles.astype(_np.int32)))
        om.vertex_colors = o3d.utility.Vector3dVector(
            _np.clip(mesh.vertex_colors, 0, 1).astype(_np.float64))
        om = _denoise_mesh(om)
        om = _smooth_mesh(om)
        v = _np.asarray(om.vertices, _np.float32)
        t = _np.asarray(om.triangles, _np.int32)
        if v.shape[0] < 3 or t.shape[0] < 1:
            return mesh
        cols = _np.asarray(om.vertex_colors)
        if cols.shape[0] != v.shape[0]:
            cols = _nearest_colors(mesh.vertices_mm, mesh.vertex_colors, v)
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
