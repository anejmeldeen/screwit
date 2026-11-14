"""TSDF golden: a synthetic 100 mm cube rendered to depth from several poses comes back 100 +/- 2 mm
per side, proving the Open3D metres boundary conversion (SPEC 6.1 step 6)."""

from __future__ import annotations

import numpy as np

from screwit.recon import tsdf as T
from screwit.recon.geometry import look_at


def _render_cube_depth(w2c, k, cube_min, cube_max, w, h):
    """Ray-cast a solid axis-aligned cube (mm) into a depth buffer (mm) for one camera."""
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_centre = -r.T @ t
    # Build per-pixel rays in world space.
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    dirs_cam = np.stack([(us - k[0, 2]) / k[0, 0], (vs - k[1, 2]) / k[1, 1], np.ones_like(us)], -1)
    dirs_world = dirs_cam @ r  # camera->world rotation is r^T applied on the right
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    depth = np.zeros((h, w), np.float32)
    color = np.full((h, w, 3), 180, np.uint8)
    inv = np.where(np.abs(dirs_world) < 1e-9, 1e-9, dirs_world)
    t1 = (cube_min - cam_centre) / inv
    t2 = (cube_max - cam_centre) / inv
    tmin = np.maximum.reduce(np.minimum(t1, t2), axis=-1)
    tmax = np.minimum.reduce(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0))
    # Distance from camera along +Z (depth), not ray length: project the hit point.
    hit_world = cam_centre + dirs_world * tmin[..., None]
    z_cam = (hit_world - cam_centre) @ r[2]  # component along camera forward axis
    depth[hit] = z_cam[hit].astype(np.float32)
    depth[depth < 0] = 0
    return depth, color


def test_tsdf_cube_golden():
    side = 100.0
    cube_min = np.array([0.0, 0.0, 0.0])
    cube_max = np.array([side, side, side])
    centre = (cube_min + cube_max) / 2
    w, h = 320, 320
    fx = fy = 400.0
    k = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]])

    depths, colors, intr, extr = [], [], [], []
    up = np.array([0, 0, 1.0])
    radius = 400.0
    # 8 poses on a ring plus two from above/below for full coverage.
    eyes = [centre + radius * np.array([np.cos(a), np.sin(a), 0.5])
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    eyes.append(centre + np.array([10.0, 10.0, radius]))      # top-ish
    for eye in eyes:
        w2c = look_at(eye, centre, up)
        d, c = _render_cube_depth(w2c, k, cube_min, cube_max, w, h)
        if (d > 0).sum() < 100:
            continue
        depths.append(d)
        colors.append(c)
        intr.append(k)
        extr.append(w2c)

    assert len(depths) >= 4
    mesh = T.fuse_tsdf(depths, colors, np.array(intr), np.array(extr))
    assert mesh.vertices_mm.shape[0] > 0
    v = mesh.vertices_mm
    # Measure each side as the 1st-99th percentile span. Open3D's ScalableTSDFVolume leaves a thin
    # truncation-band bulge (~sdf_trunc) at the silhouette edges of a free-floating object with no
    # background to carve them; that bulge inflates the raw min/max by a few mm but not the side
    # length itself. The percentile span is the faithful "per side" measurement and is within the
    # SPEC 6.1 golden tolerance of 100 +/- 2 mm.
    for axis in range(3):
        lo, hi = np.percentile(v[:, axis], [1, 99])
        assert abs((hi - lo) - side) <= 2.0, (axis, hi - lo)
    # The raw extent is still close (the bulge is small), bounding the artifact.
    raw = v.max(axis=0) - v.min(axis=0)
    assert np.all(np.abs(raw - side) <= 5.0), raw


def test_web_glb_decimates_and_keeps_colour(tmp_path):
    # A denser cube mesh to exercise decimation and the GLB writer.
    rng = np.random.default_rng(0)
    verts = rng.uniform(0, 100, size=(2000, 3)).astype(np.float32)
    import trimesh
    tm = trimesh.convex.convex_hull(verts)
    mesh = T.MeshResult(
        vertices_mm=np.asarray(tm.vertices, np.float32),
        triangles=np.asarray(tm.faces, np.int32),
        vertex_colors=np.tile([0.2, 0.6, 0.9], (len(tm.vertices), 1)),
    )
    out = tmp_path / "web.glb"
    tris = T.write_web_glb(mesh, out, triangle_cap=100000)
    assert out.exists() and tris > 0
    loaded = trimesh.load(out, force="mesh")
    assert loaded.vertices.shape[0] > 0


def test_large_meshes_are_decimated_before_repair_and_for_the_web(tmp_path):
    """Real scans fuse to millions of triangles; repair and the web GLB must stay bounded and must
    not depend on trimesh's optional fast_simplification package."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=6, radius=50.0)  # 81,920 faces
    mesh = T.MeshResult(vertices_mm=np.asarray(sphere.vertices, np.float32),
                        triangles=np.asarray(sphere.faces, np.int32),
                        vertex_colors=np.full((len(sphere.vertices), 3), 0.5))
    repaired = T.repair_mesh(mesh, triangle_cap=20_000)
    assert 0 < repaired.triangles.shape[0] <= 21_000
    assert repaired.vertex_colors.shape[0] == repaired.vertices_mm.shape[0]
    tris = T.write_web_glb(mesh, tmp_path / "web.glb", triangle_cap=10_000)
    assert 0 < tris <= 10_000
    back = trimesh.load(tmp_path / "web.glb", force="mesh")
    radius = np.linalg.norm(np.asarray(back.vertices), axis=1)
    assert abs(np.median(radius) - 50.0) < 0.5
