"""Rules learned from the first real-model CPU scan (DA3 on X2's phone clip, 2026-11-14).

- The object mask must isolate the object from the table it stands on: keeping all valid depth fused
  a 1.6 m x 1.6 m slab of table into a 6.4M-triangle mesh.
- A box is not a cylinder: a 100 mm cube was fitted as a 50 mm-radius cylinder with 13% inliers.
- The part box must be tight: the PCA box of a cube came out 165 x 164 x 131 mm.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from screwit.contracts import Cameras
from screwit.interfaces import FrameSet
from screwit.perception import measure_fit
from screwit.perception.object_mask import GeometricObjectMasker


def _cams(target, radius, n=8, grid=(96, 72), height=250.0):
    w, h = grid
    k = ((80.0, 0, w / 2), (0, 80.0, h / 2), (0, 0, 1.0))
    frames = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        eye = np.array([target[0] + radius * math.cos(ang), target[1] + radius * math.sin(ang),
                        height])
        fwd = np.asarray(target, float) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r = np.stack([right, down, fwd])
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = -r @ eye
        frames.append({"frame": f"f{i}.jpg", "width": w, "height": h,
                       "world_to_camera": tuple(tuple(float(x) for x in row) for row in m), "K": k})
    return Cameras(scan_id="syn", reconstruction_revision=1, frames=frames)


def _render_depth(scene: trimesh.Trimesh, cam) -> np.ndarray:
    w, h = cam.width, cam.height
    k = np.asarray(cam.K, float)
    m = np.asarray(cam.world_to_camera, float)
    r, t = m[:3, :3], m[:3, 3]
    origin = -r.T @ t
    ys, xs = np.mgrid[0:h, 0:w]
    dirs_cam = np.stack([(xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs, float)],
                        axis=-1).reshape(-1, 3)
    dirs = dirs_cam @ r  # camera -> world
    hits, ray_idx, _ = scene.ray.intersects_location(
        np.repeat(origin[None], len(dirs), axis=0), dirs, multiple_hits=False)
    depth = np.zeros(h * w, np.float32)
    cam_pts = hits @ r.T + t
    depth[ray_idx] = cam_pts[:, 2]
    return depth.reshape(h, w)


def test_masker_keeps_the_object_and_drops_table_and_clutter():
    cube = trimesh.creation.box(extents=(100, 100, 100))
    cube.apply_translation((0, 0, 50))
    clutter = trimesh.creation.box(extents=(60, 60, 60))
    clutter.apply_translation((480, 0, 30))
    table = trimesh.creation.box(extents=(1400, 1400, 2))
    table.apply_translation((0, 0, -1))
    scene = trimesh.util.concatenate([cube, clutter, table])
    cams = _cams((0, 0, 40), radius=420)
    depths = [_render_depth(scene, c) for c in cams.frames]
    fs = FrameSet(images=[], depth_mm=depths, confidence=[], cameras=cams, full_res_paths=[])

    masker = GeometricObjectMasker()
    masks = masker.masks(fs)
    assert masker.last_info.method == "support_plane_clusters"

    inside = outside = 0
    for depth, mask, cam in zip(depths, masks, cams.frames, strict=True):
        k = np.asarray(cam.K, float)
        m = np.asarray(cam.world_to_camera, float)
        ys, xs = np.nonzero(mask)
        z = depth[ys, xs]
        p = np.stack([(xs - k[0, 2]) / k[0, 0] * z, (ys - k[1, 2]) / k[1, 1] * z, z], axis=1)
        world = (p - m[:3, 3]) @ m[:3, :3]
        # Allowed: the cube, plus the table band directly at its base (one sample spacing, about
        # 12 mm at this coarse test resolution) that the masker puts back so the bottom is kept.
        on_cube = (np.abs(world[:, 0]) <= 52) & (np.abs(world[:, 1]) <= 52) & (world[:, 2] > 1)
        at_base = (np.abs(world[:, 0]) <= 64) & (np.abs(world[:, 1]) <= 64) & (world[:, 2] <= 1)
        inside += int(on_cube.sum())
        outside += int((~on_cube & ~at_base).sum())
    assert inside > 500
    assert outside / max(1, inside + outside) < 0.02


def test_masker_falls_back_without_a_support_plane():
    cube = trimesh.creation.box(extents=(100, 100, 100))
    cube.apply_translation((0, 0, 50))
    cams = _cams((0, 0, 40), radius=420)
    depths = [_render_depth(cube, c) for c in cams.frames]
    fs = FrameSet(images=[], depth_mm=depths, confidence=[], cameras=cams, full_res_paths=[])
    masker = GeometricObjectMasker()
    masks = masker.masks(fs)
    assert masker.last_info.method == "all_valid_depth"
    assert all(np.array_equal(m, d > 0) for m, d in zip(masks, depths, strict=True))


def test_obb_of_a_rotated_cube_is_tight():
    cube = trimesh.creation.box(extents=(100, 100, 100))
    cube.apply_transform(trimesh.transformations.rotation_matrix(math.radians(30), (0, 0, 1)))
    pts = trimesh.sample.sample_surface(cube, 20000, seed=0)[0]
    obb = measure_fit.oriented_bounding_box(pts)
    assert np.allclose(sorted(obb.extent_mm), [100, 100, 100], atol=2.0)


def test_a_box_part_gets_box_dimensions_not_a_fake_cylinder():
    from screwit.perception.service import PerceptionImpl

    cube = trimesh.creation.box(extents=(100, 80, 60))
    pts = trimesh.sample.sample_surface(cube, 20000, seed=0)[0]
    features, dims = PerceptionImpl("fake")._features_for("p1", "part", pts)
    assert features == []
    got = {d.name: d.value.value_mm for d in dims}
    assert abs(got["length"] - 100) < 2 and abs(got["width"] - 80) < 2 and abs(got["thickness"] - 60) < 2
