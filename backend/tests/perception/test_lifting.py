"""2D->3D label lifting onto the mesh (SPEC 6.2 step 3)."""

from __future__ import annotations

import math

import numpy as np
import trimesh

from screwit.contracts import Cameras
from screwit.perception import lifting


def _ring_cameras(scan_id, center, radius, n=12, grid=(256, 256)):
    w, h = grid
    fx = fy = 300.0
    K = ((fx, 0.0, w / 2), (0.0, fy, h / 2), (0.0, 0.0, 1.0))
    frames = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        eye = np.array([center[0] + radius * math.cos(ang),
                        center[1] + radius * math.sin(ang), center[2] + 30.0])
        fwd = np.asarray(center, float) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=0)
        t = -R @ eye
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = t
        frames.append({"frame": f"f{i}.jpg", "width": w, "height": h,
                       "world_to_camera": tuple(tuple(float(x) for x in row) for row in m), "K": K})
    return Cameras(scan_id=scan_id, reconstruction_revision=1, frames=frames)


def _render_label_and_depth(mesh, vert_label, cam):
    """Produce a per-frame label image and depth map by projecting the mesh vertices."""
    w, h = cam.width, cam.height
    label = np.zeros((h, w), dtype=np.int32)
    depth = np.zeros((h, w), dtype=np.float32)
    zbuf = np.full((h, w), np.inf)
    K = np.asarray(cam.K, float)
    m = np.asarray(cam.world_to_camera, float)
    R, t = m[:3, :3], m[:3, 3]
    verts = np.asarray(mesh.vertices, float)
    cam_pts = verts @ R.T + t
    z = cam_pts[:, 2]
    uvw = cam_pts @ K.T
    px = uvw[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    py = uvw[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    # Splat each vertex into a small disc so the label image is dense.
    for idx in range(len(verts)):
        if z[idx] <= 0:
            continue
        j, i = int(round(px[idx])), int(round(py[idx]))
        for dj in range(-2, 3):
            for di in range(-2, 3):
                jj, ii = j + dj, i + di
                if 0 <= ii < h and 0 <= jj < w and z[idx] < zbuf[ii, jj]:
                    zbuf[ii, jj] = z[idx]
                    depth[ii, jj] = z[idx]
                    label[ii, jj] = vert_label[idx]
    return label, depth


def test_two_part_mesh_lifts_over_95_percent_correct():
    # Two stacked boxes: part 1 (label 1) below, part 2 (label 2) above, distinct in Z.
    lower = trimesh.creation.box(extents=(40, 40, 40))
    lower.apply_translation((0, 0, 20))
    upper = trimesh.creation.box(extents=(30, 30, 30))
    upper.apply_translation((0, 0, 60))
    mesh = trimesh.util.concatenate([lower, upper])

    verts = np.asarray(mesh.vertices, float)
    truth = np.where(verts[:, 2] < 40.0, 1, 2).astype(np.int32)

    cams = _ring_cameras("syn", center=(0, 0, 40), radius=160)
    masks, depths = [], []
    for cam in cams.frames:
        lab, dep = _render_label_and_depth(mesh, truth, cam)
        masks.append(lab)
        depths.append(dep)

    labels = lifting.vote_labels(verts, masks, depths, cams, occlusion_margin_mm=3.0)
    labels = lifting.smooth_labels(labels, np.asarray(mesh.faces, int), rounds=2)

    voted = labels > 0
    correct = (labels[voted] == truth[voted]).mean()
    assert voted.mean() > 0.9  # most vertices got at least one vote
    assert correct > 0.95


def test_split_mesh_by_label_produces_two_parts():
    lower = trimesh.creation.box(extents=(20, 20, 20))
    lower.apply_translation((0, 0, 10))
    upper = trimesh.creation.box(extents=(20, 20, 20))
    upper.apply_translation((0, 0, 40))
    mesh = trimesh.util.concatenate([lower, upper])
    verts = np.asarray(mesh.vertices, float)
    labels = np.where(verts[:, 2] < 25.0, 1, 2).astype(np.int32)
    subs = lifting.split_mesh_by_label(mesh, labels)
    assert set(subs) == {1, 2}
    assert len(subs[1].vertices) > 0 and len(subs[2].vertices) > 0
