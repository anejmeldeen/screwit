"""The pipeline is object-agnostic (AGENTS.md "The demo object can be anything").

Runs the full parts() path on a NON-plumbing synthetic object -- a mug (round cylindrical body)
plus an L-bracket (flat plate) -- and asserts it produces sensible, correctly named parts with a
geometry-appropriate feature each and NO snaps (no tubular table matches a mug or a bracket).
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from screwit.contracts import Cameras
from screwit.interfaces import FrameSet
from screwit.perception.labels import Labeller
from screwit.perception.service import PerceptionImpl


def _project_label_depth(mesh, vert_label, cam):
    w, h = cam.width, cam.height
    lab = np.zeros((h, w), np.int32)
    dep = np.zeros((h, w), np.float32)
    zb = np.full((h, w), np.inf)
    K = np.asarray(cam.K, float)
    m = np.asarray(cam.world_to_camera, float)
    R, t = m[:3, :3], m[:3, 3]
    verts = np.asarray(mesh.vertices, float)
    cam_pts = verts @ R.T + t
    z = cam_pts[:, 2]
    uvw = cam_pts @ K.T
    px = uvw[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    py = uvw[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    for k in range(len(verts)):
        if z[k] <= 0:
            continue
        for dj in range(-3, 4):
            for di in range(-3, 4):
                jj, ii = int(round(px[k])) + dj, int(round(py[k])) + di
                if 0 <= ii < h and 0 <= jj < w and z[k] < zb[ii, jj]:
                    zb[ii, jj] = z[k]
                    dep[ii, jj] = z[k]
                    lab[ii, jj] = vert_label[k]
    return lab, dep


def _ring(center, radius, n=12, grid=(220, 220)):
    w, h = grid
    K = ((320.0, 0, w / 2), (0, 320.0, h / 2), (0, 0, 1.0))
    frames = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        eye = np.array([center[0] + radius * math.cos(ang),
                        center[1] + radius * math.sin(ang), center[2] + 40.0])
        fwd = np.asarray(center, float) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=0)
        t = -R @ eye
        mm = np.eye(4)
        mm[:3, :3] = R
        mm[:3, 3] = t
        frames.append({"frame": f"f{i}.jpg", "width": w, "height": h,
                       "world_to_camera": tuple(tuple(float(x) for x in r) for r in mm), "K": K})
    return Cameras(scan_id="syn_mug", reconstruction_revision=1, frames=frames)


def test_non_plumbing_object_gets_sensible_parts_and_no_snaps(storage, tmp_path):
    # Part 1: a mug body -> a 50 mm-diameter cylinder (no tube/bore table entry near 50 mm).
    body = trimesh.creation.cylinder(radius=25.0, height=90.0, sections=48)
    body.apply_translation((0, 0, 45))
    # Part 2: an L-bracket face -> a thin flat plate (plane role by geometry).
    plate = trimesh.creation.box(extents=(80, 60, 4))
    plate.apply_translation((90, 0, 30))
    mesh = trimesh.util.concatenate([body, plate])
    mesh_path = tmp_path / "mug.glb"
    mesh.export(mesh_path)

    verts = np.asarray(mesh.vertices, float)
    vert_label = np.where(verts[:, 0] < 45.0, 1, 2).astype(np.int32)  # x<45 body, else bracket

    cams = _ring(center=(45, 0, 45), radius=320)
    labels, depths = [], []
    for cam in cams.frames:
        lab, dep = _project_label_depth(mesh, vert_label, cam)
        labels.append(lab)
        depths.append(dep)

    fs = FrameSet(images=[], depth_mm=depths, confidence=[], cameras=cams, full_res_paths=[])
    fs.part_label_images = labels

    # Names come from data, not hard-coded plumbing: inject the fake's name list for this scan.
    labeller = Labeller("fake", fake_names={"syn_mug": ["mug body", "bracket"]})
    svc = PerceptionImpl("fake", labeller=labeller)
    parts, graph = svc.parts("syn_mug", fs, [li > 0 for li in labels], mesh_path, 1, storage)

    assert len(parts) == 2
    by_id = {p.part_id: p for p in parts}
    # Sensible, data-driven labels (no "plumbing" anywhere). Labels are shown Title Case.
    labels_seen = {p.label for p in parts}
    assert labels_seen == {"Mug Body", "Bracket"}

    # Geometry picked the right role per part.
    body_part = by_id["p1"]
    bracket_part = by_id["p2"]
    assert body_part.features[0].type == "cylinder"
    assert bracket_part.features[0].type == "plane"
    # The mug body is ~50 mm diameter; the fit is close and does NOT snap (no table near 50 mm).
    assert body_part.features[0].radius.scan_mm == 25.0 or abs(body_part.features[0].radius.scan_mm - 25.0) < 4.0

    # No snap on ANY measurement of ANY part (object has no matching snap table).
    for p in parts:
        for f in p.features:
            for field in ("radius", "length", "extent_u", "extent_v", "flat_depth", "depth"):
                m = getattr(f, field, None)
                if m is not None:
                    assert m.snapped_mm is None, f"{p.part_id}.{f.feature_id}.{field} should not snap"
                    assert m.preferred in ("scan", "caliper")

    # Scene graph is valid and object-agnostic (category is generic, not a plumbing term).
    assert sorted(graph.parts) == sorted(p.part_id for p in parts)
    assert "trap" not in graph.object.category.lower()
    assert "sink" not in graph.object.summary.lower()
