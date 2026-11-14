"""Lift 2D part masks to the 3D mesh (SPEC 6.2 step 3).

For each mesh vertex: project it into every keyframe with ``P = K [R | t]`` built from that
frame's ``world_to_camera`` (scan-frame, OpenCV axes, millimetres), look up the mask label, and
take a majority vote, ignoring views where the vertex is occluded (its camera-space depth is
further than that frame's depth map by a margin). Then smooth the per-vertex labels over the mesh
adjacency graph, and split the mesh into one trimesh per part.

Everything is scan-frame millimetres. ``K`` applies to the model-grid image the mask lives on.
Label 0 is reserved for "no part" / background; part labels are 1..N.
"""

from __future__ import annotations

import numpy as np

from screwit.contracts import Cameras

BACKGROUND = 0


def _extrinsic(world_to_camera) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(world_to_camera, dtype=float)
    return m[:3, :3], m[:3, 3]


def project_vertices(vertices_mm: np.ndarray, K: np.ndarray, R: np.ndarray,
                     t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world (scan-frame, mm) vertices into one camera.

    Returns ``(pixels, cam_z)`` where ``pixels`` is Nx2 float (column u, row v) and ``cam_z`` is
    the camera-space depth (OpenCV: +Z forward, mm). Uses pixel coordinates with ``P = K [R | t]``
    exactly as SPEC 6.1 step 3 and 6.2 step 3 require (never normalised coordinates with K again).
    """
    cam = vertices_mm @ R.T + t  # world -> camera, OpenCV axes
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uvw = cam @ K.T
    px = uvw[:, 0] / safe_z
    py = uvw[:, 1] / safe_z
    return np.stack([px, py], axis=1), z


def _sample_mask(mask: np.ndarray, px: float, py: float) -> int:
    h, w = mask.shape
    j = int(round(px))
    i = int(round(py))
    if 0 <= i < h and 0 <= j < w:
        return int(mask[i, j])
    return -1  # outside the image: no vote


def _sample_depth(depth: np.ndarray, px: float, py: float) -> float:
    h, w = depth.shape
    j = int(round(px))
    i = int(round(py))
    if 0 <= i < h and 0 <= j < w:
        return float(depth[i, j])
    return 0.0


def vote_labels(vertices_mm: np.ndarray, part_label_masks: list[np.ndarray],
                depth_mm: list[np.ndarray], cameras: Cameras,
                *, occlusion_margin_mm: float = 5.0) -> np.ndarray:
    """Per-vertex label by occlusion-aware majority vote across keyframes.

    ``part_label_masks[k]`` is an int label image for keyframe k (0 = background, 1..N = parts);
    ``depth_mm[k]`` is that frame's depth map (0 = invalid). ``cameras`` supplies K and
    ``world_to_camera`` per frame, in the same order as the mask/depth lists.
    """
    verts = np.asarray(vertices_mm, dtype=float).reshape(-1, 3)
    n_labels = 1 + max((int(m.max()) for m in part_label_masks), default=0)
    votes = np.zeros((len(verts), n_labels), dtype=np.int32)

    for mask, depth, cam in zip(part_label_masks, depth_mm, cameras.frames, strict=True):
        K = np.asarray(cam.K, dtype=float)
        R, t = _extrinsic(cam.world_to_camera)
        pixels, cam_z = project_vertices(verts, K, R, t)
        for idx in range(len(verts)):
            if cam_z[idx] <= 0:  # behind the camera
                continue
            u, v = pixels[idx]
            d = _sample_depth(depth, u, v)
            if d > 0 and cam_z[idx] > d + occlusion_margin_mm:
                continue  # occluded by nearer surface in this view
            label = _sample_mask(mask, u, v)
            if label > 0:
                votes[idx, label] += 1
    # Background wins only when no part collected a vote.
    labels = votes[:, 1:].argmax(axis=1) + 1
    labels = np.where(votes[:, 1:].sum(axis=1) == 0, BACKGROUND, labels)
    return labels.astype(np.int32)


def _vertex_adjacency(faces: np.ndarray, n_vertices: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def smooth_labels(labels: np.ndarray, faces: np.ndarray, *, rounds: int = 3) -> np.ndarray:
    """A few rounds of neighbour majority voting over the mesh adjacency graph (SPEC 6.2 step 3)."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    adj = _vertex_adjacency(np.asarray(faces, dtype=int), len(labels))
    n_labels = int(labels.max()) + 1
    for _ in range(rounds):
        nxt = labels.copy()
        for i, neigh in enumerate(adj):
            if not neigh:
                continue
            counts = np.bincount([labels[j] for j in neigh], minlength=n_labels)
            counts[labels[i]] += 1  # keep own label as a tie-breaker toward stability
            nxt[i] = int(counts.argmax())
        labels = nxt
    return labels


def split_mesh_by_label(mesh, labels: np.ndarray) -> dict[int, object]:
    """Split a trimesh into one submesh per non-background label.

    A face belongs to the part that holds the majority of its three vertices. Returns a dict of
    ``label -> trimesh.Trimesh`` for every label that owns at least one face (background excluded).
    """
    import trimesh

    labels = np.asarray(labels, dtype=np.int32)
    faces = np.asarray(mesh.faces, dtype=int)
    face_label = np.zeros(len(faces), dtype=np.int32)
    for fi, (a, b, c) in enumerate(faces):
        tri = [labels[a], labels[b], labels[c]]
        counts = np.bincount([x for x in tri if x != BACKGROUND], minlength=int(labels.max()) + 1) \
            if any(x != BACKGROUND for x in tri) else None
        face_label[fi] = int(counts.argmax()) if counts is not None else BACKGROUND

    out: dict[int, object] = {}
    vcolors = getattr(mesh.visual, "vertex_colors", None)
    for label in sorted(set(int(x) for x in face_label) - {BACKGROUND}):
        keep = faces[face_label == label]
        used = np.unique(keep)
        remap = {int(old): new for new, old in enumerate(used)}
        new_faces = np.array([[remap[int(v)] for v in f] for f in keep], dtype=np.int64)
        sub = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[used], faces=new_faces,
                              process=False)
        if vcolors is not None:
            sub.visual.vertex_colors = np.asarray(vcolors)[used]
        out[label] = sub
    return out
