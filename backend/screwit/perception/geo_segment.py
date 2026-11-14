"""Geometry-based part segmentation (P3) — CPU-only, no GPU, no ML weights.

Decomposes ONE object mesh into coherent sub-parts by local thickness (Shape Diameter Function) and
mesh connectivity. This is the CPU-viable alternative to SAM 2 (box prompts swallow the whole object)
and SAM 3 (segments whole objects by concept, not sub-parts): neither does part-level on a single
connected object without a GPU. Geometry does: on headphones it cleanly separates the thin headband
from the bulky ear cups in ~1 s. Parts are unnamed here; the caller names each with the VLM.

Method:
1. SDF per face = inward ray distance to the opposite surface (local thickness), via Open3D's C++
   raycaster (fast).
2. Over-segment: split into thickness bands, then connected components over the face-adjacency graph
   so each seed region is spatially coherent.
3. Merge the smallest region into its most thickness-similar mesh-adjacent neighbour until the target
   part count remains (keeps thin and thick parts distinct instead of one region dominating).

Returns a dict ``label -> trimesh.Trimesh`` (same shape as ``lifting.split_mesh_by_label``), so it
drops into ``PerceptionImpl.parts`` in place of the 2D-mask lifting path.
"""

from __future__ import annotations

import numpy as np

MIN_SEED_FACES = 200        # drop specks when seeding
DEFAULT_TARGET_PARTS = 5


def segment_mesh(mesh, target_parts: int = DEFAULT_TARGET_PARTS) -> dict[int, object]:
    """Split ``mesh`` (a trimesh) into up to ``target_parts`` coherent submeshes by thickness +
    connectivity. Returns ``{1: trimesh, 2: trimesh, ...}`` (1-based labels, like the lifting path).
    Falls back to a single part on any failure or a tiny/degenerate mesh."""
    import trimesh

    try:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) < max(50, MIN_SEED_FACES):
            return _single(mesh)
        face_label = _label_faces(mesh, target_parts)
        if face_label is None:
            return _single(mesh)
        out: dict[int, object] = {}
        next_label = 1
        for u in _by_size(face_label):
            keep = faces[face_label == u]
            used = np.unique(keep)
            remap = {int(o): i for i, o in enumerate(used)}
            new_faces = np.array([[remap[int(v)] for v in f] for f in keep], dtype=np.int64)
            sub = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[used], faces=new_faces,
                                  process=False)
            vcol = getattr(mesh.visual, "vertex_colors", None)
            if vcol is not None:
                sub.visual.vertex_colors = np.asarray(vcol)[used]
            out[next_label] = sub
            next_label += 1
        return out or _single(mesh)
    except Exception:  # noqa: BLE001 - segmentation is best-effort; never fail the scan
        return _single(mesh)


def _single(mesh) -> dict[int, object]:
    return {1: mesh}


def _by_size(face_label: np.ndarray):
    uniq, cnt = np.unique(face_label, return_counts=True)
    return uniq[np.argsort(-cnt)]


def face_sdf(mesh) -> np.ndarray:
    """Per-face local thickness via one inward ray along -normal (Open3D C++ raycaster)."""
    import open3d as o3d

    V = np.asarray(mesh.vertices, np.float32)
    F = np.asarray(mesh.faces, np.uint32)
    fc = np.asarray(mesh.triangles_center, np.float32)
    fn = np.asarray(mesh.face_normals, np.float32)
    diam = float(np.linalg.norm(np.ptp(V, axis=0))) or 1.0
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V, o3d.core.float32), o3d.core.Tensor(F, o3d.core.uint32))
    rays = np.concatenate([fc - fn * (1e-3 * diam), -fn], axis=1).astype(np.float32)
    sdf = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    finite = np.isfinite(sdf)
    sdf[~finite] = np.nanmedian(sdf[finite]) if finite.any() else diam
    return sdf


def _label_faces(mesh, target_parts: int) -> np.ndarray | None:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    faces = np.asarray(mesh.faces, dtype=np.int64)
    fc = np.asarray(mesh.triangles_center, dtype=np.float64)
    sdf = face_sdf(mesh)
    n = len(faces)

    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adj) == 0:
        return None
    g = sparse.coo_matrix(
        (np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
        shape=(n, n)).tocsr()

    # Seed: thickness bands x connected components.
    q = np.quantile(sdf, [0.4, 0.75])
    band = np.digitize(sdf, q)
    lab = -np.ones(n, dtype=np.int64)
    nxt = 0
    for b in (0, 1, 2):
        idx = np.where(band == b)[0]
        if len(idx) < 50:
            continue
        _, comp = connected_components(g[idx][:, idx], directed=False)
        for c in np.unique(comp):
            fcs = idx[comp == c]
            if len(fcs) < MIN_SEED_FACES:
                continue
            lab[fcs] = nxt
            nxt += 1
    if nxt == 0:
        return None
    # Assign leftover (speck) faces to the nearest labelled face.
    if (lab < 0).any():
        known = lab >= 0
        tree = cKDTree(fc[known])
        _, nn = tree.query(fc[~known])
        lab[~known] = lab[known][nn]

    def region_adjacency_weighted(lbl):
        """Shared-boundary strength between regions: how many adjacent face-pairs cross each region
        pair. A long shared seam (the headband's two arms meeting over the top bridge) outweighs a
        single-face touch (an arm brushing a cushion), so merges follow real connectivity."""
        a, b = lbl[adj[:, 0]], lbl[adj[:, 1]]
        diff = a != b
        pairs = {}
        for x, y in zip(a[diff], b[diff], strict=True):
            k = (min(x, y), max(x, y))
            pairs[k] = pairs.get(k, 0) + 1
        return pairs

    def mean_thick(lbl, u):
        return float(sdf[lbl == u].mean())

    def centroid(lbl, u):
        return fc[lbl == u].mean(axis=0)

    # Reduce to target_parts by repeatedly merging the smallest region. Prefer the mesh-adjacent
    # neighbour it shares the LONGEST seam with (ties broken by thickness similarity): this keeps a
    # thin headband arm joined to the other arm over the top bridge rather than to a cushion it only
    # brushes. When the smallest region is a DISCONNECTED component (no shared seam at all -- common
    # on sparse CPU fusion, which is exactly why a headband came out in two halves), fall back to the
    # spatially NEAREST region by centroid, so detached pieces still rejoin instead of leaving dozens
    # of fragments. Either way the loop always makes progress and reaches target_parts.
    while len(set(lab.tolist())) > max(1, target_parts):
        uniq, cnt = np.unique(lab, return_counts=True)
        small = int(uniq[np.argmin(cnt)])
        pairs = region_adjacency_weighted(lab)
        neigh = [(b if a == small else a, w) for (a, b), w in pairs.items() if small in (a, b)]
        if neigh:
            st = mean_thick(lab, small)
            max_w = max(w for _, w in neigh)
            # Candidates within 60% of the strongest seam; among them pick the most thickness-similar.
            strong = [u for u, w in neigh if w >= 0.6 * max_w]
            target = min(strong, key=lambda u: abs(mean_thick(lab, u) - st))
        else:
            # Disconnected: merge into the nearest OTHER region by centroid distance.
            sc = centroid(lab, small)
            others = [u for u in uniq.tolist() if u != small]
            target = min(others, key=lambda u: float(np.linalg.norm(centroid(lab, u) - sc)))
        lab[lab == small] = target
    return lab
