"""Part segmentation for any object: geometric candidates, chosen and named by the VLM (P3, 6.2).

Geometry alone cannot decide what a "part" is (two touching ear cups have the same thickness; a
pizza-box lid and base meet in a seam that looks like any other edge), and a VLM cannot draw a
precise boundary on a mesh or reliably read numbered patches (on the headphones it grouped the two
cups into one part). So the work is split:

1. ``over_segment`` cuts the mesh into about 10 patches along its creases and necks: spectral
   clustering of the face graph, with edges across concave creases weighted down (part boundaries
   sit in concave creases, the "minima rule"). Each patch should lie inside one part.
2. ``merge_hierarchy`` merges the most strongly joined neighbours one pair at a time, giving one
   candidate partition per part count (on the headphones, 3 regions is exactly cup, cup, band).
3. ``render_partition`` draws candidates with 2-6 regions in named colours; the caller asks the VLM
   once to pick the candidate that matches the object's real parts and to name each region by
   colour. Choosing between pictures and naming colours is the part a VLM is good at.
4. Without a VLM (fake provider, failed call) the caller takes the candidate nearest a target count.

Nothing here knows any object category. Everything runs on a ~8k-face working copy of the mesh, and
labels are carried back to the full mesh by nearest face.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

WORK_FACES = 8000
DEFAULT_PATCHES = 10
MAX_PARTS = 6
MIN_PATCH_AREA = 0.02      # a patch below 2% of the surface is merged into a neighbour
CONVEX_WEIGHT = 0.15       # convex edges barely count as boundaries (Liu & Zhang 2004)

_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
    [220, 190, 255], [170, 110, 40], [128, 0, 0], [170, 255, 195], [128, 128, 0]], dtype=float)


@dataclass
class Patches:
    work: object             # trimesh.Trimesh working copy
    labels: np.ndarray       # patch id per working face, 0..n-1
    n: int


# ------------------------------------------------------------------------------- working mesh

def work_mesh(mesh, faces: int = WORK_FACES):
    """Decimated, cleaned working copy (trimesh). Spectral clustering and rendering scale with the
    face count, and 8k faces resolve parts far below the scan's noise."""
    import open3d as o3d
    import trimesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if len(f) > faces:
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        om = om.simplify_quadric_decimation(target_number_of_triangles=faces)
        om.remove_degenerate_triangles()
        om.remove_unreferenced_vertices()
        v, f = np.asarray(om.vertices), np.asarray(om.triangles)
    w = trimesh.Trimesh(v, f, process=True)
    # Drop specks: components under 1% of the faces carry no part information.
    comps = w.split(only_watertight=False)
    if len(comps) > 1:
        keep = [c for c in comps if len(c.faces) >= 0.01 * len(w.faces)] or [max(comps, key=lambda c: len(c.faces))]
        w = trimesh.util.concatenate(keep)
    return w


def _edges(w):
    adj = np.asarray(w.face_adjacency)
    ang = np.asarray(w.face_adjacency_angles)
    convex = np.asarray(w.face_adjacency_convex)
    ev = np.asarray(w.face_adjacency_edges)
    elen = np.linalg.norm(w.vertices[ev[:, 0]] - w.vertices[ev[:, 1]], axis=1)
    return adj, ang, convex, elen


def _affinity(w, delta: float = 0.3):
    """Face-graph edge weights: high inside smooth surface, low across concave creases."""
    adj, ang, convex, _ = _edges(w)
    d_ang = np.where(convex, CONVEX_WEIGHT, 1.0) * (1.0 - np.cos(ang))
    fc = w.triangles_center
    d_geo = np.linalg.norm(fc[adj[:, 0]] - fc[adj[:, 1]], axis=1)
    d = (delta * d_geo / max(float(d_geo.mean()), 1e-12)
         + (1 - delta) * d_ang / max(float(d_ang.mean()), 1e-12))
    return adj, np.exp(-d / max(float(d.mean()), 1e-12))


# ------------------------------------------------------------------------------- over-segmentation

def over_segment(mesh, n_patches: int = DEFAULT_PATCHES) -> Patches:
    """Spectral over-segmentation into about ``n_patches`` crease-bounded patches."""
    from scipy import sparse
    from scipy.cluster.vq import kmeans2
    from scipy.sparse.linalg import eigsh

    w = work_mesh(mesh)
    nf = len(w.faces)
    if nf < 50:
        return Patches(w, np.zeros(nf, dtype=np.int64), 1)
    adj, wgt = _affinity(w)
    k = int(max(2, min(n_patches, nf // 50)))
    mat = sparse.coo_matrix((np.r_[wgt, wgt], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                            shape=(nf, nf)).tocsr()
    deg = np.asarray(mat.sum(axis=1)).ravel()
    dinv = sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    norm = dinv @ mat @ dinv
    try:
        _, vecs = eigsh(norm, k=k, which="LA", tol=1e-4, v0=np.ones(nf))
    except Exception:  # noqa: BLE001 - eigensolver failure -> one patch, caller falls back
        return Patches(w, np.zeros(nf, dtype=np.int64), 1)
    emb = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    _, lab = kmeans2(emb, k, minit="++", seed=0)
    lab = _clean_islands(w, lab)
    return Patches(w, lab, int(lab.max()) + 1)


def _clean_islands(w, lab: np.ndarray) -> np.ndarray:
    """Make every patch one connected piece, and fold patches under MIN_PATCH_AREA into the
    neighbour they share the longest boundary with. Isolated specks with no neighbour are kept."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    adj, _, _, elen = _edges(w)
    nf = len(w.faces)
    same = lab[adj[:, 0]] == lab[adj[:, 1]]
    g = sparse.coo_matrix((np.ones(int(same.sum())), (adj[same, 0], adj[same, 1])), shape=(nf, nf))
    _, comp = connected_components(g, directed=False)
    area = w.area_faces
    total = float(area.sum()) or 1.0
    stuck: set[int] = set()
    while True:
        comp = np.unique(comp, return_inverse=True)[1]
        a = np.bincount(comp, weights=area)
        small = [int(s) for s in np.argsort(a) if a[s] < MIN_PATCH_AREA * total and int(s) not in stuck]
        if not small:
            break
        s = small[0]
        ca, cb = comp[adj[:, 0]], comp[adj[:, 1]]
        cross = (ca != cb) & ((ca == s) | (cb == s))
        if not cross.any():
            stuck.add(s)
            continue
        other = np.where(ca[cross] == s, cb[cross], ca[cross])
        lens = np.bincount(other, weights=elen[cross], minlength=int(comp.max()) + 1)
        comp[comp == s] = int(np.argmax(lens))
        stuck = set()
    return np.unique(comp, return_inverse=True)[1]


def patch_adjacency(p: Patches) -> dict[tuple[int, int], float]:
    """Shared boundary length (mm) between each pair of touching patches."""
    adj, _, _, elen = _edges(p.work)
    a, b = p.labels[adj[:, 0]], p.labels[adj[:, 1]]
    out: dict[tuple[int, int], float] = {}
    for x, y, ln in zip(a[a != b], b[a != b], elen[a != b], strict=True):
        key = (int(min(x, y)), int(max(x, y)))
        out[key] = out.get(key, 0.0) + float(ln)
    return out


# ------------------------------------------------------------------------------- grouping

def merge_hierarchy(p: Patches) -> dict[int, list[int]]:
    """Every partition from ``p.n`` patches down to 1 group, built by repeatedly merging the two
    touching groups joined most strongly (cut affinity relative to the smaller group's volume).
    Returns {group count: group id per patch}, with group 0 the largest by area in each partition.
    Patches that touch nothing are merged last, into the largest group."""
    adj, wgt = _affinity(p.work)
    nf = len(p.work.faces)
    deg = np.bincount(adj[:, 0], weights=wgt, minlength=nf) + np.bincount(adj[:, 1], weights=wgt, minlength=nf)
    area = p.work.area_faces
    group = list(range(p.n))
    out: dict[int, list[int]] = {}
    while True:
        out[len(set(group))] = _by_area(group, p.labels, area)
        if len(set(group)) <= 1:
            break
        g_face = np.asarray(group)[p.labels]
        vol = np.bincount(g_face, weights=deg, minlength=p.n)
        a, b = g_face[adj[:, 0]], g_face[adj[:, 1]]
        cross = a != b
        cut: dict[tuple[int, int], float] = {}
        for x, y, ww in zip(a[cross], b[cross], wgt[cross], strict=True):
            key = (int(min(x, y)), int(max(x, y)))
            cut[key] = cut.get(key, 0.0) + float(ww)
        if cut:
            (x, y), _ = max(cut.items(),
                            key=lambda kv: kv[1] / max(min(vol[kv[0][0]], vol[kv[0][1]]), 1e-12))
        else:  # disconnected groups: fold the smallest into the largest
            ga = np.bincount(g_face, weights=area, minlength=p.n)
            live = sorted(set(group), key=lambda g: ga[g])
            x, y = live[-1], live[0]
        group = [x if g == y else g for g in group]
    return out


def _by_area(group: list[int], labels: np.ndarray, area: np.ndarray) -> list[int]:
    g = np.asarray(group)
    ga = np.bincount(g[labels], weights=area, minlength=len(group))
    order = sorted(set(group), key=lambda x: -ga[x])
    ids = {x: i for i, x in enumerate(order)}
    return [ids[x] for x in group]


def merge_by_affinity(p: Patches, target: int) -> list[int]:
    """The partition with ``target`` groups from ``merge_hierarchy`` (clamped to what exists)."""
    h = merge_hierarchy(p)
    k = min(h, key=lambda c: abs(c - max(1, target)))
    return h[k]


# ------------------------------------------------------------------------------- full-mesh labels

def full_face_labels(mesh, p: Patches, patch_group: list[int], rounds: int = 2) -> np.ndarray:
    """Group id per face of the full-resolution mesh: nearest working face, then a couple of rounds
    of neighbour-majority smoothing so the boundaries are clean."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    tri = np.asarray(mesh.triangles_center)
    _, nn = cKDTree(p.work.triangles_center).query(tri, k=1)
    lab = np.asarray(patch_group, dtype=np.int64)[p.labels[nn]]
    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0:
        return lab
    nf, ng = len(tri), int(lab.max()) + 1
    a = sparse.coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                          shape=(nf, nf)).tocsr()
    for _ in range(rounds):
        onehot = sparse.coo_matrix((np.ones(nf), (np.arange(nf), lab)), shape=(nf, ng)).tocsr()
        votes = (a @ onehot + onehot).toarray()
        lab = votes.argmax(axis=1)
    return lab


def submeshes(mesh, face_labels: np.ndarray) -> dict[int, object]:
    """{group id: trimesh} from per-face labels, keeping vertex colours."""
    out = {}
    for g in np.unique(face_labels):
        sub = mesh.submesh([np.flatnonzero(face_labels == g)], append=True)
        if len(sub.faces):
            out[int(g)] = sub
    return out


# ------------------------------------------------------------------------------- rendering

# Region colours for the candidate renders, in group order (group 0 is the largest), with the names
# the VLM uses to refer to them. Chosen to stay distinguishable under shading.
REGION_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (220, 40, 40)), ("green", (40, 170, 60)), ("blue", (45, 95, 225)),
    ("yellow", (240, 210, 40)), ("orange", (245, 130, 30)), ("purple", (150, 55, 195))]


def render_partition(p: Patches, patch_group: list[int], *, size: int = 384) -> bytes:
    """PNG of one candidate partition: two views, each group in its REGION_COLORS colour."""
    colors = np.array([c for _, c in REGION_COLORS], dtype=float)
    face_group = np.asarray(patch_group, dtype=np.int64)[p.labels]
    return _render(p.work, colors[face_group % len(colors)], size=size)


def _render(w, face_rgb: np.ndarray, *, size: int, angles=(35.0, 215.0), elev_deg: float = 40.0,
            numbers: np.ndarray | None = None) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import distance_transform_edt

    centre = w.vertices.mean(axis=0)
    v = w.vertices - centre
    try:
        font = ImageFont.load_default(size=max(18, size // 18))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    tiles = []
    for az in angles:
        a, e = np.radians(az), np.radians(elev_deg)
        eye = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        fwd = -eye
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right) or 1.0
        up = np.cross(right, fwd)
        x, y, z = v @ right, v @ up, v @ fwd
        s = (size - 40) / max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
        px = (x - x.min()) * s + 20 + (size - 40 - np.ptp(x) * s) / 2
        py = (y.max() - y) * s + 20 + (size - 40 - np.ptp(y) * s) / 2
        light = np.clip(-(w.face_normals @ fwd), 0.0, 1.0)
        order = np.argsort(-z[w.faces].mean(axis=1))           # far to near
        img = Image.new("RGB", (size, size), (24, 26, 32))
        dr = ImageDraw.Draw(img)
        ids = Image.new("RGB", (size, size), (0, 0, 0)) if numbers is not None else None
        di = ImageDraw.Draw(ids) if ids is not None else None
        for f in order:
            poly = [(float(px[i]), float(py[i])) for i in w.faces[f]]
            dr.polygon(poly, fill=tuple(int(c) for c in face_rgb[f] * (0.45 + 0.55 * light[f])))
            if di is not None:
                fid = int(f) + 1
                di.polygon(poly, fill=(fid & 255, (fid >> 8) & 255, (fid >> 16) & 255))
        if numbers is not None:
            idbuf = np.asarray(ids).astype(np.int64)
            face_at = idbuf[..., 0] + (idbuf[..., 1] << 8) + (idbuf[..., 2] << 16) - 1
            num_at = np.where(face_at >= 0, numbers[np.clip(face_at, 0, len(w.faces) - 1)], -1)
            for num in np.unique(numbers):
                visible = num_at == num
                if visible.sum() < 120:
                    continue
                dist = distance_transform_edt(visible)
                cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
                dr.text((int(cx), int(cy)), str(int(num)), fill=(255, 255, 255), font=font,
                        anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
        tiles.append(np.asarray(img))
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(tiles, axis=1)).save(buf, format="PNG")
    return buf.getvalue()


def render_patches(p: Patches, *, size: int = 512) -> bytes:
    """PNG of the raw patches, each in its own colour with its number (a debugging aid)."""
    face_rgb = _PALETTE[p.labels % len(_PALETTE)]
    return _render(p.work, face_rgb, size=size, numbers=p.labels)
