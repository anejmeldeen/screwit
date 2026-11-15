"""Part segmentation for any object: geometric patches, grouped and named by the VLM (P3, 6.2).

Geometry alone cannot decide what a "part" is (two touching ear cups have the same thickness; a
pizza-box lid and base meet in a seam that looks like any other edge), and a VLM cannot draw a
precise boundary on a mesh. So the work is split:

1. ``over_segment`` cuts the mesh into about 10 patches along its creases and necks: spectral
   clustering of the face graph, with edges across concave creases weighted down (part boundaries
   sit in concave creases, the "minima rule"). Each patch should lie inside one part.
2. ``render_patch_grid`` draws one tile per patch: the whole object in grey, that patch in red,
   from the same two sides in every tile. The caller asks the VLM once to group the patch numbers
   into named parts. The VLM is reliable about what kind of part a patch is and how many copies
   exist, not about which copy (it paired two cushion tops as one ear cup), so ``regroup_instances``
   re-splits every repeated kind by location, and ``grouping_to_owner`` completes the map.
   (Two designs failed on real scans first: one image of numbered colours, which the VLM misread,
   and choosing among geometric merges, which offered no correct option once a scan hole split the
   headband.)
3. Without a VLM (fake provider, failed call) ``merge_hierarchy`` merges the most strongly joined,
   most similarly thick neighbours down to the proposed part count.

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
    """Face-graph edge weights: high inside smooth surface, low across concave creases.

    Scans come in pieces (a hole where no two views agreed splits a band in two). Separate pieces
    are disconnected in the mesh graph, and spectral clustering then cuts along the gaps instead of
    along real part boundaries. So faces of different pieces that lie within a small distance
    (3% of the object's size) are bridged with weak edges (a quarter of the median weight): pieces of
    one part are glued, while a real crease still costs more to cross than a bridge is worth."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    adj, ang, convex, _ = _edges(w)
    d_ang = np.where(convex, CONVEX_WEIGHT, 1.0) * (1.0 - np.cos(ang))
    fc = w.triangles_center
    d_geo = np.linalg.norm(fc[adj[:, 0]] - fc[adj[:, 1]], axis=1)
    d = (delta * d_geo / max(float(d_geo.mean()), 1e-12)
         + (1 - delta) * d_ang / max(float(d_ang.mean()), 1e-12))
    wgt = np.exp(-d / max(float(d.mean()), 1e-12))
    nf = len(w.faces)
    g = sparse.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(nf, nf))
    ncomp, comp = connected_components(g, directed=False)
    if ncomp > 1:
        diag = float(np.linalg.norm(np.ptp(w.vertices, axis=0)))
        pairs = cKDTree(fc).query_pairs(r=0.03 * diag, output_type="ndarray")
        if len(pairs):
            pairs = pairs[comp[pairs[:, 0]] != comp[pairs[:, 1]]]
        if len(pairs):
            adj = np.concatenate([adj, pairs])
            wgt = np.concatenate([wgt, np.full(len(pairs), 0.25 * float(np.median(wgt)))])
    return adj, wgt


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
    lab = _fold_isolated(w, lab)
    return Patches(w, lab, int(lab.max()) + 1)


def _fold_isolated(w, lab: np.ndarray) -> np.ndarray:
    """Give every patch that touches no other patch (a disconnected scrap of the mesh) to the patch
    nearest to it in space, so no candidate partition offers an invisible speck as a region."""
    from scipy.spatial import cKDTree

    adj = np.asarray(w.face_adjacency)
    n = int(lab.max()) + 1
    if n < 2:
        return lab
    a, b = lab[adj[:, 0]], lab[adj[:, 1]]
    touching = np.zeros(n, bool)
    touching[a[a != b]] = True
    touching[b[a != b]] = True
    area = np.bincount(lab, weights=w.area_faces, minlength=n)
    lonely = [p for p in range(n) if not touching[p] and area[p] < 0.5 * area.max()]
    if not lonely or len(lonely) == n:
        return lab
    fc = w.triangles_center
    others = ~np.isin(lab, lonely)
    tree = cKDTree(fc[others])
    other_lab = lab[others]
    lab = lab.copy()
    for p in lonely:
        sel = lab == p
        _, nn = tree.query(fc[sel], k=1)
        lab[sel] = np.bincount(other_lab[nn]).argmax()
    return np.unique(lab, return_inverse=True)[1]


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
    touching groups with the best score: the cut affinity relative to the smaller group's volume,
    times a thickness-similarity factor. The factor is what keeps a thin band from joining a bulky
    cup before it joins the rest of the band, when a scan hole makes the band's own pieces only
    weakly connected (local thickness, the shape diameter function, is the classic part cue).
    Returns {group count: group id per patch}, with group 0 the largest by area in each partition.
    Patches that touch nothing are merged last, into the largest group."""
    adj, wgt = _affinity(p.work)
    nf = len(p.work.faces)
    deg = np.bincount(adj[:, 0], weights=wgt, minlength=nf) + np.bincount(adj[:, 1], weights=wgt, minlength=nf)
    area = p.work.area_faces
    thick = _face_thickness(p.work)
    group = list(range(p.n))
    out: dict[int, list[int]] = {}
    while True:
        out[len(set(group))] = _by_area(group, p.labels, area)
        if len(set(group)) <= 1:
            break
        g_face = np.asarray(group)[p.labels]
        vol = np.bincount(g_face, weights=deg, minlength=p.n)
        t_med = {g: _weighted_median(thick[g_face == g], area[g_face == g]) for g in set(group)}
        a, b = g_face[adj[:, 0]], g_face[adj[:, 1]]
        cross = a != b
        cut: dict[tuple[int, int], float] = {}
        for x, y, ww in zip(a[cross], b[cross], wgt[cross], strict=True):
            key = (int(min(x, y)), int(max(x, y)))
            cut[key] = cut.get(key, 0.0) + float(ww)
        if cut:
            tm, vl = t_med, vol

            def score(kv, tm=tm, vl=vl):
                (x, y), c = kv
                ratio = abs(np.log(max(tm[x], 1e-6) / max(tm[y], 1e-6)))
                return c / max(min(vl[x], vl[y]), 1e-12) * np.exp(-ratio / THICKNESS_SCALE)
            (x, y), _ = max(cut.items(), key=score)
        else:  # disconnected groups: fold the smallest into the largest
            ga = np.bincount(g_face, weights=area, minlength=p.n)
            live = sorted(set(group), key=lambda g: ga[g])
            x, y = live[-1], live[0]
        group = [x if g == y else g for g in group]
    return out


THICKNESS_SCALE = 0.3   # a 1.8x thickness ratio cuts a merge score to about 0.14x


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


def _face_thickness(w) -> np.ndarray:
    """Local thickness per face: distance along the inward normal to the opposite surface (Open3D
    raycaster). Faces whose ray escapes (open scans) get the median."""
    import open3d as o3d

    v = np.asarray(w.vertices, np.float32)
    f = np.asarray(w.faces, np.uint32)
    fc = np.asarray(w.triangles_center, np.float32)
    fn = np.asarray(w.face_normals, np.float32)
    diag = float(np.linalg.norm(np.ptp(v, axis=0))) or 1.0
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
    rays = np.concatenate([fc - fn * (1e-3 * diag), -fn], axis=1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    ok = np.isfinite(t) & (t < 0.5 * diag)
    fill = float(np.median(t[ok])) if ok.any() else diag
    return np.where(ok, t, fill)


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


def submeshes(mesh, face_labels: np.ndarray, min_frac: float = 0.005) -> dict[int, object]:
    """{group id: trimesh} from per-face labels, keeping vertex colours. A group with under
    ``min_frac`` of the faces is not a part: its faces join the nearest other group."""
    from scipy.spatial import cKDTree

    face_labels = np.asarray(face_labels).copy()
    counts = np.bincount(face_labels)
    tiny = [g for g in np.flatnonzero(counts) if counts[g] < min_frac * len(face_labels)]
    big = ~np.isin(face_labels, tiny)
    if tiny and big.any():
        fc = np.asarray(mesh.triangles_center)
        _, nn = cKDTree(fc[big]).query(fc[~big], k=1)
        face_labels[~big] = face_labels[big][nn]
    out = {}
    for g in np.unique(face_labels):
        sub = mesh.submesh([np.flatnonzero(face_labels == g)], append=True)
        if len(sub.faces):
            out[int(g)] = sub
    return out


def candidate_counts(p: Patches, hierarchy: dict[int, list[int]], max_parts: int = MAX_PARTS,
                     min_area: float = MIN_PATCH_AREA) -> list[int]:
    """Region counts worth offering: 2..max_parts where every region is a visible share of the
    surface (at least ``min_area``), so no option shows an invisible region."""
    area = p.work.area_faces
    total = float(area.sum()) or 1.0
    out = []
    for k in range(2, max_parts + 1):
        if k not in hierarchy:
            continue
        g = np.asarray(hierarchy[k])[p.labels]
        if np.bincount(g, weights=area, minlength=k).min() >= min_area * total:
            out.append(k)
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


def render_patch_grid(p: Patches, *, view: int = 240, cols: int = 2) -> bytes:
    """PNG grid with one tile per patch: the whole object in grey with that one patch in red, from
    the SAME two sides in every tile, with the patch number in the corner. Identical viewpoints let
    a VLM tell which side of the object each patch is on (with a per-tile camera, "the near cup"
    meant a different cup in different tiles and it paired up the wrong ones)."""
    from PIL import Image, ImageDraw, ImageFont

    w = p.work
    grey = np.array([150.0, 150.0, 150.0])
    red = np.array([235.0, 45.0, 45.0])
    try:
        font = ImageFont.load_default(size=view // 6)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    tiles = []
    for pid in range(p.n):
        rgb = np.where((p.labels == pid)[:, None], red, grey)
        img = Image.open(io.BytesIO(_render(w, rgb, size=view))).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width - 1, img.height - 1], outline=(90, 90, 100), width=2)
        dr.text((10, 6), str(pid), fill=(255, 255, 255), font=font, stroke_width=3,
                stroke_fill=(0, 0, 0))
        tiles.append(np.asarray(img))
    rows = (len(tiles) + cols - 1) // cols
    tiles += [np.full_like(tiles[0], 24)] * (rows * cols - len(tiles))
    grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows)],
                          axis=0)
    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format="PNG")
    return buf.getvalue()


_SIDE_WORDS = {"left", "right", "front", "back", "rear", "top", "bottom", "upper", "lower", "inner",
               "outer", "first", "second", "third", "one", "two", "a", "b", "l", "r"}


def part_type(label: str) -> str:
    """The kind of a part, without the words that only tell identical parts apart: "Left Ear Cup",
    "right ear cups" and "Ear cup 2" are all "ear cup"."""
    import re

    words = re.findall(r"[a-z]+", (label or "").lower())
    words = [w for w in words if w not in _SIDE_WORDS]
    if words and len(words[-1]) > 3 and words[-1].endswith("s"):
        words[-1] = words[-1][:-1]
    return " ".join(words) or (label or "").lower()


def regroup_instances(p: Patches, groups: list[list[int]], labels: list[str]) -> list[list[int]]:
    """Re-derive which patches form which copy of a repeated part.

    A VLM reliably says what kind of part each patch is and how many copies there are (two ear
    cups), but not which copy a patch belongs to: on a real scan it paired the two cushion tops as
    one cup and the two side walls as the other. Location settles that. For every part kind with m
    copies, split the surface of all its patches into m regions by where it is (area-weighted
    k-means on face centres, farthest-point seeded, so a stray sliver cannot become a cluster of its
    own), give each patch to the region that holds most of its area, then hand each region to the
    copy whose original patches it overlaps most. Kinds with one copy are left as they are."""
    area = p.work.area_faces
    fc = p.work.triangles_center
    out = [list(g) for g in groups]
    by_type: dict[str, list[int]] = {}
    for gi, lb in enumerate(labels):
        by_type.setdefault(part_type(lb), []).append(gi)
    for idxs in by_type.values():
        m = len(idxs)
        members = sorted({q for gi in idxs for q in groups[gi] if isinstance(q, int) and 0 <= q < p.n})
        if m < 2 or len(members) < m:
            continue
        sel = np.isin(p.labels, members)
        pts, wts, owners = fc[sel], area[sel], p.labels[sel]
        patch_area = np.bincount(p.labels, weights=area, minlength=p.n)
        cl = _weighted_kmeans(pts, wts, m, seed_groups=[
            [int(q) for q in sorted(members, key=lambda q: -patch_area[q])[:m]]], owners=owners)
        share = np.zeros((p.n, m))
        np.add.at(share, (owners, cl), wts)
        clusters = [[q for q in members if int(np.argmax(share[q])) == k] for k in range(m)]
        if any(not c for c in clusters):
            continue
        free = list(range(m))
        for gi in sorted(idxs, key=lambda g: -len(groups[g])):
            best = max(free, key=lambda k: len(set(clusters[k]) & set(groups[gi])))
            out[gi] = clusters[best]
            free.remove(best)
    return out


def _weighted_kmeans(pts: np.ndarray, w: np.ndarray, k: int, iters: int = 30, *,
                     seed_groups: list[list[int]] | None = None,
                     owners: np.ndarray | None = None) -> np.ndarray:
    """Deterministic area-weighted k-means. Lloyd runs from each seeding (the centres of the k
    largest patches, then farthest-point seeds) and the split with the lowest weighted squared
    error wins. Farthest-point seeding alone puts a centre on a stray far sliver and keeps it there
    (on a real scan it split one sliver off and left both ear cups together)."""
    starts = []
    if seed_groups and owners is not None:
        for grp in seed_groups:
            if len(grp) == k:
                starts.append(np.array([np.average(pts[owners == q], axis=0, weights=w[owners == q])
                                        for q in grp]))
    c0 = np.average(pts, axis=0, weights=w)
    seeds = [int(np.argmax(np.linalg.norm(pts - c0, axis=1)))]
    while len(seeds) < k:
        d = np.min(np.linalg.norm(pts[:, None, :] - pts[seeds][None, :, :], axis=2), axis=1)
        seeds.append(int(np.argmax(d)))
    starts.append(pts[seeds].astype(float))
    best_lab, best_sse = None, np.inf
    for cent in starts:
        lab = np.zeros(len(pts), np.int64)
        for _ in range(iters):
            lab = np.argmin(np.linalg.norm(pts[:, None, :] - cent[None, :, :], axis=2), axis=1)
            new = np.array([np.average(pts[lab == j], axis=0, weights=w[lab == j]) if (lab == j).any()
                            else cent[j] for j in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        sse = float(np.sum(w * np.sum((pts - cent[lab]) ** 2, axis=1)))
        if sse < best_sse:
            best_lab, best_sse = lab, sse
    return best_lab


def grouping_to_owner(p: Patches, groups: list[list[int]]) -> list[int] | None:
    """Complete patch -> part map (indices into ``groups``) from a VLM reply. A patch listed twice
    keeps its first part; a patch listed nowhere joins the part of the touching patch it shares the
    longest boundary with (or the nearest patch, if it touches none). None if nothing valid."""
    owner = [-1] * p.n
    for gi, members in enumerate(groups):
        for m in members:
            if isinstance(m, int) and 0 <= m < p.n and owner[m] < 0:
                owner[m] = gi
    if all(o < 0 for o in owner):
        return None
    shared = patch_adjacency(p)
    cent = np.array([p.work.triangles_center[p.labels == i].mean(axis=0) for i in range(p.n)])
    for _ in range(p.n):
        missing = [i for i in range(p.n) if owner[i] < 0]
        if not missing:
            break
        for i in missing:
            best, best_len = -1, 0.0
            for (x, y), ln in shared.items():
                j = y if x == i else x if y == i else None
                if j is not None and owner[j] >= 0 and ln > best_len:
                    best, best_len = owner[j], ln
            if best < 0:
                known = [j for j in range(p.n) if owner[j] >= 0]
                best = owner[min(known, key=lambda j: float(np.linalg.norm(cent[j] - cent[i])))]
            owner[i] = best
    return owner
