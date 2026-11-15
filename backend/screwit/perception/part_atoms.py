"""Part segmentation for any object: geometric patches, grouped and named by the VLM (P3, 6.2).

Geometry alone cannot decide what a "part" is (two touching ear cups have the same thickness; a
pizza-box lid and base meet in a seam that looks like any other edge), and a VLM cannot draw a
precise boundary on a mesh. So the work is split:

1. ``over_segment`` cuts the mesh into about 10 patches along its creases and necks: spectral
   clustering of the face graph, with edges across concave creases weighted down (part boundaries
   sit in concave creases, the "minima rule"). Each patch should lie inside one part.
2. ``render_patch_tiles`` draws one image per patch: the whole object in grey, that patch in red,
   from the same two sides in every image. The caller asks the VLM once to group the patch numbers
   into named parts. The VLM is reliable about what kind of part a patch is and how many copies
   exist, not about which copy (it paired two cushion tops as one ear cup), so ``regroup_instances``
   re-splits every repeated kind by location, and ``grouping_to_owner`` completes the map.
   (Three designs failed on real scans first: one image of numbered colours, which the VLM misread,
   choosing among geometric merges, which offered no correct option once a scan hole split the
   headband, and all tiles drawn into one numbered grid, where it mixed up which tile was which.)
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
    face count, and 8k faces resolve parts far below the scan's noise. The cover the mesh stage
    puts under the object (down-facing faces on the support, recon/completion.py) is left out: no
    camera saw it, it carries no part boundaries, and its large flat discs pulled the patches
    around. Its faces still get a part (``full_face_labels`` gives every face the nearest one's)."""
    import open3d as o3d
    import trimesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if len(f) > 100:
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        cover = (nz < -0.9) & (tri[:, :, 2].max(axis=1) < v[:, 2].min() + 1.5)
        if 0 < cover.sum() < 0.5 * len(f):
            f = f[~cover]
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
    of neighbour-majority smoothing so the boundaries are clean. Faces of the cover under the object
    (left out of the working mesh, see ``work_mesh``) take the part directly above them instead: the
    nearest face to a point under a cup's middle can be another part's wall."""
    import open3d as o3d
    from scipy import sparse
    from scipy.spatial import cKDTree

    tri = np.asarray(mesh.triangles_center)
    _, nn = cKDTree(p.work.triangles_center).query(tri, k=1)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    fn = np.asarray(mesh.face_normals)
    cover = (fn[:, 2] < -0.9) & (v[np.asarray(mesh.faces)][:, :, 2].max(axis=1) < v[:, 2].min() + 1.5)
    if cover.any():
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(p.work.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(p.work.faces, np.uint32)))
        start = tri[cover] + [0.0, 0.0, 0.5]
        rays = np.concatenate([start, np.tile([0.0, 0.0, 1.0], (len(start), 1))], axis=1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        ids = hit["primitive_ids"].numpy().astype(np.int64)
        ok = np.isfinite(hit["t_hit"].numpy()) & (ids >= 0) & (ids < len(p.work.faces))
        sel = np.flatnonzero(cover)[ok]
        nn[sel] = ids[ok]
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


# ------------------------------------------------------------------------------- connector fold

# A connector is a small part wedged against larger parts (a hinge between a headband and an ear
# cup, a bracket between two panels). The segmenter and the VLM both tend to call it its own part,
# but it physically belongs to the neighbour it is attached to. These thresholds decide "small and
# tucked in" from geometry alone -- no object category, no label text (AGENTS.md).
CONNECTOR_MAX_AREA = 0.15     # a part above 15% of the surface is never treated as a connector
CONNECTOR_MIN_BURIED = 0.12   # at least this share of its surface sits against other parts
CONNECTOR_NEAR_FRAC = 0.02    # "against another part" = within 2% of the object diagonal


def _group_face_labels(p: Patches, owner: list[int]) -> np.ndarray:
    """Group id per working face from a patch->group ``owner`` map."""
    return np.asarray(owner, dtype=np.int64)[p.labels]


def _group_buried_fraction(p: Patches, gface: np.ndarray, g: int, diag: float) -> float:
    """Area-weighted share of group ``g``'s faces that sit within ``CONNECTOR_NEAR_FRAC`` of the
    object diagonal of some OTHER group's surface. A connector is tucked between parts, so most of
    it is close to another part; a free-standing small part (a knob on a shaft) is mostly exposed,
    so this is small. Purely geometric -- it knows nothing about what the part is."""
    from scipy.spatial import cKDTree

    fc = p.work.triangles_center
    area = p.work.area_faces
    mine = gface == g
    if not mine.any() or mine.all():
        return 0.0
    tree = cKDTree(fc[~mine])
    d, _ = tree.query(fc[mine])
    near = d < CONNECTOR_NEAR_FRAC * diag
    my_area = area[mine]
    denom = float(my_area.sum()) or 1.0
    return float(my_area[near].sum()) / denom


def _group_shared_boundary(p: Patches, owner: list[int]) -> dict:
    """Inter-group shared-boundary length, keyed (min,max) group id, aggregated from the patch
    adjacency. The host a connector folds into is the group it shares the longest seam with."""
    shared = patch_adjacency(p)
    out: dict[tuple[int, int], float] = {}
    for (x, y), ln in shared.items():
        gx, gy = owner[x], owner[y]
        if gx != gy:
            out[(min(gx, gy), max(gx, gy))] = out.get((min(gx, gy), max(gx, gy)), 0.0) + ln
    return out


def fold_connectors(p: Patches, owner: list[int], *, min_groups: int = 2) -> list[int]:
    """``fold_connectors_cut`` without kinds: every connector folds whole, the patches stay as they
    are, and only the new patch->group map is returned."""
    return fold_connectors_cut(p, owner, min_groups=min_groups)[1]


def fold_connectors_cut(p: Patches, owner: list[int], *, min_groups: int = 2,
                        kinds: list[str] | None = None) -> tuple[Patches, list[int], list[int]]:
    """Fold small connector groups into the neighbour they are attached to, generically.

    ``kinds`` (optional, the kind key of each group in ``owner``) says which groups are copies of
    the same part (two ear cups, four legs). A connector that borders two or more copies of its
    host's kind is not given whole to one of them: it sits between the copies (the strip where two
    ear cups meet went whole to the cup with the slightly longer seam, which then reached into the
    other cup), so its faces are cut between those copies, each face going to the copy nearest to
    it. Returns (patches, owner, parent): the patches (re-cut when a connector was split), the new
    patch->group map, and for every patch the group it had in the input ``owner``.

    A group is a connector when it is both small (under ``CONNECTOR_MAX_AREA`` of the surface) and
    substantially buried against other parts (``_group_buried_fraction`` over ``CONNECTOR_MIN_BURIED``).
    Such a group is not its own part: it belongs to the neighbour it shares the longest boundary
    with. Folds the smallest connector first and re-evaluates, so a connector that touches another
    connector resolves once its neighbour has settled. Never drops below ``min_groups`` groups and
    never folds a group into one that is itself still a connector in the same pass.

    Returns a new patch->group ``owner`` map, relabelled to contiguous ids 0..k-1 with group 0 the
    largest by area (matching ``merge_hierarchy``/``_by_area``). Pure geometry: no label or category
    is consulted, so it treats a hinge, a bracket and any other wedged connector the same way.
    """
    from scipy.spatial import cKDTree

    owner = list(owner)
    parent = list(owner)
    plabels, n = p.labels.copy(), p.n
    area = p.work.area_faces
    total = float(area.sum()) or 1.0
    diag = float(np.linalg.norm(np.ptp(p.work.vertices, axis=0))) or 1.0
    for _ in range(max(1, len(set(owner)))):
        p = Patches(p.work, plabels, n)
        groups = sorted(set(owner))
        if len(groups) <= min_groups:
            break
        gface = _group_face_labels(p, owner)
        garea = {g: float(area[gface == g].sum()) for g in groups}
        conn = {}
        for g in groups:
            if garea[g] >= CONNECTOR_MAX_AREA * total:
                continue
            buried = _group_buried_fraction(p, gface, g, diag)
            if buried >= CONNECTOR_MIN_BURIED:
                conn[g] = buried
        if not conn:
            break
        # Smallest connector first.
        g = min(conn, key=lambda k: garea[k])
        shared = _group_shared_boundary(p, owner)
        seams = {}
        for (a, b), ln in shared.items():
            if a == g:
                seams[b] = seams.get(b, 0.0) + ln
            elif b == g:
                seams[a] = seams.get(a, 0.0) + ln
        # Prefer a host that is not itself a connector; among those, the longest seam.
        hosts = {h: ln for h, ln in seams.items() if h not in conn} or seams
        if not hosts:
            break
        host = max(hosts, key=hosts.get)
        copies = ([h for h in hosts if kinds[h] == kinds[host]]
                  if kinds is not None and host < len(kinds) else [host])
        if len(copies) < 2:
            owner = [host if o == g else o for o in owner]
            continue
        # Between copies of one kind: cut it, each face to the nearest copy.
        faces = np.flatnonzero(gface == g)
        near_faces = np.flatnonzero(np.isin(gface, copies))
        _, nn = cKDTree(p.work.triangles_center[near_faces]).query(p.work.triangles_center[faces])
        nearest = gface[near_faces][nn]
        for h in copies:
            for q in np.unique(plabels[faces[nearest == h]]):
                mine = faces[(nearest == h) & (plabels[faces] == q)]
                plabels[mine] = n
                owner.append(h)
                parent.append(parent[q])
                n += 1
        owner = [copies[0] if o == g else o for o in owner]   # the emptied patches
    p = Patches(p.work, plabels, n)
    # Relabel contiguous, largest group first (group 0), like merge_hierarchy's partitions.
    gface = _group_face_labels(p, owner)
    live = sorted(set(owner), key=lambda g: -float(area[gface == g].sum()))
    remap = {g: i for i, g in enumerate(live)}
    return p, [remap[o] for o in owner], parent


# ------------------------------------------------------------------------------- paired sub-parts

LAYER_SHARE = 0.7   # share of a repeated kind's boundary that must be with its host kind


def merge_paired_subparts(p: Patches, owner: list[int], kinds: list[str]) -> list[int]:
    """Merge a repeated sub-part into the repeated part it sits on, generically.

    A kind with two or more copies whose boundary is mostly (``LAYER_SHARE``) with ONE other kind,
    where that other kind has the same number of copies, connects to the rest of the object and
    leans on this kind less than this kind leans on it, is a layer of that part
    rather than a component of its own: the VLM read each ear cup of a headphone scan as "ear cup"
    plus "ear cushion" and the cut between the two followed a crease, not the cushion, so neither
    piece was a real cushion. Each copy joins the copy it shares the longest boundary with; nothing
    is merged unless that pairs the copies one to one. Pure geometry plus the VLM's own kind keys
    (``kinds``, one per group in ``owner``): no category or label text is consulted. Parts that
    only touch each other (a lid on a box) and single parts (a knob on a door) are kept.

    Returns the new patch->group map; group ids are those of ``owner`` (merged groups vanish)."""
    owner = list(owner)
    for _ in range(len(set(kinds))):
        groups = sorted(set(owner))
        by_kind: dict[str, list[int]] = {}
        for g in groups:
            by_kind.setdefault(kinds[g], []).append(g)
        seams = _group_shared_boundary(p, owner)

        def neighbours(gs: list[int], seams: dict = seams) -> dict[int, float]:
            out: dict[int, float] = {}
            for (a, b), ln in seams.items():
                for x, y in ((a, b), (b, a)):
                    if x in gs and y not in gs:
                        out[y] = out.get(y, 0.0) + ln
            return out

        merged = False
        for kind, members in by_kind.items():
            if len(members) < 2:
                continue
            near = neighbours(members)
            total = sum(near.values())
            by_host: dict[str, float] = {}
            for g, ln in near.items():
                by_host[kinds[g]] = by_host.get(kinds[g], 0.0) + ln
            if not total:
                continue
            host_kind = max(by_host, key=by_host.get)
            share = by_host[host_kind] / total
            hosts = by_kind[host_kind]
            if share < LAYER_SHARE or len(hosts) != len(members):
                continue
            back = neighbours(hosts)
            back_total = sum(back.values())
            back_share = sum(ln for g, ln in back.items() if kinds[g] == kind) / (back_total or 1.0)
            if back_share >= share or back_share > 1.0 - 1e-9:
                continue        # the pair only leans on itself: two parts, not a layer of one
            pairing = {}
            for g in members:
                own = {h: ln for (a, b), ln in seams.items() for h in (a, b)
                       if g in (a, b) and h != g and h in hosts}
                if own:
                    pairing[g] = max(own, key=own.get)
            if len(pairing) != len(members) or len(set(pairing.values())) != len(members):
                continue
            owner = [pairing.get(o, o) for o in owner]
            merged = True
            break
        if not merged:
            break
    return owner


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


def render_patch_tiles(p: Patches, *, view: int = 320) -> list[bytes]:
    """One PNG per patch: the whole object in grey with that one patch in red, from the SAME two
    sides in every image. Identical viewpoints let a VLM tell which side of the object each patch
    is on (with a per-patch camera, "the near cup" meant a different cup in different images and it
    paired up the wrong ones). The caller sends them as separate images, each introduced by its
    number: drawn into one numbered grid, the VLM misread which tile was which."""
    grey = np.array([150.0, 150.0, 150.0])
    red = np.array([235.0, 45.0, 45.0])
    return [_render(p.work, np.where((p.labels == pid)[:, None], red, grey), size=view)
            for pid in range(p.n)]


SPLIT_SHARE = 0.25   # a patch with this share of its area in a second copy is cut, not handed whole


def _location_clusters(p: Patches, members: list[int], m: int) -> list[list[int]]:
    """``members`` split into ``m`` groups by where they are (patch-level majority of the
    area-weighted k-means used by ``regroup_instances``)."""
    sel = np.flatnonzero(np.isin(p.labels, members))
    patch_area = np.bincount(p.labels, weights=p.work.area_faces, minlength=p.n)
    cl = _weighted_kmeans(p.work.triangles_center[sel], p.work.area_faces[sel], m, seed_groups=[
        [int(q) for q in sorted(members, key=lambda q: -patch_area[q])[:m]]], owners=p.labels[sel])
    share = np.zeros((p.n, m))
    np.add.at(share, (p.labels[sel], cl), p.work.area_faces[sel])
    return [[q for q in members if int(np.argmax(share[q])) == k] for k in range(m)]


def kinds_to_groups(p: Patches, kinds: list, notes: list) -> tuple[list[list[int]], list[str], list]:
    """Turn a VLM reading (``kinds``: objects with ``.name`` and ``.copies``; ``notes``: objects
    with ``.patch`` and ``.kind``) into one patch list per copy, the kind key of each list, and the
    kind object behind it. Every patch list of a kind starts as all of its patches; which copy owns
    which patch is ``regroup_instances``' job.

    A repeated kind is only kept when it can be cut out of every copy. That fails when the kind has
    fewer patches than copies (one ear cup's pad was its own patch, the other cup's pad was inside
    a whole-cup patch), or when its copies would not sit on different copies of a larger host with
    as many copies (a pad's top and a side wall of the same cup, called the two "ear pads"). Two
    failing kinds with the same number of copies are first tried as one kind named two ways (one
    cup read as "ear cup", the other as "ear cushion"); otherwise the failing kind's patches join
    the kind they share the longest boundary with, so identical components always come out the
    same. Patches with no valid kind are left out (``grouping_to_owner`` gives them to a
    neighbour)."""
    by_key = {}
    for k in kinds:
        key = (k.name or "").strip().lower()
        if key and key not in by_key:
            by_key[key] = k
    kind_of: dict[int, str] = {}
    for note in notes:
        key = (note.kind or "").strip().lower()
        if isinstance(note.patch, int) and 0 <= note.patch < p.n and note.patch not in kind_of \
                and key in by_key:
            kind_of[note.patch] = key
    shared = patch_adjacency(p)

    def members(key: str) -> list[int]:
        return sorted(q for q, v in kind_of.items() if v == key)

    def border(qs: list[int], others: list[int] | None = None) -> dict:
        """Shared boundary length from patches ``qs`` to each other kind (or to ``others``)."""
        qset = set(qs)
        out: dict = {}
        for (a, b), ln in shared.items():
            for x, y in ((a, b), (b, a)):
                if x in qset and y not in qset and y in kind_of:
                    if others is None and kind_of[y] != kind_of[x]:
                        out[kind_of[y]] = out.get(kind_of[y], 0.0) + ln
                    elif others is not None and y in others:
                        out[y] = out.get(y, 0.0) + ln
        return out

    def cut_out_of_every_copy(key: str) -> bool:
        mine, m = members(key), int(by_key[key].copies)
        if m < 2:
            return True
        if len(mine) < m:
            return False
        # The host is the larger kind it shares the longest boundary with: a pad sits on a cup;
        # a cup is not checked against a small hinge between the two cups.
        near = {k: ln for k, ln in border(mine).items() if kind_area(k) > kind_area(key)}
        if not near:
            return True
        host = max(near, key=near.get)
        host_members, hm = members(host), int(by_key[host].copies)
        if hm != m or len(host_members) < hm:
            return True     # only a pairing of copies with copies can be checked
        host_copies = _location_clusters(p, host_members, hm)
        seats = set()
        for copy in _location_clusters(p, mine, m):
            lens = [sum(border(copy, hc).values()) for hc in host_copies]
            seats.add(int(np.argmax(lens)) if max(lens) > 0 else -1 - len(seats))
        return len(seats) == m

    patch_area = np.bincount(p.labels, weights=p.work.area_faces, minlength=p.n)

    def kind_area(key: str) -> float:
        return float(sum(patch_area[q] for q in members(key)))

    for _ in range(2 * len(by_key)):
        bad = [k for k in by_key if members(k) and not cut_out_of_every_copy(k)]
        if not bad:
            break
        # Two kinds that each fail with the same number of copies may be one kind named two ways
        # (one ear cup read as "ear cup", the other as "ear cushion"): unite them if together
        # they can be cut out of every copy, under the name of the larger.
        united = False
        for a in sorted(bad, key=kind_area):
            for b in bad:
                if b == a or int(by_key[a].copies) != int(by_key[b].copies):
                    continue
                before = dict(kind_of)
                bigger = by_key[a] if kind_area(a) > kind_area(b) else by_key[b]
                for q in members(a):
                    kind_of[q] = b
                if cut_out_of_every_copy(b):
                    by_key[b] = bigger
                    united = True
                    break
                kind_of.clear()
                kind_of.update(before)
            if united:
                break
        if united:
            continue
        # Otherwise the smallest failing kind joins the kind it shares the longest boundary with,
        # and the united kind is named after whichever of the two covers more surface (two ear
        # cups folded into a small "hinge" kind are still ear cups).
        key = min(bad, key=kind_area)
        near = border(members(key))
        if not near:     # touches no other kind: keep it whole, as one part
            by_key[key] = _with_copies(by_key[key], 1)
            continue
        target = max(near, key=near.get)
        if kind_area(key) > kind_area(target):
            by_key[target] = _with_copies(by_key[key], int(by_key[target].copies))
        for q in members(key):
            kind_of[q] = target
    groups, keys, objs = [], [], []
    for key, kind in by_key.items():
        mine = members(key)
        for _ in range(min(max(1, int(kind.copies)), len(mine))):
            groups.append(list(mine))
            keys.append(key)
            objs.append(kind)
    return groups, keys, objs


def _with_copies(kind, copies: int):
    try:
        return kind.model_copy(update={"copies": copies})
    except AttributeError:   # a plain object in tests
        kind.copies = copies
        return kind


def regroup_instances(p: Patches, groups: list[list[int]],
                      kinds: list[str]) -> tuple[Patches, list[list[int]]]:
    """Decide which patches form which copy of a repeated part, from where they are.

    ``groups`` are patch lists and ``kinds`` the kind key of each; the lists of one kind are its
    copies. A VLM reliably says what kind of part each patch is and how many copies there are (two
    ear cups), but not which copy a patch belongs to: on a real scan it paired the two cushion tops
    as one cup and the two side walls as the other. For every kind with m copies, split the surface
    of all its patches into m regions by location (area-weighted k-means on face centres, seeded at
    the m largest patches, so a stray sliver cannot become a cluster of its own), give each patch to
    the region that holds most of its area, then hand each region to the copy whose list it
    overlaps most. Kinds with one copy are left as they are.

    A patch with at least SPLIT_SHARE of its area in a second region is cut along the regions
    instead (its pieces become new patches): a strip along the bases of two touching ear cups went
    whole to one cup, which then showed a sliver under the other. Returns the (possibly re-cut)
    patches and the patch lists, one per group."""
    area = p.work.area_faces
    fc = p.work.triangles_center
    plabels = p.labels.copy()
    n = p.n
    out = [list(g) for g in groups]
    by_kind: dict[str, list[int]] = {}
    for gi, key in enumerate(kinds):
        by_kind.setdefault(key, []).append(gi)
    for idxs in by_kind.values():
        m = len(idxs)
        members = sorted({q for gi in idxs for q in groups[gi] if isinstance(q, int) and 0 <= q < p.n})
        if m < 2 or len(members) < m:
            continue
        sel = np.flatnonzero(np.isin(plabels, members))
        pts, wts, owners = fc[sel], area[sel], plabels[sel]
        patch_area = np.bincount(plabels, weights=area, minlength=n)
        cl = _weighted_kmeans(pts, wts, m, seed_groups=[
            [int(q) for q in sorted(members, key=lambda q: -patch_area[q])[:m]]], owners=owners)
        share = np.zeros((n, m))
        np.add.at(share, (owners, cl), wts)
        main = {q: int(np.argmax(share[q])) for q in members}
        clusters = [[q for q in members if main[q] == k] for k in range(m)]
        if any(not c for c in clusters):
            continue
        for q in members:
            for k in range(m):
                if k != main[q] and share[q, k] >= SPLIT_SHARE * share[q].sum():
                    plabels[sel[(owners == q) & (cl == k)]] = n     # this piece becomes patch n
                    clusters[k].append(n)
                    n += 1
        free = list(range(m))
        for gi in sorted(idxs, key=lambda g: -len(groups[g])):
            best = max(free, key=lambda k: len(set(clusters[k]) & set(groups[gi])))
            out[gi] = clusters[best]
            free.remove(best)
    return Patches(p.work, plabels, n), out


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


def choose_grouping(p: Patches, replies: list) -> tuple[Patches, list[int], list] | None:
    """Pick the most typical of several VLM readings (objects with ``.kinds`` and
    ``.patch_notes``, see labels.PatchGrouping) and complete it. Returns (patches, owner, kind per
    part), where the patches are re-cut by ``regroup_instances``, owner maps each patch to a part
    index and the list gives each part's kind object; None if no reply gives a valid map.

    Every reply is turned into copies (``kinds_to_groups``) and regrouped by location, then each
    original patch is related to every other one as different kind of part, same kind but another
    copy, or same part. The reply whose relations agree with the other replies' over the most
    surface (pairs weighted by both patches' areas) wins: one stray answer cannot outvote two that
    agree, whichever way it went wrong."""
    area = np.bincount(p.labels, weights=p.work.area_faces, minlength=p.n)
    cands = []
    for r in replies:
        groups, keys, objs = kinds_to_groups(p, r.kinds, r.patch_notes)
        if not groups:
            continue
        p2, lists = regroup_instances(p, groups, keys)
        owner = grouping_to_owner(p2, lists)
        if owner is None:
            continue
        share = np.zeros((p.n, len(lists)))
        np.add.at(share, (p.labels, np.asarray(owner)[p2.labels]), p.work.area_faces)
        part = share.argmax(axis=1)                       # each original patch's main part
        kind = np.array([keys[g] for g in part], dtype=object)
        rel = np.where(part[:, None] == part[None, :], 2, np.where(kind[:, None] == kind[None, :], 1, 0))
        cands.append((p2, owner, objs, rel))
    if not cands:
        return None
    w = np.outer(area, area)
    np.fill_diagonal(w, 0.0)
    score = [sum(float((w * (a[3] == b[3])).sum()) for b in cands if b is not a) for a in cands]
    p2, owner, objs, _ = cands[int(np.argmax(score))]
    return p2, owner, objs


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
