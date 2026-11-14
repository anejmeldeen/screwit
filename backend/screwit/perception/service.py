"""PerceptionService implementation (SPEC 4.3, 4.4, 5 parts routes, 6.2; P3).

Persists parts.json and scene_graph.json through the Storage protocol (tmpdir fake in tests).
All geometry is scan-frame millimetres (SPEC 3.1). Every Part / SceneGraph returned is built
through the Pydantic contracts, so a drift from the contract fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from screwit.contracts import (
    OBB,
    CylinderFeature,
    DerivedDimension,
    Feature,
    Guess,
    Measurement,
    MeasureRequest,
    MeasureResponse,
    MergeRequest,
    OwnDimension,
    Part,
    PartPatch,
    PlaneFeature,
    SceneGraph,
    expected_preferred,
    measurement,
)
from screwit.contracts.models import FEATURE_MEASUREMENT_FIELDS
from screwit.interfaces import FrameSet, PerceptionService, Storage
from screwit.perception import lifting, measure_fit
from screwit.perception.labels import Labeller
from screwit.perception.relations import SceneBuilder
from screwit.perception.render import render_mesh_png
from screwit.perception.snapping import snap_diameter

PARTS_KEY = "scans/{scan_id}/parts.json"
GRAPH_KEY = "scans/{scan_id}/scene_graph.json"


def _target_part_count(names: list[str]) -> int:
    """How many geometry parts to produce: follow the count of parts the VLM proposed (clamped to a
    sensible 2-6), so the split granularity tracks what the object actually has."""
    n = len([x for x in (names or []) if x and x.strip()])
    return max(2, min(6, n or 4))


def _geometry_name_hint(names: list[str]) -> str:
    """A soft naming hint for a geometry part: the VLM names it from the render, but we pass the
    object's proposed part vocabulary so it picks from the right category set."""
    cands = [x for x in (names or []) if x and x.strip()]
    if not cands:
        return "part"
    return "one of: " + ", ".join(cands[:8])


def _norm_label(label: str) -> str:
    """Normalize a part name for same-name comparison: lowercase, strip, drop a trailing digit
    disambiguator and a trailing plural 's', so 'Ear Pads', 'ear pad', 'Ear Pad 2' all match."""
    import re

    s = (label or "").strip().lower()
    s = re.sub(r"\s+\d+$", "", s)       # drop a trailing number (disambiguator)
    s = re.sub(r"s$", "", s)            # crude singularization ('pads' -> 'pad')
    return s


def _duplicate_counts(labels: list[str]) -> dict[str, int]:
    """Count how many groups carry each normalized label (to decide which need a numeric suffix)."""
    counts: dict[str, int] = {}
    for lb in labels:
        k = _norm_label(lb)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _split_by_density_valley(verts: np.ndarray, diag: float) -> list[np.ndarray]:
    """Fallback split for a pool that is one connected lump but is really two lobes joined by a thin
    neck (two ear pads bridged by the headband, with no clean gap the connectivity test can find).

    Project the points onto their principal horizontal axis and histogram them. Two lobes show two
    density peaks with a valley between (the neck). If a clear valley exists -- a low-density bin
    between two substantial peaks -- cut there into two parts. Otherwise keep one part. Conservative:
    only splits on a pronounced, well-separated valley, so a genuinely single part stays one.
    """
    import numpy as np

    if len(verts) < 200:
        return [verts]
    c = verts - verts.mean(0)
    # Principal axis via covariance (the object's longest direction).
    try:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        return [verts]
    axis = vt[0]
    t = c @ axis
    nb = 24
    hist, edges = np.histogram(t, bins=nb)
    # Find the two tallest peaks and the lowest bin strictly between them.
    peak = int(np.argmax(hist))
    # Look for a second peak well away from the first (>= 1/4 of the span).
    far = [i for i in range(nb) if abs(i - peak) >= nb // 4 and hist[i] > 0]
    if not far:
        return [verts]
    peak2 = max(far, key=lambda i: hist[i])
    lo, hi = sorted((peak, peak2))
    between = hist[lo + 1:hi]
    if len(between) == 0:
        return [verts]
    valley = lo + 1 + int(np.argmin(between))
    valley_v = hist[valley]
    peaks_min = min(hist[peak], hist[peak2])
    # A real neck: the valley is much emptier than both peaks and both peaks are substantial.
    if peaks_min < 0.15 * len(verts) / 2 or valley_v > 0.35 * peaks_min:
        return [verts]
    cut = edges[valley + 1]
    left = verts[t < cut]
    right = verts[t >= cut]
    if min(len(left), len(right)) < 100:
        return [verts]
    return [left, right]


def _split_pool_into_lobes(verts: np.ndarray, diag: float) -> list[np.ndarray]:
    """Split one name's pooled vertices into spatially distinct lobes.

    Two physically separate parts that share a name (the two ear pads) are joined in the fused mesh
    only by a thin bridge (the headband running between them). Voxel-downsample the pool, connect
    points within a FINE radius (a few voxels), and take connected components: the dense lobes stay
    connected within themselves, but the thin bridge is too sparse to hold them together at that
    radius, so the two pads fall into separate components. A single connected part (one arch) stays
    one. Components below a small fraction of the pool are reattached to their nearest big lobe so
    stray voxels do not become spurious parts. Returns one vertex array per lobe (>=1).
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(verts) < 50:
        return [verts]
    voxel = max(1.5, 0.01 * diag)            # downsample resolution ~1% of object size
    keys = np.floor(verts / voxel).astype(np.int64)
    _, uniq_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    pts = verts[uniq_idx]                     # one representative point per voxel
    if len(pts) < 8:
        return [verts]
    radius = 2.2 * voxel                      # connect neighbours within ~2 voxels
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return [verts]
    g = csr_matrix((np.ones(len(pairs) * 2),
                    (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                   shape=(len(pts), len(pts)))
    ncomp, comp = connected_components(g, directed=False)
    if ncomp == 1:
        return _split_by_density_valley(verts, diag)
    sizes = np.bincount(comp)
    big = np.where(sizes >= max(8, int(0.15 * sizes.max())))[0]  # keep only sizeable lobes
    if len(big) <= 1:
        return [verts]
    # Map each ORIGINAL vertex to its voxel's component, then to the nearest BIG lobe.
    vox_comp = comp                                   # component per voxel-point
    big_centroids = {c: pts[comp == c].mean(0) for c in big}
    big_ids = list(big_centroids)
    bc = np.array([big_centroids[c] for c in big_ids])
    # For voxels whose component is not big, snap to nearest big lobe centroid.
    comp_to_big = {}
    for c in range(ncomp):
        if c in big_centroids:
            comp_to_big[c] = c
        else:
            cen = pts[comp == c].mean(0)
            comp_to_big[c] = big_ids[int(np.argmin(np.linalg.norm(bc - cen, axis=1)))]
    orig_comp = vox_comp[inverse]
    orig_big = np.array([comp_to_big[c] for c in orig_comp])
    lobes = [verts[orig_big == c] for c in big_ids]
    return [lb for lb in lobes if len(lb) >= 50] or [verts]


def _titlecase_label(label: str) -> str:
    """Capitalize a part label for display (SPEC/UX: object and part names are shown Title Case).

    Capitalizes the first letter of each word but preserves tokens that already carry internal
    capitals or digits (acronyms / codes): "ear cups" -> "Ear Cups", "USB port" -> "USB Port",
    "D-shaft" -> "D-Shaft", "3.5mm jack" -> "3.5mm Jack". Hyphenated words are title-cased per part.
    """
    def _word(w: str) -> str:
        if not w:
            return w
        # Already has an internal capital or is all caps (acronym) -> leave as is but ensure the
        # first alpha is upper.
        if any(c.isupper() for c in w[1:]) or w.isupper():
            return w[0].upper() + w[1:] if w[0].isalpha() else w
        return w[:1].upper() + w[1:]

    parts = []
    for token in (label or "").split(" "):
        parts.append("-".join(_word(seg) for seg in token.split("-")))
    return " ".join(parts).strip() or label


def _resnap(feature_type: str, m: Measurement) -> Measurement:
    """Re-evaluate the snap on a feature radius Measurement, preserving caliper/user_confirmed.

    A caliper value always wins (it is physical) and overrides a snap, so when caliper_mm is set
    the snap is dropped; otherwise the role table decides the snap from the scan radius.
    """
    if m.caliper_mm is not None:
        return measurement(m.scan_mm, caliper_mm=m.caliper_mm, user_confirmed=m.user_confirmed)
    snap = snap_diameter(feature_type, m.scan_mm)
    if snap is None:
        return measurement(m.scan_mm, user_confirmed=m.user_confirmed)
    return measurement(m.scan_mm, snapped_mm=snap.snapped_radius_mm, snap_rule=snap.snap_rule,
                       user_confirmed=m.user_confirmed)


def _scale_measurement(m: Measurement, s: float, feature_type: str) -> Measurement:
    """Analytic /scale on a Measurement: multiply scan_mm by s, keep caliper_mm, re-evaluate snap."""
    scan = None if m.scan_mm is None else m.scan_mm * s
    base = Measurement(scan_mm=scan, snapped_mm=None, snap_rule=None, caliper_mm=m.caliper_mm,
                       user_confirmed=m.user_confirmed,
                       preferred=expected_preferred(scan, None, m.caliper_mm))
    return _resnap(feature_type, base) if feature_type in ("cylinder", "bore") else base


def _cylinder_fit_ok(cf) -> bool:
    """A cylinder fit is a real feature only if most points sit on it, with a loose bound on the
    residual (end-cap vertices inflate the RMSE of a true cylinder: a synthetic 50 mm mug body fits
    with 98% inliers and 3.5 mm RMSE). Measured on a real-model CPU scan of a 100 mm cube: 13%
    inliers and 11 mm RMSE for a "radius" of 50 mm, which this rejects."""
    return (cf.inlier_fraction >= CYLINDER_MIN_INLIERS
            and cf.fit_rmse_mm <= max(1.0, CYLINDER_MAX_RELATIVE_RMSE * cf.radius_mm))


CYLINDER_MIN_INLIERS = 0.5
CYLINDER_MAX_RELATIVE_RMSE = 0.25


class PerceptionImpl(PerceptionService):
    """Parts/measurement/scene-graph service. ``provider`` selects the LLM path (fake in tests).

    No object category is hard-coded here: part names, labels, categories, relations and failure
    modes come from the Labeller/SceneBuilder (Claude, or injected fake data); snapping comes from
    the role-keyed tables in ``snapping`` and simply does not fire for roles with no table.
    """

    def __init__(self, provider: str = "fake", *, labeller: Labeller | None = None,
                 scene_builder: SceneBuilder | None = None, segmenter=None):
        self.labeller = labeller or Labeller(provider)
        self.scene_builder = scene_builder or SceneBuilder(provider)
        self.provider = provider
        # Optional multi-part segmenter (Grounded SAM 2). Injected in tests; in the live path it is
        # created lazily when a SAM 2 checkpoint is configured. None -> single-part fallback.
        self._segmenter = segmenter

    # ---------------------------------------------------------------- persistence helpers

    def _load_parts(self, scan_id: str, storage: Storage) -> list[Part]:
        raw = storage.get_bytes(PARTS_KEY.format(scan_id=scan_id)).decode()
        return [Part.model_validate(p) for p in json.loads(raw)]

    def _save_parts(self, scan_id: str, parts: list[Part], storage: Storage) -> None:
        data = json.dumps([json.loads(p.model_dump_json()) for p in parts], indent=2).encode()
        storage.put_bytes(PARTS_KEY.format(scan_id=scan_id), data, "application/json")

    def _load_graph(self, scan_id: str, storage: Storage) -> SceneGraph:
        raw = storage.get_bytes(GRAPH_KEY.format(scan_id=scan_id)).decode()
        return SceneGraph.model_validate_json(raw)

    def _save_graph(self, scan_id: str, graph: SceneGraph, storage: Storage) -> None:
        storage.put_bytes(GRAPH_KEY.format(scan_id=scan_id), graph.model_dump_json(indent=2).encode(),
                          "application/json")

    # ---------------------------------------------------------------- object mask (6.2 step 1)

    def object_mask(self, frames: FrameSet, object_hint, stored_prompts) -> list[np.ndarray]:
        """One bool mask per keyframe, board excluded.

        Real path: SAM 2 video predictor with stored hero prompts (positive label 1, negative
        label 0), or a Grounding DINO box from the object name plus one negative point on the
        board. Without a GPU the live path uses the model-free GeometricObjectMasker (support
        plane + clusters near the point the cameras orbit). stored_prompts may carry a
        precomputed per-frame mask for the hero scans ({"masks": [HxW bool, ...]}).
        """
        if stored_prompts and "masks" in stored_prompts:
            return [np.asarray(m, dtype=bool) for m in stored_prompts["masks"]]
        from screwit.perception.object_mask import GeometricObjectMasker

        masker = GeometricObjectMasker()
        masks = masker.masks(frames)
        self.last_mask_info = masker.last_info
        return masks

    # ---------------------------------------------------------------- parts (6.2 steps 2-6)

    def parts(self, scan_id: str, frames: FrameSet, masks: list[np.ndarray], mesh_path: Path,
              reconstruction_revision: int, storage: Storage) -> tuple[list[Part], SceneGraph]:
        import trimesh

        names = self.labeller.name_parts(scan_id, frames.full_res_paths).names

        mesh = trimesh.load(mesh_path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        # An empty mesh means an earlier stage found no object surface (commonly: the scale board was
        # not visible and/or the object was not isolated on a flat surface, so the mask kept nothing).
        # Fail here with an actionable message instead of a cryptic numpy error.
        if verts.size == 0 or faces.size == 0:
            raise ValueError(
                "no object surface was reconstructed - the object could not be isolated. "
                "Re-record with the object on a flat surface and the printed scale board lying flat "
                "beside it, fully in view."
            )

        # How to split the object into parts:
        # 1. Precomputed per-frame part-label images (hero fixtures) -> lift them onto the mesh.
        # 2. Otherwise GEOMETRY segmentation of the mesh (CPU, no GPU/model): local thickness +
        #    connectivity. This is the live multi-part path -- SAM 2/3 cannot do part-level on a
        #    single object without a GPU, geometry can. Parts are named from their render by the VLM.
        label_images = getattr(frames, "part_label_images", None)
        if label_images is not None:
            # Hero-fixture path: precomputed per-frame label images lifted onto the mesh.
            label_images = [np.asarray(li, dtype=np.int32) for li in label_images]
            part_names = names
            labels = lifting.vote_labels(verts, label_images, frames.depth_mm, frames.cameras)
            labels = lifting.smooth_labels(labels, faces)
            submeshes = lifting.split_mesh_by_label(mesh, labels)
            parts, part_points = self._build_parts_named_by_order(
                scan_id, submeshes, part_names, reconstruction_revision, storage)
        else:
            # Live geometry path: segment by thickness+connectivity, name each region from its
            # render, then collapse over-splits -- merge regions the VLM gave the SAME name that are
            # also spatially adjacent (the 4 slices of one ear cushion become one), while keeping
            # same-named but SEPARATE regions distinct (the two ear cushions stay two). This yields
            # the natural "Ear Pad 1 / Ear Pad 2 / Headband" instead of five look-alike "Ear Pads".
            from screwit.perception import geo_segment

            submeshes = geo_segment.segment_mesh(mesh, target_parts=_target_part_count(names))
            parts, part_points = self._build_geometry_parts(
                scan_id, mesh, submeshes, names, reconstruction_revision, storage)

        graph = self.scene_builder.build(scan_id, parts, part_points)
        self._save_parts(scan_id, parts, storage)
        self._save_graph(scan_id, graph, storage)
        return parts, graph

    def _build_parts_named_by_order(self, scan_id, submeshes, part_names, recon_rev, storage):
        """Hero-fixture build: one Part per submesh, named by segment order (labels carry meaning)."""
        parts: list[Part] = []
        part_points: dict[str, np.ndarray] = {}
        for label in sorted(submeshes):
            sub = submeshes[label]
            pid = f"p{label}"
            name = part_names[label - 1] if label - 1 < len(part_names) else "part"
            sub_verts = np.asarray(sub.vertices, dtype=float)
            part_points[pid] = sub_verts
            thumb = render_mesh_png(sub_verts, np.asarray(sub.faces, dtype=int),
                                    getattr(sub.visual, "vertex_colors", None))
            glb_key = f"scans/{scan_id}/parts/{pid}.glb"
            png_key = f"scans/{scan_id}/renders/{pid}.png"
            storage.put_bytes(glb_key, sub.export(file_type="glb"), "model/gltf-binary")
            storage.put_bytes(png_key, thumb, "image/png")
            label_info = self.labeller.label_part(scan_id, name, thumb)
            parts.append(self._build_part(scan_id, pid, name, label_info, sub_verts,
                                          recon_rev, glb_key, png_key))
        return parts, part_points

    def _build_geometry_parts(self, scan_id, mesh, submeshes, names, recon_rev, storage):
        """Geometry build: name each segmented region from its render, pool regions by name, split
        each name's pool into spatially-separated lobes (so the two ear pads become two parts even
        though the thickness segmentation mixed them), then emit one Part per lobe.

        Object-agnostic: only VLM names and spatial separation; no object category is hard-coded.
        """
        import numpy as np
        import trimesh

        V = np.asarray(mesh.vertices, dtype=float)
        F = np.asarray(mesh.faces, dtype=np.int64)
        diag = float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 1.0
        vcol = getattr(mesh.visual, "vertex_colors", None)
        vcol = np.asarray(vcol) if vcol is not None else None

        # 1. Name each raw region from its render, and record which mesh vertices it covers (nearest-
        #    vertex of the region's own vertices back onto the full mesh).
        from scipy.spatial import cKDTree

        full_tree = cKDTree(V)
        pooled: dict[str, dict] = {}   # norm-name -> {label, info, vidx:set}
        for label in sorted(submeshes):
            sub = submeshes[label]
            sv = np.asarray(sub.vertices, dtype=float)
            if len(sv) == 0:
                continue
            thumb = render_mesh_png(sv, np.asarray(sub.faces, dtype=int),
                                    getattr(sub.visual, "vertex_colors", None))
            info = self.labeller.label_part(scan_id, _geometry_name_hint(names), thumb)
            key = _norm_label(info.label)
            _, idx = full_tree.query(sv, k=1)
            slot = pooled.setdefault(key, {"label": info.label, "info": info, "vidx": set()})
            slot["vidx"].update(int(i) for i in np.unique(idx))

        # 2. For each name pool, split its vertices into spatially-separated lobes.
        lobes: list[dict] = []  # {info, label, verts_idx (np array of mesh vertex indices)}
        for _key, slot in pooled.items():
            vidx = np.array(sorted(slot["vidx"]), dtype=np.int64)
            for lobe_v in _split_pool_into_lobes(V[vidx], diag):
                # Map the lobe's points back to mesh vertex indices.
                _, li = full_tree.query(lobe_v, k=1)
                lobes.append({"info": slot["info"], "label": slot["label"],
                              "vidx": np.unique(li).astype(np.int64)})

        # 3. Disambiguate duplicate display names (Ear Pad 1 / Ear Pad 2), then build a Part per lobe
        #    by extracting the full-mesh faces whose vertices fall in the lobe.
        name_counts = _duplicate_counts([lb["label"] for lb in lobes])
        seen: dict[str, int] = {}
        # Largest lobes first for stable part ids.
        lobes.sort(key=lambda lb: -len(lb["vidx"]))

        parts: list[Part] = []
        part_points: dict[str, np.ndarray] = {}
        for gi, lb in enumerate(lobes):
            pid = f"p{gi + 1}"
            keep_v = np.zeros(len(V), bool)
            keep_v[lb["vidx"]] = True
            face_mask = keep_v[F].all(axis=1)
            used = np.unique(F[face_mask]) if face_mask.any() else lb["vidx"]
            remap = -np.ones(len(V), np.int64)
            remap[used] = np.arange(len(used))
            sub_faces = remap[F[face_mask]] if face_mask.any() else np.empty((0, 3), np.int64)
            sub_verts = V[used]
            part_points[pid] = sub_verts
            sub_kwargs = {"vertices": sub_verts, "faces": sub_faces, "process": False}
            if vcol is not None:
                sub_kwargs["vertex_colors"] = vcol[used]
            merged = trimesh.Trimesh(**sub_kwargs)
            info = lb["info"]
            base = lb["label"]
            if name_counts.get(_norm_label(base), 0) > 1:
                seen[_norm_label(base)] = seen.get(_norm_label(base), 0) + 1
                disp = f"{base} {seen[_norm_label(base)]}"
            else:
                disp = base
            thumb = render_mesh_png(sub_verts, sub_faces, vcol[used] if vcol is not None else None)
            glb_key = f"scans/{scan_id}/parts/{pid}.glb"
            png_key = f"scans/{scan_id}/renders/{pid}.png"
            storage.put_bytes(glb_key, merged.export(file_type="glb"), "model/gltf-binary")
            storage.put_bytes(png_key, thumb, "image/png")
            part = self._build_part(scan_id, pid, disp, info, sub_verts, recon_rev, glb_key, png_key)
            part = part.model_copy(update={"label": _titlecase_label(disp)})
            parts.append(part)
        return parts, part_points

    def _build_part(self, scan_id, pid, name, label_info, verts, recon_rev, glb_key,
                    png_key) -> Part:
        obb = measure_fit.oriented_bounding_box(verts)
        features, dimensions = self._features_for(pid, name, verts)
        return Part(
            part_id=pid, reconstruction_revision=recon_rev, measurement_revision=1,
            label=_titlecase_label(label_info.label), label_confidence=label_info.label_confidence,
            description=label_info.description, mesh_uri=glb_key, thumbnail_uri=png_key,
            centroid_mm=tuple(verts.mean(axis=0)), obb=self._obb_contract(obb),
            dimensions=dimensions, features=features,
            material_guess=Guess(value=label_info.material, confidence=label_info.material_confidence),
            source="auto")

    def _obb_contract(self, obb):
        return OBB(center=obb.center_mm, extent=obb.extent_mm, rotation_quat=obb.rotation_quat)

    def _features_for(self, pid, name, verts):
        """Fit one primitive feature for the part, choosing the role from geometry (SPEC 6.2 4).

        The role comes from the part's shape, not from any object category: a part that is much
        flatter in one direction than the other two is fitted as a plane; anything else is fitted
        as a cylinder. The name is only a tie-break hint. This keeps the pass object-agnostic (a
        bracket face is a plane, a mug body or a shaft is a cylinder). A person corrects it through
        PATCH, which is why every dimension is a Measurement with its source.
        """
        extent = np.ptp(verts, axis=0) if len(verts) else np.zeros(3)
        srt = np.sort(extent)
        flat = srt[2] > 0 and srt[0] < 0.25 * srt[2]  # one dimension much smaller -> plate/panel
        name_says_plane = any(w in name.lower() for w in ("panel", "plane", "plate", "bracket"))
        if flat or name_says_plane:
            pf = measure_fit.fit_plane(verts)
            feat: Feature = PlaneFeature(
                feature_id=f"{pid}_f1", point_mm=pf.point_mm, normal=pf.normal,
                fit_rmse_mm=pf.fit_rmse_mm, inlier_fraction=pf.inlier_fraction,
                extent_u=measurement(pf.extent_u_mm), extent_v=measurement(pf.extent_v_mm))
            dims = [OwnDimension(name="width", value=measurement(pf.extent_u_mm)),
                    OwnDimension(name="height", value=measurement(pf.extent_v_mm))]
            return [feat], dims
        cf = measure_fit.fit_cylinder(verts)
        if not _cylinder_fit_ok(cf):
            # Not round (a box, a bracket arm, a handle...): no cylinder feature, which would only
            # hold a made-up diameter. Report the tight box dimensions instead, sorted largest
            # first, so the agent and the user still get real numbers.
            obb = measure_fit.oriented_bounding_box(verts)
            ext = sorted(measure_fit.robust_extents(verts, obb), reverse=True)
            dims = [OwnDimension(name=n, value=measurement(float(v)))
                    for n, v in zip(("length", "width", "thickness"), ext, strict=True)]
            return [], dims
        feat = CylinderFeature(
            feature_id=f"{pid}_f1", axis_point_mm=cf.axis_point_mm, axis_dir=cf.axis_dir,
            fit_rmse_mm=cf.fit_rmse_mm, inlier_fraction=cf.inlier_fraction,
            radius=_resnap("cylinder", measurement(cf.radius_mm)),
            length=measurement(cf.length_mm))
        dims = [DerivedDimension(name="outer_diameter", derived_from=f"{pid}_f1.radius", factor=2.0),
                DerivedDimension(name="length", derived_from=f"{pid}_f1.length")]
        return [feat], dims

    # ---------------------------------------------------------------- patch_part (SPEC 5)

    def patch_part(self, scan_id: str, part_id: str, patch: PartPatch, storage: Storage) -> Part:
        parts = self._load_parts(scan_id, storage)
        idx = next(i for i, p in enumerate(parts) if p.part_id == part_id)
        part = parts[idx]
        data = json.loads(part.model_dump_json())

        changed_measurement = False
        if patch.label is not None:
            data["label"] = patch.label
        if patch.description is not None:
            data["description"] = patch.description

        for edit in patch.measurements:
            feat = part.feature(edit.feature_id)
            if feat is None:
                raise ValueError(f"unknown feature {edit.feature_id} on part {part_id}")
            allowed = FEATURE_MEASUREMENT_FIELDS[feat.type]
            if edit.field not in allowed:
                raise ValueError(f"field {edit.field!r} not editable on {feat.type}; allowed {allowed}")
            fdata = next(f for f in data["features"] if f["feature_id"] == edit.feature_id)
            cur = getattr(feat, edit.field)
            caliper = edit.caliper_mm if edit.caliper_mm is not None else cur.caliper_mm
            user_confirmed = edit.user_confirmed if edit.user_confirmed is not None \
                else cur.user_confirmed
            # caliper overrides a snap; a bare user_confirmed keeps any existing snap.
            if caliper is not None:
                new_m = measurement(cur.scan_mm, caliper_mm=caliper, user_confirmed=user_confirmed)
            elif cur.snapped_mm is not None:
                new_m = measurement(cur.scan_mm, snapped_mm=cur.snapped_mm, snap_rule=cur.snap_rule,
                                    user_confirmed=user_confirmed)
            else:
                new_m = measurement(cur.scan_mm, user_confirmed=user_confirmed)
            fdata[edit.field] = json.loads(new_m.model_dump_json())
            changed_measurement = True

        if changed_measurement:
            data["measurement_revision"] = part.measurement_revision + 1
            data["source"] = "user_edited"
        parts[idx] = Part.model_validate(data)
        self._save_parts(scan_id, parts, storage)
        return parts[idx]

    # ---------------------------------------------------------------- merge_parts (SPEC 5)

    def merge_parts(self, scan_id: str, req: MergeRequest, storage: Storage) -> Part:
        parts = self._load_parts(scan_id, storage)
        by_id = {p.part_id: p for p in parts}
        sources = [by_id[pid] for pid in req.part_ids]
        recon_rev = sources[0].reconstruction_revision

        # New id after the current maximum so a retired id is never reused.
        max_n = max(int(p.part_id[1:]) for p in parts if p.part_id[1:].isdigit())
        new_id = f"p{max_n + 1}"

        verts = self._merged_points(scan_id, sources, storage)
        label = req.label or sources[0].label
        label_info = self.labeller.label_part(scan_id, label, b"")
        glb_key = f"scans/{scan_id}/parts/{new_id}.glb"
        png_key = f"scans/{scan_id}/renders/{new_id}.png"
        self._write_merged_mesh(scan_id, new_id, sources, storage, glb_key, png_key)
        merged = self._build_part(scan_id, new_id, label, label_info, verts, recon_rev, glb_key,
                                  png_key)
        merged = merged.model_copy(update={"label": label})

        remaining = [p for p in parts if p.part_id not in req.part_ids] + [merged]
        self._save_parts(scan_id, remaining, storage)

        # Recompute scene-graph relations for the merged part (drop relations to retired ids).
        graph = self._load_graph(scan_id, storage)
        retired = set(req.part_ids)
        pts = {p.part_id: self._part_vertices(scan_id, p, storage) for p in remaining}
        new_graph = self.scene_builder.build(scan_id, remaining, pts)
        graph = graph.model_copy(update={
            "parts": [p.part_id for p in remaining],
            "relations": [r for r in graph.relations
                          if r.a not in retired and r.b not in retired] + [
                r for r in new_graph.relations
                if (r.a == new_id or r.b == new_id)],
            "failure_hypotheses": [h for h in graph.failure_hypotheses if h.part_id not in retired],
        })
        self._save_graph(scan_id, graph, storage)
        self._retired_ids = list(req.part_ids)  # exposed so the API can stale CAD (SPEC 5)
        return merged

    def retired_ids(self) -> list[str]:
        """Source part ids retired by the most recent merge (so the API can mark CAD stale)."""
        return getattr(self, "_retired_ids", [])

    def _merged_points(self, scan_id, sources, storage) -> np.ndarray:
        chunks = [self._part_vertices(scan_id, p, storage) for p in sources]
        chunks = [c for c in chunks if len(c)]
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))

    def _write_merged_mesh(self, scan_id, new_id, sources, storage, glb_key, png_key):
        import trimesh

        meshes = []
        for p in sources:
            try:
                raw = storage.get_bytes(p.mesh_uri)
                meshes.append(trimesh.load(file_obj=_BytesFile(raw), file_type="glb", force="mesh"))
            except Exception:
                continue
        merged = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
        storage.put_bytes(glb_key, merged.export(file_type="glb"), "model/gltf-binary")
        thumb = render_mesh_png(np.asarray(merged.vertices, float), np.asarray(merged.faces, int),
                                getattr(merged.visual, "vertex_colors", None))
        storage.put_bytes(png_key, thumb, "image/png")

    def _part_vertices(self, scan_id, part: Part, storage: Storage) -> np.ndarray:
        import trimesh

        try:
            raw = storage.get_bytes(part.mesh_uri)
            m = trimesh.load(file_obj=_BytesFile(raw), file_type="glb", force="mesh")
            return np.asarray(m.vertices, dtype=float)
        except Exception:
            return np.asarray(part.obb.center, dtype=float).reshape(1, 3)

    # ---------------------------------------------------------------- measure (SPEC 5)

    def measure(self, scan_id: str, req: MeasureRequest, storage: Storage) -> MeasureResponse:
        a = self._resolve_point(scan_id, req.a, storage)
        b = self._resolve_point(scan_id, req.b, storage)
        if isinstance(req.a, str) or isinstance(req.b, str):
            # part-part: closest points on the two full meshes
            pa = self._endpoint_for(scan_id, req.a, b, storage)
            pb = self._endpoint_for(scan_id, req.b, pa, storage)
            a, b = pa, pb
        dist = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        return MeasureResponse(distance_mm=dist, a_mm=tuple(float(x) for x in a),
                               b_mm=tuple(float(x) for x in b))

    def _resolve_point(self, scan_id, spec, storage) -> np.ndarray:
        if isinstance(spec, str):
            parts = {p.part_id: p for p in self._load_parts(scan_id, storage)}
            return np.asarray(parts[spec].centroid_mm, dtype=float)
        return np.asarray(spec, dtype=float)

    def _endpoint_for(self, scan_id, spec, toward: np.ndarray, storage) -> np.ndarray:
        if not isinstance(spec, str):
            return np.asarray(spec, dtype=float)
        parts = {p.part_id: p for p in self._load_parts(scan_id, storage)}
        pts = self._part_vertices(scan_id, parts[spec], storage)
        if len(pts) == 0:
            return np.asarray(parts[spec].centroid_mm, dtype=float)
        d = np.linalg.norm(pts - toward[None, :], axis=1)
        return pts[int(d.argmin())]

    # ---------------------------------------------------------------- rescale_parts (SPEC 5)

    def rescale_parts(self, scan_id: str, s: float, reconstruction_revision: int,
                      storage: Storage) -> list[Part]:
        parts = self._load_parts(scan_id, storage)
        out: list[Part] = []
        for part in parts:
            data = json.loads(part.model_dump_json())
            data["reconstruction_revision"] = reconstruction_revision
            data["measurement_revision"] = part.measurement_revision + 1
            data["centroid_mm"] = [c * s for c in part.centroid_mm]
            data["obb"]["center"] = [c * s for c in part.obb.center]
            data["obb"]["extent"] = [c * s for c in part.obb.extent]
            for fdata, feat in zip(data["features"], part.features, strict=True):
                for key in ("axis_point_mm", "center_mm", "point_mm"):
                    if key in fdata and fdata[key] is not None:
                        fdata[key] = [c * s for c in fdata[key]]
                for field in FEATURE_MEASUREMENT_FIELDS[feat.type]:
                    fdata[field] = json.loads(
                        _scale_measurement(getattr(feat, field), s, feat.type).model_dump_json())
            for ddata, dim in zip([d for d in data["dimensions"]], part.dimensions, strict=True):
                if isinstance(dim, OwnDimension):
                    ft = "own"
                    ddata["value"] = json.loads(_scale_measurement(dim.value, s, ft).model_dump_json())
            # rescale is physical, not a user edit; keep source as-is.
            data["source"] = part.source
            out.append(Part.model_validate(data))
        self._save_parts(scan_id, out, storage)
        return out

    # ---------------------------------------------------------------- render_view (SPEC 6.3)

    def render_view(self, scan_id: str, part_ids: list[str], angle_deg: float,
                    storage: Storage) -> bytes:
        import trimesh

        parts = {p.part_id: p for p in self._load_parts(scan_id, storage)}
        meshes = []
        for pid in part_ids:
            try:
                raw = storage.get_bytes(parts[pid].mesh_uri)
                meshes.append(trimesh.load(file_obj=_BytesFile(raw), file_type="glb", force="mesh"))
            except Exception:
                continue
        if not meshes:
            return render_mesh_png(np.empty((0, 3)), np.empty((0, 3), dtype=int))
        merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        return render_mesh_png(np.asarray(merged.vertices, float), np.asarray(merged.faces, int),
                               getattr(merged.visual, "vertex_colors", None), angle_deg=angle_deg)


class _BytesFile:
    """Minimal file-like wrapper so trimesh.load can read bytes from storage."""

    def __init__(self, data: bytes):
        import io

        self._b = io.BytesIO(data)

    def __getattr__(self, name):
        return getattr(self._b, name)
