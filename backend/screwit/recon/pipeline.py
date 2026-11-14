"""The reconstruction pipeline (SPEC 6.1; owner: P2).

Implements the ``ReconPipeline`` protocol: frames -> geometry -> scale -> object_mask -> splat + mesh
-> parts, reporting every stage transition and writing artifacts through the ``Storage`` protocol
(SPEC 3.2 layout). The object_mask and parts stages call the injected ``PerceptionService``. The
no-board origin is fixed after the object mask and before the splat and mesh stages write anything
(SPEC 3.1). ``rescale`` is the analytic uniform rescale for ``POST /scale`` (SPEC 5).

No training, no per-scene optimization (SPEC 1.5). torch is optional and only touched by the VRAM
helper when present.
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from screwit.contracts import (
    BBox,
    ReconQuality,
    Reconstruction,
    ScaleInfo,
    ScaleRequest,
    Scan,
)
from screwit.contracts.models import SCAN_TO_WEB_DEFAULT
from screwit.interfaces import (
    PerceptionService,
    ProgressFn,
    StaleRevision,
    Storage,
)

from . import frames as frames_mod
from . import gaussians as gauss_mod
from . import geometry as geom_mod
from . import scale as scale_mod
from . import tsdf as tsdf_mod

log = logging.getLogger(__name__)

# The gaussian head must keep at least this many gaussians inside the object box per object pixel,
# or the splat is rebuilt from depth (see the splat stage).
MIN_HEAD_OBJECT_COVERAGE = 0.05

VRAM_THRESHOLD_BYTES = 2 * 1024 * 1024 * 1024  # assert VRAM is back under this before SAM 2 loads


def release_vram(threshold_bytes: int = VRAM_THRESHOLD_BYTES, *objects) -> int | None:
    """Drop retained GPU tensors, collect, empty the cache, and assert allocation is under the
    threshold before the next model loads (SPEC 6.1 VRAM note). Returns the bytes still allocated,
    or None when torch/CUDA is unavailable."""
    import gc

    for _ in objects:
        pass  # references are dropped by the caller; listed here for intent
    try:
        import torch
    except ImportError:
        gc.collect()
        return None
    gc.collect()
    if not torch.cuda.is_available():  # pragma: no cover - no GPU in dev
        return None
    torch.cuda.empty_cache()  # pragma: no cover
    allocated = int(torch.cuda.memory_allocated())  # pragma: no cover
    if allocated > threshold_bytes:  # pragma: no cover
        raise RuntimeError(f"VRAM not released: {allocated} bytes still allocated")
    return allocated


def box_object_mask(frame_set_depth: list[np.ndarray], centre_xy_model: tuple[float, float],
                    half_extent_px: int = 80) -> list[np.ndarray]:
    """Stub main-object mask used until P3's SAM 2 function lands (SPEC 6.1): a box around the
    board-relative object centre, in model-grid pixels."""
    masks = []
    for d in frame_set_depth:
        h, w = d.shape
        m = np.zeros((h, w), bool)
        cx = int(np.clip(centre_xy_model[0], 0, w - 1))
        cy = int(np.clip(centre_xy_model[1], 0, h - 1))
        x0, x1 = max(0, cx - half_extent_px), min(w, cx + half_extent_px)
        y0, y1 = max(0, cy - half_extent_px), min(h, cy + half_extent_px)
        m[y0:y1, x0:x1] = True
        masks.append(m)
    return masks


@dataclass
class _GeomChoice:
    model: geom_mod.GeometryModel
    name: str


class _ReconGaussianIO:
    """X5's ``GaussianIO`` backed by this package's PLY reader and SPZ writer, so the phone splat is
    packed exactly like the full one."""

    def read_ply(self, path: Path) -> gauss_mod.GaussianCloud:
        return gauss_mod.read_gaussians_ply(Path(path))

    def write_spz(self, cloud, path: Path) -> dict:
        return gauss_mod.write_spz(cloud, Path(path))


class ReconPipelineImpl:
    """Concrete ``ReconPipeline``. The geometry model is injected so tests use the CPU fake."""

    def __init__(self, geometry_model: geom_mod.GeometryModel | None = None,
                 opacity_min: float = 0.1, gaussian_cap: int = 1_500_000,
                 frames_provider=None) -> None:
        self._geometry_model = geometry_model
        self.opacity_min = opacity_min
        self.gaussian_cap = gaussian_cap
        # Optional hook used by the end-to-end CPU test: a callable (scan, storage, frames_dir) ->
        # FramesIndex that renders a synthetic scene instead of decoding a video with ffmpeg.
        self._frames_provider = frames_provider

    # -- helpers ------------------------------------------------------------

    def _geometry(self) -> geom_mod.GeometryModel:
        if self._geometry_model is None:  # pragma: no cover - GPU path
            self._geometry_model = geom_mod.DepthAnything3Adapter()
        return self._geometry_model

    def _geometry_with_cache(self, scan, storage, frames_dir, model_names, timer):
        """Run geometry, reusing a cached GeometryResult when the same video+settings were solved
        before. The DA3 call is the pipeline's slowest stage; caching it makes a re-run of the same
        clip near-instant for everything downstream. Cache key = video bytes + view cap + device +
        model id + infer_gs. Disable with SCREWIT_GEOCACHE=0."""
        from screwit.recon import geocache

        model = self._geometry()
        device = geom_mod._default_device()
        est_s = 8.0 * len(model_names) if device == "cpu" else 0.5 * len(model_names)

        key = None
        if geocache.cache_enabled() and scan.input.type == "video":
            try:
                video_bytes = storage.get_bytes(scan.input.uri)
                key = geocache.video_key(
                    video_bytes,
                    model_id=getattr(model, "model_id", type(model).__name__),
                    view_cap=len(model_names), device=device,
                    infer_gs=bool(getattr(model, "infer_gs", True)),
                )
            except Exception:  # noqa: BLE001 - hashing must never break the scan
                key = None

        if key is not None:
            cached = geocache.load(key)
            if cached is not None and cached.frames == list(model_names):
                timer.tick(0.5, f"reusing cached geometry for {len(model_names)} views")
                return cached

        timer.tick(0.05, f"solving geometry for {len(model_names)} views")
        with timer.creep(0.05, 0.5, est_s, f"solving geometry for {len(model_names)} views"):
            result = model.run(frames_dir / "model", model_names)
        if key is not None:
            geocache.store(key, result)
        return result

    @staticmethod
    def _key(scan_id: str, rel: str) -> str:
        return f"scans/{scan_id}/{rel}"

    # -- run ----------------------------------------------------------------

    def run(self, scan: Scan, storage: Storage, perception: PerceptionService,
            progress: ProgressFn) -> None:
        scan_id = scan.scan_id
        rev = scan.reconstruction_revision
        timer = _StageTimer(progress)
        with tempfile.TemporaryDirectory(prefix="recon_") as td:
            scratch = Path(td)

            # ---- frames ----
            timer.start("frames")
            frames_dir = scratch / "frames"
            index = self._stage_frames(scan, storage, frames_dir, timer)
            frame_names = [e.frame for e in index.frames]
            timer.done("frames", f"{len(frame_names)} keyframes")

            # ---- geometry ----
            timer.start("geometry")
            model_names = geom_mod.cap_views(frame_names)  # CPU fallback caps the view count
            result = self._geometry_with_cache(scan, storage, frames_dir, model_names, timer)
            self._write_geometry_raw(scan_id, storage, result, timer)
            timer.done("geometry")

            # ---- scale ----
            timer.start("scale")
            scale_res = self._stage_scale(storage, frames_dir, index, result)
            # Align the gaussians to the depth frame at runtime (the SPEC scale_factor quirk is not
            # assumed; see cpu-verify/FINDINGS.md). Then apply the ONE scan-frame similarity (from
            # the board) to the aligned gaussians, the cameras and the depth together. Depth is in
            # model units, so the similarity scale converts it to scan-frame millimetres.
            aligned_cloud, gauss_align = geom_mod.align_gaussians_to_depth(result)
            cloud_scan = gauss_mod.apply_similarity(aligned_cloud, scale_res.r, scale_res.t,
                                                    scale_res.s)
            extr_scan = geom_mod.apply_similarity_to_cameras(
                result.extrinsics, scale_res.r, scale_res.t, scale_res.s)
            depth_scan = [d * scale_res.s for d in result.depth]
            timer.done("scale", f"{scale_res.info.method}, {scale_res.info.corner_views} views")

            # ---- object mask ----
            timer.start("object_mask")
            frame_set = self._build_frame_set(scan_id, rev, storage, frames_dir, index,
                                              result, extr_scan, depth_scan)
            with timer.creep(0.1, 0.9, 10.0, "isolating the object"):
                masks = perception.object_mask(frame_set, None, None)
            bbox_min, bbox_max, origin_shift = self._object_bbox(depth_scan, masks, result,
                                                                 extr_scan, scale_res)
            # Center the object: move its bottom-center to the scan-frame origin. This is a rigid
            # translation applied AFTER the (board or learned) scale, so it is scale- and
            # measurement-neutral (every measurement is a difference of points). Done on BOTH paths
            # now -- previously only the no-board path centered, so a with-board scan sat offset by
            # the board-to-object distance and rendered off-center. Centering also makes rescale
            # (which scales about the origin) better-behaved and puts the object on the grid.
            cloud_scan = gauss_mod.translate(cloud_scan, origin_shift)
            extr_scan = extr_scan.copy()
            extr_scan[:, :3, 3] += (extr_scan[:, :3, :3] @ (-origin_shift))
            bbox_min = bbox_min + origin_shift
            bbox_max = bbox_max + origin_shift
            timer.done("object_mask", _mask_note(perception))

            # Release VRAM before SAM 2 (no-op in dev); geometry tensors are dropped here.
            release_vram()

            # ---- splat + mesh ----
            timer.start("splat")

            def _crop(cloud):
                cloud = gauss_mod.crop_to_bbox(cloud, bbox_min, bbox_max, margin_mm=10.0)
                return gauss_mod.prune_opacity(cloud, self.opacity_min)

            def _depth_cloud():
                c = gauss_mod.apply_similarity(geom_mod.gaussians_from_depth(result, frame_set.images),
                                               scale_res.r, scale_res.t, scale_res.s)
                # Match the whole-scene centering shift applied above (both paths now).
                c = gauss_mod.translate(c, origin_shift)
                return _crop(c)

            splat_source = "gaussian head"
            if not gauss_align.trustworthy:
                # The head's gaussians disagree with its own depth; splat the depth instead so the
                # splat matches the mesh (and the origin shift, if any, is applied the same way).
                splat_source = "depth maps (gaussian head misaligned)"
                cloud_scan = _depth_cloud()
            else:
                cloud_scan = _crop(cloud_scan)
                # Pixel-aligned heads give about one gaussian per object pixel. Far fewer inside the
                # object box means the head's geometry sits elsewhere even though the global fit
                # passed (real CPU portrait scan: 1,192 kept out of about 540k object pixels).
                object_pixels = int(sum(int(m.sum()) for m in masks))
                if object_pixels and len(cloud_scan) < MIN_HEAD_OBJECT_COVERAGE * object_pixels:
                    splat_source = (f"depth maps (gaussian head left {len(cloud_scan)} gaussians "
                                    f"on {object_pixels} object pixels)")
                    cloud_scan = _depth_cloud()
            cloud_scan = gauss_mod.voxel_downsample(cloud_scan, self.gaussian_cap)
            splat_manifest = self._write_splat(scan_id, storage, cloud_scan, scratch)
            align_note = (f", gauss-align s={gauss_align.scale:.3f} "
                          f"rmse={gauss_align.relative_rmse:.1%} of depth, "
                          f"{gauss_align.inlier_fraction:.0%} inliers" if gauss_align.applied else "")
            timer.done("splat", f"{len(cloud_scan)} gaussians from {splat_source}{align_note}")

            timer.start("mesh")
            # TSDF fusion is a blocking per-frame loop; creep the bar while it runs so the long
            # mesh stage (can be ~90s on CPU) does not sit frozen.
            with timer.creep(0.05, 0.9, 60.0, "fusing mesh"):
                mesh_tris, mesh_glb_path = self._stage_mesh(scan_id, storage, frame_set, depth_scan,
                                                            masks, extr_scan, result, scratch)
            # Phone-sized variants from the full assets, if the X5 optimizer is installed.
            full_splat_path = splat_manifest.get("ply_path") if len(cloud_scan) > 0 else None
            mesh_mobile_key, splat_mobile_key = self._maybe_mobile_variants(
                scan_id, storage, mesh_glb_path, full_splat_path, scratch)
            timer.done("mesh", f"{mesh_tris} web triangles")

            # ---- reconstruction.json ----
            recon = self._write_reconstruction(
                scan_id, rev, storage, scale_res, frame_names, index,
                result, extr_scan, cloud_scan, bbox_min, bbox_max,
                splat_has=len(cloud_scan) > 0,
                mesh_mobile_uri=mesh_mobile_key, splat_mobile_uri=splat_mobile_key)

            # ---- parts (P3) ----
            timer.start("parts")
            mesh_path = storage.local_path(self._key(scan_id, "mesh/object.glb"), scratch)
            with timer.creep(0.05, 0.9, 20.0, "detecting and naming parts"):
                perception.parts(scan_id, frame_set, masks, mesh_path, rev, storage)
            timer.done("parts")
        _ = recon

    # -- stage frames -------------------------------------------------------

    def _stage_frames(self, scan: Scan, storage: Storage, frames_dir: Path, timer=None):
        frames_dir.mkdir(parents=True, exist_ok=True)
        if self._frames_provider is not None:
            index = self._frames_provider(scan, storage, frames_dir)
        else:
            index = self._extract_frames(scan, storage, frames_dir)
        # Upload full-res, model-grid and frames.json, ticking as each keyframe lands.
        n = len(index.frames)
        for i, entry in enumerate(index.frames):
            storage.put_file(self._key(scan.scan_id, entry.full_res), frames_dir / entry.frame,
                             "image/jpeg")
            storage.put_file(self._key(scan.scan_id, entry.model),
                             frames_dir / "model" / entry.frame, "image/jpeg")
            if timer is not None and n:
                timer.tick((i + 1) / n, f"{i + 1}/{n} keyframes")
        storage.put_file(self._key(scan.scan_id, "frames/frames.json"),
                         frames_dir / "frames.json", "application/json")
        return index

    def _extract_frames(self, scan: Scan, storage: Storage, frames_dir: Path):
        inp = scan.input
        if inp.type == "video":
            scratch = frames_dir.parent
            video_path = storage.local_path(inp.uri, scratch)
            return frames_mod.extract_video_frames(video_path, frames_dir, duration_s=inp.duration_s)
        if inp.type == "photos":
            scratch = frames_dir.parent
            photo_dir = storage.local_path(inp.uri, scratch)
            photos = sorted(Path(photo_dir).glob("*")) if Path(photo_dir).is_dir() else [Path(photo_dir)]
            return frames_mod.extract_photo_frames(photos, frames_dir)
        raise ValueError(f"reconstruction does not run frames for input {inp.type}")  # pragma: no cover

    # -- geometry raw -------------------------------------------------------

    def _write_geometry_raw(self, scan_id: str, storage: Storage, result: geom_mod.GeometryResult,
                            timer=None):
        n = len(result.frames)
        for i, name in enumerate(result.frames):
            stem = Path(name).stem
            buf_d = _npz_bytes(depth=result.depth[i])
            storage.put_bytes(self._key(scan_id, f"geometry/depth/{stem}.npz"), buf_d,
                              "application/octet-stream")
            buf_c = _npz_bytes(conf=result.confidence[i])
            storage.put_bytes(self._key(scan_id, f"geometry/conf/{stem}.npz"), buf_c,
                              "application/octet-stream")
            if timer is not None and n:
                # Inference is done by here; this writes results, so scale the bar over the back
                # half (0.5 -> 1.0) to reflect that the heavy solve already happened.
                timer.tick(0.5 + 0.5 * (i + 1) / n, f"geometry {i + 1}/{n} views")

    # -- stage scale --------------------------------------------------------

    def _stage_scale(self, storage: Storage, frames_dir: Path, index, result):
        from PIL import Image

        by_name = {e.frame: e for e in index.frames}
        per_view: dict[int, dict[int, np.ndarray]] = {}
        grid_maps: dict[int, dict] = {}
        # Iterate the frames the geometry result actually covers (may be a CPU-capped subset),
        # so view indices line up with result.intrinsics / result.extrinsics.
        for vi, name in enumerate(result.frames):
            entry = by_name[name]
            grid_maps[vi] = entry.grid
            img = np.asarray(Image.open(frames_dir / entry.frame).convert("RGB"))
            ids, corners = scale_mod.detect_board_corners(img)
            if ids is None:
                continue
            per_view[vi] = {int(cid): corners[k] for k, cid in enumerate(ids)}

        total_corner_obs = sum(len(v) for v in per_view.values())
        if total_corner_obs >= 8 and len(per_view) >= 2:
            try:
                return scale_mod.scale_from_board(result.intrinsics, result.extrinsics,
                                                  per_view, grid_maps)
            except ValueError:
                pass
        # Learned fallback. DA3-nested reports a metric scale_factor (depth in metres) -> convert
        # metres to mm; a non-metric model has scale_factor None -> apply its learned_scale directly.
        learned = result.learned_scale if result.learned_scale is not None else 1.0
        is_metric = result.scale_factor is not None
        return scale_mod.scale_from_learned(learned, metric=is_metric)

    # -- frame set for perception ------------------------------------------

    def _build_frame_set(self, scan_id, rev, storage, frames_dir, index, result, extr_scan, depth_scan):
        from PIL import Image

        from screwit.interfaces import FrameSet

        by_name = {e.frame: e for e in index.frames}
        entries = [by_name[n] for n in result.frames]
        images = [np.asarray(Image.open(frames_dir / "model" / e.frame).convert("RGB"))
                  for e in entries]
        cameras = geom_mod.to_cameras(scan_id, rev,
                                      _with_extrinsics(result, extr_scan))
        full_paths = [frames_dir / e.frame for e in entries]
        return FrameSet(
            images=images,
            depth_mm=[d.astype(np.float32) for d in depth_scan],
            confidence=[c.astype(np.float32) for c in result.confidence],
            cameras=cameras,
            full_res_paths=full_paths,
        )

    # -- object bbox / origin ----------------------------------------------

    def _object_bbox(self, depth_scan, masks, result, extr_scan, scale_res):
        """Back-project masked depth to a scan-frame point cloud and take its axis-aligned bbox.
        Returns (min, max, origin_shift) where origin_shift moves object_bottom_center to the
        scan-frame origin (used only in the no-board case)."""
        pts = []
        for i, (depth, mask) in enumerate(zip(depth_scan, masks, strict=False)):
            k = result.intrinsics[i]
            w2c = extr_scan[i]
            ys, xs = np.nonzero(mask & (depth > 0))
            if xs.size == 0:
                continue
            z = depth[ys, xs]
            x = (xs - k[0, 2]) / k[0, 0] * z
            y = (ys - k[1, 2]) / k[1, 1] * z
            cam = np.stack([x, y, z], axis=1)
            r = w2c[:3, :3]
            t = w2c[:3, 3]
            world = (cam - t) @ r  # inverse of R x + t (R orthonormal): x = R^T (p - t)
            pts.append(world)
        if not pts:
            return np.array([-50, -50, 0.0]), np.array([50, 50, 100.0]), np.zeros(3)
        allp = np.concatenate(pts, axis=0)
        bmin = allp.min(axis=0)
        bmax = allp.max(axis=0)
        centre = (bmin + bmax) / 2.0
        # object_bottom_center: centre of the bbox projected onto the support plane (z = bmin.z).
        obc = np.array([centre[0], centre[1], bmin[2]])
        origin_shift = -obc  # move obc to origin
        return bmin, bmax, origin_shift

    # -- splat --------------------------------------------------------------

    def _write_splat(self, scan_id, storage, cloud, scratch):
        ply_path = scratch / "scene.ply"
        spz_path = scratch / "scene.spz"
        gauss_mod.write_ply(cloud, ply_path)
        manifest = gauss_mod.write_spz(cloud, spz_path)
        storage.put_file(self._key(scan_id, "splat/scene.ply"), ply_path, "application/octet-stream")
        storage.put_file(self._key(scan_id, "splat/scene.spz"), spz_path, "application/octet-stream")
        manifest["ply_path"] = ply_path
        manifest["spz_path"] = spz_path
        return manifest

    # -- mesh ---------------------------------------------------------------

    def _stage_mesh(self, scan_id, storage, frame_set, depth_scan, masks, extr_scan, result, scratch):
        masked_depth = [np.where(m, d, 0.0).astype(np.float32) for d, m in zip(depth_scan, masks, strict=False)]
        colors = frame_set.images
        mesh = tsdf_mod.fuse_tsdf(masked_depth, colors, result.intrinsics, extr_scan)
        mesh = tsdf_mod.repair_mesh(mesh)
        # Sparse CPU scans sometimes fragment and the largest-component rule drops the object,
        # yielding an empty mesh. Retry once, looser: wider truncation to bridge gaps and keep ALL
        # sizeable components (not just the largest), so the object survives.
        if mesh.triangles.shape[0] < 100:
            mesh = tsdf_mod.fuse_tsdf(masked_depth, colors, result.intrinsics, extr_scan,
                                      keep_largest=False, sdf_trunc=0.012)
            mesh = tsdf_mod.repair_mesh(mesh)
        full_path = scratch / "object_full.ply"
        glb_path = scratch / "object.glb"
        tsdf_mod.write_full_ply(mesh, full_path)
        tris = tsdf_mod.write_web_glb(mesh, glb_path)
        # Never overwrite an existing good mesh with an empty one.
        if tris < 1:
            existing = storage.head(self._key(scan_id, "mesh/object.glb"))
            if existing is not None:
                raise ValueError("mesh fusion produced no triangles; keeping the previous mesh")
        storage.put_file(self._key(scan_id, "mesh/object_full.ply"), full_path,
                         "application/octet-stream")
        storage.put_file(self._key(scan_id, "mesh/object.glb"), glb_path, "model/gltf-binary")
        return tris, glb_path

    # -- mobile variants (X5 optimizer; optional) --------------------------

    def _maybe_mobile_variants(self, scan_id, storage, full_mesh_path, full_splat_path, scratch):
        """Produce phone-sized variants (AGENTS.md Performance budgets) with the X5 optimizer, upload
        them under scans/<id>/mobile/, and return their storage keys.

        A failure here (for example gltfpack not installed) does not fail the scan: the full assets
        are already written and the viewer falls back to them, so this logs and returns None keys.
        """
        from screwit import optimize

        out_dir = scratch / "mobile"
        out_dir.mkdir(parents=True, exist_ok=True)
        splat_arg = str(full_splat_path) if full_splat_path is not None else None
        try:
            variants = optimize.make_mobile_variants(str(full_mesh_path), splat_arg, str(out_dir),
                                                     gaussian_io=_ReconGaussianIO())
        except Exception:  # noqa: BLE001 - optional optimisation, never fatal
            log.warning("mobile variants failed for %s; serving full assets only", scan_id,
                        exc_info=True)
            return None, None
        mesh_key = splat_key = None
        if variants.mesh_mobile_path is not None and Path(variants.mesh_mobile_path).exists():
            mesh_key = self._key(scan_id, "mobile/object.glb")
            storage.put_file(mesh_key, Path(variants.mesh_mobile_path), "model/gltf-binary")
        if variants.splat_mobile_path is not None and Path(variants.splat_mobile_path).exists():
            splat_key = self._key(scan_id, "mobile/scene.spz")
            storage.put_file(splat_key, Path(variants.splat_mobile_path), "application/octet-stream")
        return mesh_key, splat_key

    # -- reconstruction.json -----------------------------------------------

    def _write_reconstruction(self, scan_id, rev, storage, scale_res, frame_names, index,
                              result, extr_scan, cloud, bbox_min, bbox_max, splat_has,
                              mesh_mobile_uri=None, splat_mobile_uri=None) -> Reconstruction:
        keyframes = [self._key(scan_id, f"frames/{n}") for n in frame_names[:2]]
        mean_conf = float(np.mean([float(c.mean()) for c in result.confidence])) if result.confidence else 0.0
        recon = Reconstruction(
            scan_id=scan_id,
            reconstruction_revision=rev,
            splat_uri=self._key(scan_id, "splat/scene.spz") if splat_has else None,
            mesh_uri=self._key(scan_id, "mesh/object.glb"),
            cameras_uri=self._key(scan_id, "geometry/cameras.json"),
            keyframes=keyframes,
            scale=scale_res.info,
            frame_origin=scale_res.frame_origin,
            scan_to_web=SCAN_TO_WEB_DEFAULT,
            object_bbox_mm=BBox(min=tuple(float(v) for v in bbox_min),
                                max=tuple(float(v) for v in bbox_max)),
            quality=ReconQuality(frames_used=index.kept, total_frames=index.total_sampled or index.kept,
                                 mean_depth_confidence=round(mean_conf, 4),
                                 gaussians=len(cloud) if splat_has else None),
            splat_mobile_uri=splat_mobile_uri,
            mesh_mobile_uri=mesh_mobile_uri,
        )
        # cameras.json (scan-frame extrinsics)
        cameras = geom_mod.to_cameras(scan_id, rev, _with_extrinsics(result, extr_scan))
        storage.put_bytes(self._key(scan_id, "geometry/cameras.json"),
                          cameras.model_dump_json(indent=2).encode(), "application/json")
        storage.put_bytes(self._key(scan_id, "reconstruction.json"),
                          recon.model_dump_json(indent=2).encode(), "application/json")
        return recon

    # -- rescale (POST /scale) ---------------------------------------------

    def rescale(self, scan_id: str, req: ScaleRequest, storage: Storage,
                perception: PerceptionService) -> Reconstruction:
        """Analytic uniform rescale (SPEC 5). Multiplies every scan-frame length by ``s`` about the
        origin and bumps ``reconstruction_revision``; raises StaleRevision on a mismatch."""
        raw = storage.get_bytes(self._key(scan_id, "reconstruction.json"))
        current = Reconstruction.model_validate_json(raw)
        if current.reconstruction_revision != req.expected_revision:
            raise StaleRevision(
                f"expected_revision {req.expected_revision} != current "
                f"{current.reconstruction_revision}")

        s = scale_mod.rescale_factor_from_two_points(req.point_a_mm, req.point_b_mm,
                                                     req.known_length_mm)
        new_rev = current.reconstruction_revision + 1

        self._rescale_cameras(scan_id, storage, s, new_rev)
        self._rescale_splat(scan_id, storage, s)
        self._rescale_mesh(scan_id, storage, s)

        new_bbox = BBox(min=tuple(float(v) * s for v in current.object_bbox_mm.min),
                        max=tuple(float(v) * s for v in current.object_bbox_mm.max))
        new_scale = ScaleInfo(method="manual", board=current.scale.board,
                              corner_views=current.scale.corner_views,
                              residual_mm=current.scale.residual_mm,
                              board_tilt_deg=current.scale.board_tilt_deg,
                              learned_scale_ratio=current.scale.learned_scale_ratio)
        recon = current.model_copy(update={
            "reconstruction_revision": new_rev,
            "scale": new_scale,
            "object_bbox_mm": new_bbox,
        })
        storage.put_bytes(self._key(scan_id, "reconstruction.json"),
                          recon.model_dump_json(indent=2).encode(), "application/json")
        # Parts rescale is P3's analytic step.
        perception.rescale_parts(scan_id, s, new_rev, storage)
        return recon

    def _rescale_cameras(self, scan_id, storage, s, new_rev):
        from screwit.contracts import Cameras

        raw = storage.get_bytes(self._key(scan_id, "geometry/cameras.json"))
        cams = Cameras.model_validate_json(raw)
        new_frames = []
        for cf in cams.frames:
            m = np.array(cf.world_to_camera, dtype=float)
            m[:3, 3] *= s   # scale translation; a point X -> sX, so t -> s t keeps R X + t consistent
            new_frames.append(cf.model_copy(update={
                "world_to_camera": tuple(tuple(float(v) for v in row) for row in m)}))
        new = cams.model_copy(update={"reconstruction_revision": new_rev, "frames": new_frames})
        storage.put_bytes(self._key(scan_id, "geometry/cameras.json"),
                          new.model_dump_json(indent=2).encode(), "application/json")

    def _rescale_splat(self, scan_id, storage, s):
        import tempfile as _t

        with _t.TemporaryDirectory() as td:
            scratch = Path(td)
            ply = storage.local_path(self._key(scan_id, "splat/scene.ply"), scratch)
            cloud = gauss_mod.read_ply(ply)
            scaled = gauss_mod.apply_similarity(cloud, np.eye(3), np.zeros(3), s)
            gauss_mod.write_ply(scaled, scratch / "scene.ply")
            gauss_mod.write_spz(scaled, scratch / "scene.spz")
            storage.put_file(self._key(scan_id, "splat/scene.ply"), scratch / "scene.ply",
                             "application/octet-stream")
            storage.put_file(self._key(scan_id, "splat/scene.spz"), scratch / "scene.spz",
                             "application/octet-stream")

    def _rescale_mesh(self, scan_id, storage, s):
        import tempfile as _t

        import trimesh

        with _t.TemporaryDirectory() as td:
            scratch = Path(td)
            glb = storage.local_path(self._key(scan_id, "mesh/object.glb"), scratch)
            scene = trimesh.load(glb, force="mesh")
            scene.apply_scale(s)
            out = scratch / "object.glb"
            scene.export(out)
            storage.put_file(self._key(scan_id, "mesh/object.glb"), out, "model/gltf-binary")


def _mask_note(perception) -> str:
    """One line on how the object mask was made, if the perception service reports it."""
    info = getattr(perception, "last_mask_info", None)
    if info is None:
        return ""
    note = f"{info.method}, {info.mask_fraction:.0%} of pixels"
    if info.notes:
        note += f" ({'; '.join(info.notes)})"
    return note


def _npz_bytes(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


class _StageTimer:
    """Wraps the ProgressFn to emit a 'running' event at stage start and a 'done' event whose
    message carries the stage's wall time, so the server pipeline can be timed end to end.
    ``elapsed`` exposes the per-stage seconds for callers that want the raw numbers."""

    def __init__(self, progress: ProgressFn) -> None:
        self._progress = progress
        self._t0: float | None = None
        self._stage: str | None = None
        self.elapsed: dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._stage = stage
        self._t0 = time.monotonic()
        self._progress(stage, "running", 0.0, None)

    def tick(self, fraction: float, message: str | None = None) -> None:
        """Report intra-stage progress for the stage currently running, so a long stage (frames,
        geometry) advances the UI instead of sitting at 0% until it finishes. Clamped to (0, 1);
        the final 1.0/done is still emitted by ``done``."""
        if self._stage is None:
            return
        frac = min(max(float(fraction), 0.0), 0.999)
        self._progress(self._stage, "running", frac, message)

    @contextmanager
    def creep(self, start: float, ceiling: float, expected_s: float, message: str | None = None):
        """Advance the current stage's fraction from ``start`` toward ``ceiling`` on a timer while an
        opaque, un-instrumentable blocking call runs (e.g. DA3 inference), so the UI shows motion
        instead of freezing. An easing curve approaches but never reaches the ceiling, so the bar
        keeps moving even if the call runs longer than expected; real ticks after the call take over
        the remainder. The ticker is a daemon thread stopped on exit."""
        import math
        import threading

        stop = threading.Event()

        def _run() -> None:
            t0 = time.monotonic()
            while not stop.wait(1.0):
                elapsed = time.monotonic() - t0
                # 1 - exp(-t/tau): ~63% of (ceiling-start) at expected_s, asymptotes to ceiling.
                frac = start + (ceiling - start) * (1.0 - math.exp(-elapsed / max(expected_s, 1.0)))
                self.tick(frac, message)

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()

    def done(self, stage: str, message: str | None = None) -> None:
        dt = time.monotonic() - self._t0 if self._t0 is not None else 0.0
        self.elapsed[stage] = dt
        secs = f"{dt:.2f}s"
        full = f"{message} ({secs})" if message else secs
        self._progress(stage, "done", 1.0, full)


def _with_extrinsics(result: geom_mod.GeometryResult, extr: np.ndarray) -> geom_mod.GeometryResult:
    """Return a shallow copy of the result carrying scan-frame extrinsics (for cameras.json)."""
    return geom_mod.GeometryResult(
        frames=result.frames, width=result.width, height=result.height,
        intrinsics=result.intrinsics, extrinsics=extr,
        depth=result.depth, confidence=result.confidence, gaussians=result.gaussians,
        scale_factor=result.scale_factor, learned_scale=result.learned_scale,
    )
