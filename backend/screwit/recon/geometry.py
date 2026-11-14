"""Feed-forward geometry (SPEC 6.1 step 2).

A ``GeometryModel`` protocol returns, for a set of model-grid frames, per-frame world-to-camera
extrinsics and intrinsics (OpenCV axes), per-frame depth and confidence, and 3D gaussians, all in
ONE arbitrary model frame (the scale stage later maps that frame to the scan frame). Three
implementations:

- ``DepthAnything3Adapter``  -- DA3NESTED-GIANT-LARGE-1.1, infer_gs=True (lazy torch, @pytest.mark.gpu)
- ``MapAnythingAdapter``     -- facebook/map-anything-apache, inverts camera-to-world (lazy torch)
- ``FakeGeometryModel``      -- CPU synthetic scene used by the end-to-end tests; it reproduces the
  DA3-nested quirk (depth and extrinsic translations multiplied by scale_factor, gaussians not) so
  the gaussian correction in gaussians.apply_scale_factor is exercised.

No training and no per-scene optimization anywhere (SPEC 1.5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from screwit.contracts import CameraFrame, Cameras

from .gaussians import GaussianCloud


@dataclass
class GeometryResult:
    """Raw feed-forward output, in the model frame (not yet the scan frame).

    ``extrinsics`` are world-to-camera 4x4 (OpenCV axes). ``depth`` is in the model's own units
    (metric metres for the DA3-nested model, arbitrary otherwise); it is NOT millimetres -- the
    scale stage's similarity converts everything to scan-frame mm. ``gaussians`` live in their own
    gaussian frame, which (per the CPU findings) is NOT related to the depth frame by a hard-coded
    scale_factor; ``align_gaussians_to_depth`` estimates that similarity at runtime. ``scale_factor``
    is the DA3-nested metric factor (``None`` for non-nested models) and ``learned_scale`` the
    model's own metric estimate (DA3 nested head / MapAnything), recorded for the scale stage's
    sanity check and learned fallback.
    """

    frames: list[str]                 # frame names, e.g. "0001.jpg"
    width: int
    height: int
    intrinsics: np.ndarray            # (N, 3, 3)
    extrinsics: np.ndarray            # (N, 4, 4) world->camera, OpenCV
    depth: list[np.ndarray]           # each (H, W) float32, model units, 0 = invalid
    confidence: list[np.ndarray]      # each (H, W) float32 in [0, 1]
    gaussians: GaussianCloud          # in the gaussian frame (aligned to depth at runtime)
    scale_factor: float | None = None
    learned_scale: float | None = None


class GeometryModel(Protocol):
    def run(self, frames_dir: Path, frame_names: list[str]) -> GeometryResult:
        """Run one feed-forward pass over the model-grid frames named in ``frame_names``
        (read from ``frames_dir``). Returns everything in one model frame."""


# --------------------------------------------------------------------------- Cameras conversion

def to_cameras(scan_id: str, reconstruction_revision: int, result: GeometryResult) -> Cameras:
    """Build the ``Cameras`` contract (cameras.json) from a scan-frame GeometryResult.

    ``world_to_camera`` is row-major OpenCV, in the scan frame and millimetres. The caller must have
    already applied the scan-frame similarity to ``extrinsics`` (translation in mm).
    """
    frames: list[CameraFrame] = []
    for i, name in enumerate(result.frames):
        w2c = result.extrinsics[i]
        k = result.intrinsics[i]
        frames.append(CameraFrame(
            frame=name,
            width=result.width,
            height=result.height,
            world_to_camera=_matrix4(w2c),
            K=_matrix3(k),
        ))
    return Cameras(scan_id=scan_id, reconstruction_revision=reconstruction_revision, frames=frames)


def _matrix4(m: np.ndarray) -> tuple:
    a = np.asarray(m, dtype=float).reshape(4, 4)
    return tuple(tuple(float(v) for v in row) for row in a)


def _matrix3(m: np.ndarray) -> tuple:
    a = np.asarray(m, dtype=float).reshape(3, 3)
    return tuple(tuple(float(v) for v in row) for row in a)


def apply_similarity_to_cameras(extrinsics: np.ndarray, r: np.ndarray, t: np.ndarray,
                                s: float) -> np.ndarray:
    """Map world-to-camera extrinsics from the model frame to the scan frame under the similarity
    ``X = s R x + t`` (model point x -> scan point X).

    The camera orientation rotates by ``R`` and its centre moves by the full similarity, keeping the
    world-to-camera rotation ORTHONORMAL (OpenCV convention); the metric scale ``s`` shows up as the
    scaled camera centre, i.e. in the translation, not the rotation:
        R_wc' = R_wc R^T,  C' = s R C + t,  t_wc' = -R_wc' C'.
    """
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    out = np.zeros_like(extrinsics, dtype=np.float64)
    rt = r.T
    for i in range(extrinsics.shape[0]):
        r_wc = extrinsics[i, :3, :3]
        t_wc = extrinsics[i, :3, 3]
        c = -r_wc.T @ t_wc                 # camera centre in the model frame
        r_new = r_wc @ rt                  # orthonormal
        c_new = s * (r @ c) + t            # camera centre in the scan frame
        t_new = -r_new @ c_new
        out[i, :3, :3] = r_new
        out[i, :3, 3] = t_new
        out[i, 3, 3] = 1.0
    return out


# --------------------------------------------------------------------------- gaussian alignment

@dataclass
class GaussianAlignment:
    """Result of aligning the gaussian frame to the depth frame at runtime."""

    scale: float
    rmse_mm: float
    inlier_fraction: float
    applied: bool
    relative_rmse: float = 0.0

    @property
    def trustworthy(self) -> bool:
        return (self.applied and self.relative_rmse <= GAUSSIAN_MAX_RELATIVE_RMSE
                and self.inlier_fraction >= GAUSSIAN_MIN_INLIERS)


def align_gaussians_to_depth(result: GeometryResult, *, conf_percentile: float = 70.0,
                             max_points_per_view: int = 4000, min_points: int = 50
                             ) -> tuple[GaussianCloud, GaussianAlignment]:
    """Estimate and apply the gaussian-frame -> depth-frame similarity at runtime (CPU findings 7).

    SPEC 3.1's hard-coded "multiply gaussian means by scale_factor" was NOT confirmed on real
    images, so instead of trusting it we estimate the transform from data. DA3's gaussians are
    pixel-aligned: the gaussian tensor is ``N`` slabs of ``H*W`` and slab ``k`` corresponds to view
    ``k``. For each view we back-project its confident depth pixels into the depth world frame and
    match them (by pixel index) to that slab's gaussian means, then fit one similarity (RANSAC +
    Umeyama) over all views. If the gaussian count does not match ``N*H*W`` (e.g. the pruned PLY),
    alignment is skipped and the gaussians are returned unchanged with ``applied=False``.

    Returns the aligned cloud and a GaussianAlignment (scale, RMSE, inlier fraction) to record in
    reconstruction quality. The RMSE is in the depth frame's own units; callers scale it later.
    """
    from .scale import ransac_similarity

    cloud = result.gaussians
    n_views = len(result.frames)
    h, w = result.depth[0].shape
    expected = n_views * h * w
    if len(cloud) != expected or len(cloud) == 0:
        return cloud, GaussianAlignment(scale=1.0, rmse_mm=0.0, inlier_fraction=0.0, applied=False)

    src_pts = []  # gaussian-frame means
    dst_pts = []  # depth-frame back-projections
    rng = np.random.default_rng(0)
    for k in range(n_views):
        depth = result.depth[k]
        conf = result.confidence[k]
        k_mat = result.intrinsics[k]
        w2c = result.extrinsics[k]
        valid = depth > 0
        if conf is not None and conf.shape == depth.shape:
            thr = np.percentile(conf[valid], conf_percentile) if valid.any() else 0.0
            valid &= conf >= thr
        ys, xs = np.nonzero(valid)
        if xs.size == 0:
            continue
        if xs.size > max_points_per_view:
            sel = rng.choice(xs.size, max_points_per_view, replace=False)
            ys, xs = ys[sel], xs[sel]
        z = depth[ys, xs].astype(np.float64)
        x = (xs - k_mat[0, 2]) / k_mat[0, 0] * z
        y = (ys - k_mat[1, 2]) / k_mat[1, 1] * z
        cam = np.stack([x, y, z], axis=1)
        r = w2c[:3, :3]
        t = w2c[:3, 3]
        world = (cam - t) @ r  # x = R^T (p_cam - t)
        slab = cloud.means[k * h * w:(k + 1) * h * w].astype(np.float64)
        slab_sel = slab[ys * w + xs]
        src_pts.append(slab_sel)
        dst_pts.append(world)

    if not src_pts:
        return cloud, GaussianAlignment(scale=1.0, rmse_mm=0.0, inlier_fraction=0.0, applied=False)
    src = np.concatenate(src_pts, axis=0)
    dst = np.concatenate(dst_pts, axis=0)
    if src.shape[0] < min_points:
        return cloud, GaussianAlignment(scale=1.0, rmse_mm=0.0, inlier_fraction=0.0, applied=False)

    # RANSAC threshold in the depth frame's own units (metres for the nested model, arbitrary for
    # others): 2% of the median depth, so far-flung gaussians cannot all count as inliers.
    med_depth = float(np.median(np.concatenate([d[d > 0] for d in result.depth])))
    thr = 0.02 * med_depth
    r_fit, t_fit, s_fit, inliers, rmse = ransac_similarity(src, dst, iters=200, threshold_mm=thr)
    from .gaussians import apply_similarity

    aligned = apply_similarity(cloud, r_fit, t_fit, s_fit)
    return aligned, GaussianAlignment(scale=float(s_fit), rmse_mm=float(rmse),
                                      inlier_fraction=float(inliers.mean()), applied=True,
                                      relative_rmse=float(rmse) / med_depth if med_depth > 0 else 1.0)


# Above this RMSE (as a fraction of the median depth) or below this inlier share, the gaussian
# head's geometry disagrees with its own depth too much to trust; the pipeline then builds the splat
# from the depth maps instead (measured on the X2 phone clip on CPU: per-view gaussian/depth ratios
# from 0.8 to 2.0, so no single similarity fits).
GAUSSIAN_MAX_RELATIVE_RMSE = 0.03
GAUSSIAN_MIN_INLIERS = 0.5


def gaussians_from_depth(result: GeometryResult, images: list[np.ndarray], *, stride: int = 2,
                         conf_percentile: float = 30.0) -> GaussianCloud:
    """Pixel-aligned gaussians from the depth maps, in the depth (model) frame.

    One isotropic gaussian per ``stride``-th confident pixel, sized to its pixel footprint, coloured
    from the model-grid image. Geometry is exactly the depth the mesh is fused from, so the splat
    and the mesh agree.
    """
    from .gaussians import SH_C0

    means, scales, cols = [], [], []
    for k in range(len(result.frames)):
        depth = result.depth[k]
        conf = result.confidence[k]
        kk = result.intrinsics[k]
        w2c = result.extrinsics[k]
        img = images[k]
        ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[ys, xs]
        ok = z > 0
        if conf is not None and conf.shape == depth.shape and (conf > 0).any():
            ok &= conf[ys, xs] >= np.percentile(conf[depth > 0], conf_percentile)
        ys, xs, z = ys[ok], xs[ok], z[ok].astype(np.float64)
        x = (xs - kk[0, 2]) / kk[0, 0] * z
        y = (ys - kk[1, 2]) / kk[1, 1] * z
        cam = np.stack([x, y, z], axis=1)
        means.append((cam - w2c[:3, 3]) @ w2c[:3, :3])
        scales.append(z * stride / kk[0, 0] * 0.6)
        if img is not None and img.shape[:2] == depth.shape:
            cols.append(img[ys, xs, :3].astype(np.float32) / 255.0)
        else:
            cols.append(np.full((len(z), 3), 0.6, np.float32))
    m = np.concatenate(means).astype(np.float32)
    s = np.concatenate(scales)
    rgb = np.concatenate(cols)
    n = len(m)
    return GaussianCloud(
        means=m,
        log_scales=np.repeat(np.log(np.clip(s, 1e-9, None))[:, None], 3, axis=1).astype(np.float32),
        quats_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 3.0, np.float32),
        f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
    )




# --------------------------------------------------------------------------- GPU adapters

# CPU fallback profile (CPU findings 8): on CPU, cap views and hand every model call the device.
CPU_VIEW_CAP = 16


def _cpu_view_cap() -> int:
    """Max views fed to the model on CPU. ``SCREWIT_CPU_VIEW_CAP`` overrides the default of 16; fewer
    views is faster (DA3 time scales with view count) at some cost to coverage. Clamped to >= 2."""
    import os

    raw = os.environ.get("SCREWIT_CPU_VIEW_CAP")
    if not raw:
        return CPU_VIEW_CAP
    try:
        return max(2, int(raw))
    except ValueError:
        return CPU_VIEW_CAP


def _default_device() -> str:
    """Device for the model adapters: ``SCREWIT_DEVICE`` env, else cuda if available, else cpu."""
    import os

    env = os.environ.get("SCREWIT_DEVICE")
    if env:
        return env
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cap_views(frame_names: list[str], device: str | None = None) -> list[str]:
    """Reduce the view count on CPU (findings 8): evenly subsample to the CPU view cap."""
    dev = device or _default_device()
    cap = _cpu_view_cap()
    if dev != "cpu" or len(frame_names) <= cap:
        return list(frame_names)
    idx = np.linspace(0, len(frame_names) - 1, cap).round().astype(int)
    seen: list[str] = []
    for i in idx:
        if frame_names[i] not in seen:
            seen.append(frame_names[i])
    return seen


def _pad_extrinsics(extr: np.ndarray) -> np.ndarray:
    """Pad (N, 3, 4) world-to-camera matrices to (N, 4, 4) with a [0, 0, 0, 1] row (findings 3)."""
    extr = np.asarray(extr, dtype=np.float64)
    if extr.shape[1:] == (4, 4):
        return extr
    n = extr.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = extr
    return out


def _gaussians_from_prediction(prediction):  # pragma: no cover - needs weights
    """Build a GaussianCloud from ``prediction.gaussians`` in memory (findings 8, 10, 11): means,
    LINEAR scales (converted to log), PROBABILITY opacity (converted to an inverse-sigmoid logit),
    rotations already [w, x, y, z], DC-only SH colour."""
    from .gaussians import GaussianCloud

    g = getattr(prediction, "gaussians", None)
    if g is None:
        raise AttributeError("prediction has no gaussians (DA3-Large has no Gaussian head)")

    def arr(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    means = arr(g.means).astype(np.float32).reshape(-1, 3)
    scales = arr(g.scales).astype(np.float64).reshape(-1, 3)
    log_scales = np.log(np.clip(scales, 1e-12, None)).astype(np.float32)
    quats = arr(g.rotations).astype(np.float32).reshape(-1, 4)  # [w, x, y, z]
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    prob = np.clip(arr(g.opacities).astype(np.float64).reshape(-1), 1e-6, 1 - 1e-6)
    opac = np.log(prob / (1.0 - prob)).astype(np.float32)       # probability -> inverse-sigmoid
    sh = arr(g.harmonics) if hasattr(g, "harmonics") else arr(getattr(g, "sh", None))
    sh = np.asarray(sh, dtype=np.float32).reshape(means.shape[0], 3, -1)
    # Layout is (N, 3 channels, K coefficients) -- measured on the real model: harmonics
    # (1, N, 3, 9) for SH degree 2. The DC term is coefficient 0 of each channel, NOT the first
    # three values of the flattened row (those are R's first three coefficients).
    f_dc = sh[:, :, 0]
    return GaussianCloud(means=means, log_scales=log_scales, quats_wxyz=quats,
                         opacities=opac, f_dc=f_dc.astype(np.float32))


def _empty_cloud():
    """A GaussianCloud with no points (head skipped); the pipeline then splats from depth."""
    from .gaussians import GaussianCloud

    return GaussianCloud(
        means=np.zeros((0, 3), np.float32), log_scales=np.zeros((0, 3), np.float32),
        quats_wxyz=np.zeros((0, 4), np.float32), opacities=np.zeros((0,), np.float32),
        f_dc=np.zeros((0, 3), np.float32),
    )


class DepthAnything3Adapter:
    """DA3NESTED-GIANT-LARGE-1.1 with a Gaussian head (SPEC 6.1 step 2). Lazy-imports torch; real
    runs carry @pytest.mark.gpu. API verified against the CPU findings (cpu-verify/FINDINGS.md)."""

    model_id = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"  # Hugging Face repo id (org/name)

    def __init__(self, gs_views_interval: int = 4, ref_view_strategy: str = "middle",
                 device: str | None = None, infer_gs: bool | None = None) -> None:
        self.gs_views_interval = gs_views_interval
        self.ref_view_strategy = ref_view_strategy
        self.device = device or _default_device()
        # The Gaussian head roughly doubles per-view CPU time (cpu-verify findings). It only feeds the
        # SPLAT, never the mesh or any measurement, and the pipeline already rebuilds the splat from
        # depth when the head is absent/untrustworthy. So it can be skipped for speed with no loss of
        # measurement accuracy. SCREWIT_INFER_GS=0 turns it off; default is on (unchanged behaviour).
        if infer_gs is None:
            import os

            infer_gs = os.environ.get("SCREWIT_INFER_GS", "1").strip().lower() not in ("0", "false", "off", "no")
        self.infer_gs = infer_gs
        self._model = None

    def _load(self):  # pragma: no cover - needs weights
        if self._model is None:
            from depth_anything_3.api import DepthAnything3  # type: ignore

            self._model = DepthAnything3.from_pretrained(self.model_id).to(self.device)
        return self._model

    def run(self, frames_dir: Path, frame_names: list[str]) -> GeometryResult:  # pragma: no cover - weights
        model = self._load()
        image_paths = [str(frames_dir / n) for n in frame_names]
        # No export_dir: every export format that carries gaussians also renders a gs_video, which
        # crashes on CPU (cpu-verify/FINDINGS.md). The gaussians are read from the in-memory
        # prediction instead (linear scales, probability opacity, wxyz), as the findings recommend.
        prediction = model.inference(image_paths, infer_gs=self.infer_gs,
                                     ref_view_strategy=self.ref_view_strategy)
        # With the head off (or if it returned nothing), carry an empty cloud; the pipeline splats
        # from depth instead, so the splat still matches the mesh.
        try:
            cloud = _gaussians_from_prediction(prediction) if self.infer_gs else _empty_cloud()
        except AttributeError:
            cloud = _empty_cloud()

        depth = [np.asarray(d, dtype=np.float32) for d in prediction.depth]
        conf = [np.asarray(c, dtype=np.float32) for c in prediction.conf]
        extr = _pad_extrinsics(np.asarray(prediction.extrinsics, dtype=np.float64))  # (N,3,4)->4x4
        intr = np.asarray(prediction.intrinsics, dtype=np.float64)
        h, w = depth[0].shape
        sf = getattr(prediction, "scale_factor", None)
        scale_factor = float(sf) if sf is not None else None
        learned = scale_factor  # the nested metric factor is the model's own metric estimate
        return GeometryResult(
            frames=list(frame_names), width=w, height=h, intrinsics=intr, extrinsics=extr,
            depth=depth, confidence=conf, gaussians=cloud,
            scale_factor=scale_factor, learned_scale=learned,
        )


class MapAnythingAdapter:
    """facebook/map-anything-apache (SPEC 6.1 step 2). Returns camera-to-world; this adapter inverts
    to world-to-camera. No gaussian head, so GeometryResult carries an empty cloud. Lazy torch.
    API verified against the CPU findings (load_images; infer returns a list of per-view dicts)."""

    model_id = "facebook/map-anything-apache"

    def __init__(self, device: str | None = None) -> None:
        self.device = device or _default_device()
        self._model = None

    def _load(self):  # pragma: no cover - needs weights
        if self._model is None:
            from mapanything.models import MapAnything  # type: ignore

            self._model = MapAnything.from_pretrained(self.model_id).to(self.device)
        return self._model

    def run(self, frames_dir: Path, frame_names: list[str]) -> GeometryResult:  # pragma: no cover - weights
        from mapanything.utils.image import load_images  # type: ignore

        model = self._load()
        paths = [str(frames_dir / n) for n in frame_names]
        views = load_images(paths)
        # README default is bf16 AMP; the CPU findings require use_amp=False on CPU.
        use_amp = self.device != "cpu"
        preds = model.infer(views, memory_efficient_inference=True, use_amp=use_amp)
        # preds is a LIST of per-view dicts with a leading batch dim 1.
        c2w = np.stack([np.asarray(v["camera_poses"], dtype=np.float64)[0] for v in preds])  # (N,4,4)
        w2c = np.linalg.inv(c2w)
        intr = np.stack([np.asarray(v["intrinsics"], dtype=np.float64)[0] for v in preds])
        depth = [np.asarray(v["depth_z"], dtype=np.float32)[0, :, :, 0] for v in preds]
        conf = [np.asarray(v["conf"], dtype=np.float32)[0] for v in preds]
        h, w = depth[0].shape
        learned = float(np.asarray(preds[0]["metric_scaling_factor"]).reshape(-1)[0])
        empty = GaussianCloud(
            means=np.zeros((0, 3), np.float32), log_scales=np.zeros((0, 3), np.float32),
            quats_wxyz=np.zeros((0, 4), np.float32), opacities=np.zeros((0,), np.float32),
            f_dc=np.zeros((0, 3), np.float32),
        )
        return GeometryResult(
            frames=list(frame_names), width=w, height=h, intrinsics=intr, extrinsics=w2c,
            depth=depth, confidence=conf, gaussians=empty,
            scale_factor=None, learned_scale=learned,
        )


# --------------------------------------------------------------------------- fake CPU model

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera extrinsic (X right, Y down, Z forward) looking from eye to target."""
    eye = np.asarray(eye, float)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)       # +Z (forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)              # +X
    down = np.cross(forward, right)                    # +Y (down), right-handed
    r_cw = np.stack([right, down, forward], axis=0)    # rows map world->camera
    t = -r_cw @ eye
    m = np.eye(4)
    m[:3, :3] = r_cw
    m[:3, 3] = t
    return m


class FakeGeometryModel:
    """CPU synthetic scene (SPEC 6.1; brief). Builds a small scene -- a box object standing on a
    support plane, with a ChArUco-textured board lying flat next to it -- renders model-grid colour
    and dense depth from a ring of cameras, and returns everything in an unknown model frame.

    Per the CPU findings, the SPEC's hard-coded nested scale_factor quirk is NOT assumed. Instead
    the gaussians are emitted as PIXEL-ALIGNED slabs (one per depth pixel per view, slab k = view k)
    living in a gaussian frame that may differ from the depth frame by ``gaussian_quirk`` (a uniform
    scale). With ``gaussian_quirk != 1`` the quirk is PRESENT and ``align_gaussians_to_depth`` must
    recover that scale; with ``gaussian_quirk == 1`` it is ABSENT and alignment should return ~1.
    Both cases are exercised by the tests.
    """

    def __init__(self, n_views: int = 12, grid_wh: tuple[int, int] = (504, 280),
                 model_rotation_deg: float = 25.0,
                 model_translation=(50.0, -30.0, 20.0), model_scale: float = 0.5,
                 gaussian_quirk: float = 1.0,
                 board_squares=(5, 7), square_mm: float = 30.0,
                 object_size_mm=(80.0, 80.0, 100.0)) -> None:
        self.n_views = n_views
        self.grid_w, self.grid_h = grid_wh
        self.gaussian_quirk = gaussian_quirk
        self.board_squares = board_squares
        self.square_mm = square_mm
        self.object_size_mm = np.asarray(object_size_mm, float)
        # The unknown model-frame similarity (scan -> model). The scale stage recovers its inverse
        # from the board.
        self._sim_r = _rot_z(np.deg2rad(model_rotation_deg))
        self._sim_t = np.asarray(model_translation, float)
        self._sim_s = model_scale
        self._board_texture: np.ndarray | None = None

    # -- scene geometry, all in the SCAN frame (mm, Z up, board origin at 0) --

    def _board_corners_scan(self) -> np.ndarray:
        """Inner ChArUco corners of a squares_x by squares_y board lying flat on Z=0."""
        sx, sy = self.board_squares
        pts = []
        for j in range(1, sy):
            for i in range(1, sx):
                pts.append([i * self.square_mm, j * self.square_mm, 0.0])
        return np.asarray(pts, float)

    def _object_gaussians_scan(self) -> GaussianCloud:
        """A grid of gaussians covering the box object, standing on the support plane next to the
        board. One gaussian is deliberately elongated with a known rotation for the splat golden."""
        sx, sy = self.board_squares
        # Place the object next to the board, centred beyond the board in +X.
        cx = sx * self.square_mm + 60.0
        cy = (sy * self.square_mm) / 2.0
        ox, oy, oz = self.object_size_mm
        xs = np.linspace(cx - ox / 2, cx + ox / 2, 8)
        ys = np.linspace(cy - oy / 2, cy + oy / 2, 8)
        zs = np.linspace(1.0, oz, 10)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        means = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        n = means.shape[0]
        log_scales = np.full((n, 3), np.log(3.0), np.float32)
        quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))  # identity [w,x,y,z]
        opac = np.full((n,), 3.0, np.float32)                        # sigmoid ~0.95
        f_dc = np.tile(np.array([0.6, 0.3, 0.1], np.float32), (n, 1))
        return GaussianCloud(means=means, log_scales=log_scales, quats_wxyz=quats,
                             opacities=opac, f_dc=f_dc)

    def known_elongated_gaussian_scan(self):
        """A single gaussian with a known 40deg-about-Z rotation and a long local-x axis, for the
        splat orientation golden test. Returns (GaussianCloud, expected_quat_xyzw_scan)."""
        ang = np.deg2rad(40.0)
        qw, qz = np.cos(ang / 2), np.sin(ang / 2)
        quat_wxyz = np.array([[qw, 0, 0, qz]], np.float32)
        cloud = GaussianCloud(
            means=np.array([[150.0, 50.0, 60.0]], np.float32),
            log_scales=np.array([[np.log(25.0), np.log(2.0), np.log(2.0)]], np.float32),
            quats_wxyz=quat_wxyz,
            opacities=np.array([3.0], np.float32),
            f_dc=np.array([[0.5, 0.0, 0.0]], np.float32),
        )
        return cloud, quat_wxyz[0][[1, 2, 3, 0]]

    def _scan_to_model(self, pts_scan: np.ndarray) -> np.ndarray:
        return (self._sim_s * (pts_scan @ self._sim_r.T) + self._sim_t)

    def true_similarity_model_to_scan(self):
        """The ground-truth (R, t, s) mapping the model frame back to the scan frame (for tests)."""
        r = self._sim_r.T
        s = 1.0 / self._sim_s
        t = -s * (self._sim_t @ r.T)
        return r, t, s

    def run(self, frames_dir: Path, frame_names: list[str]) -> GeometryResult:
        """Render dense colour/depth from a camera ring (in the scan frame), express cameras and
        depth in the unknown MODEL frame, and emit pixel-aligned gaussian slabs in a gaussian frame
        that differs from the depth frame by ``gaussian_quirk`` (so runtime alignment is tested)."""
        n = len(frame_names)
        w, h = self.grid_w, self.grid_h
        fx = fy = 0.9 * w
        cx_pix, cy_pix = w / 2.0, h / 2.0
        k = np.array([[fx, 0, cx_pix], [0, fy, cy_pix], [0, 0, 1]], float)

        sx, sy = self.board_squares
        target_scan = np.array([sx * self.square_mm + 60.0, (sy * self.square_mm) / 2.0, 40.0])
        radius = 400.0
        up = np.array([0, 0, 1.0])

        r_sm, t_sm, s_sm = self._sim_r, self._sim_t, self._sim_s
        intrinsics = np.tile(k, (n, 1, 1))
        extr_scan = np.zeros((n, 4, 4))
        depth_scan_list: list[np.ndarray] = []
        conf_list: list[np.ndarray] = []

        for i in range(n):
            ang = 2 * np.pi * i / n
            eye = target_scan + radius * np.array([np.cos(ang), np.sin(ang), 0.4])
            w2c_scan = look_at(eye, target_scan, up)
            extr_scan[i] = w2c_scan
            depth, conf = self._render_depth(w2c_scan, k, w, h)
            depth_scan_list.append(depth)
            conf_list.append(conf)

        # Cameras in the model frame (orthonormal rotation; scale lives in the camera centre).
        extr_model = _cameras_scan_to_model(extr_scan, r_sm, t_sm, s_sm)
        # Depth in model units: scan camera-Z distance scaled by the scan->model scale.
        depth_model = [d * s_sm for d in depth_scan_list]

        # Pixel-aligned gaussian slabs: back-project each view's depth into the model world, then
        # offset that frame by the quirk scale so the gaussian frame may differ from the depth frame.
        gauss_model = self._build_gaussian_slabs(depth_model, conf_list, intrinsics, extr_model,
                                                 w, h)
        if self.gaussian_quirk != 1.0:
            gauss_model = _gaussians_divide(gauss_model, 1.0 / self.gaussian_quirk)

        learned = 1.0 / s_sm  # the model's own metric guess (here, exact for the fake)
        return GeometryResult(
            frames=list(frame_names), width=w, height=h,
            intrinsics=intrinsics, extrinsics=extr_model,
            depth=depth_model, confidence=conf_list, gaussians=gauss_model,
            scale_factor=self.gaussian_quirk if self.gaussian_quirk != 1.0 else None,
            learned_scale=learned,
        )

    def _build_gaussian_slabs(self, depth_model, conf_list, intrinsics, extr_model, w, h):
        """One gaussian per pixel per view (slab k = view k), placed at the back-projected depth
        point in the model world frame -- so they are pixel-aligned and consistent with depth."""
        all_means = []
        for i, depth in enumerate(depth_model):
            k_mat = intrinsics[i]
            w2c = extr_model[i]
            us, vs = np.meshgrid(np.arange(w), np.arange(h))
            z = depth.astype(np.float64).ravel()
            uu = us.ravel()
            vv = vs.ravel()
            x = (uu - k_mat[0, 2]) / k_mat[0, 0] * z
            y = (vv - k_mat[1, 2]) / k_mat[1, 1] * z
            cam = np.stack([x, y, z], axis=1)
            r = w2c[:3, :3]
            t = w2c[:3, 3]
            world = (cam - t) @ r
            # Invalid (zero-depth) pixels collapse to the camera centre; harmless for alignment,
            # which uses only confident, positive-depth pixels.
            all_means.append(world)
        means = np.concatenate(all_means, axis=0).astype(np.float32)
        ng = means.shape[0]
        log_scales = np.full((ng, 3), np.log(2.0), np.float32)
        quats = np.tile(np.array([1, 0, 0, 0], np.float32), (ng, 1))
        opac = np.full((ng,), 3.0, np.float32)
        f_dc = np.tile(np.array([0.5, 0.3, 0.1], np.float32), (ng, 1))
        return GaussianCloud(means=means, log_scales=log_scales, quats_wxyz=quats,
                             opacities=opac, f_dc=f_dc)

    def _render_depth(self, w2c_scan: np.ndarray, k: np.ndarray, w: int, h: int):
        """Dense depth of the scene (board plane Z=0 plus the box object) for one scan camera, in
        scan-frame camera-Z millimetres; 0 where nothing is hit."""
        r = w2c_scan[:3, :3]
        t = w2c_scan[:3, 3]
        cam_centre = -r.T @ t
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        dirs_cam = np.stack([(us - k[0, 2]) / k[0, 0], (vs - k[1, 2]) / k[1, 1],
                             np.ones_like(us, float)], axis=-1)
        dirs_world = dirs_cam @ r
        dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)

        sx, sy = self.board_squares
        bw, bh = sx * self.square_mm, sy * self.square_mm
        ox, oy, oz = self.object_size_mm
        cx = sx * self.square_mm + 60.0
        cy = bh / 2.0
        obj_min = np.array([cx - ox / 2, cy - oy / 2, 0.0])
        obj_max = np.array([cx + ox / 2, cy + oy / 2, oz])

        best_t = np.full((h, w), np.inf)
        # Board plane Z=0 within the board extent.
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = (0.0 - cam_centre[2]) / dirs_world[..., 2]
        hit = cam_centre + dirs_world * tp[..., None]
        on_board = (tp > 0) & (hit[..., 0] >= 0) & (hit[..., 0] <= bw) & \
                   (hit[..., 1] >= 0) & (hit[..., 1] <= bh)
        best_t = np.where(on_board & (tp < best_t), tp, best_t)
        # Object box (AABB slab test).
        inv = np.where(np.abs(dirs_world) < 1e-9, 1e-9, dirs_world)
        t1 = (obj_min - cam_centre) / inv
        t2 = (obj_max - cam_centre) / inv
        tmin = np.maximum.reduce(np.minimum(t1, t2), axis=-1)
        tmax = np.minimum.reduce(np.maximum(t1, t2), axis=-1)
        box_hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        best_t = np.where(box_hit & (tmin < best_t), tmin, best_t)

        valid = np.isfinite(best_t)
        hit_world = cam_centre + dirs_world * np.where(valid, best_t, 0.0)[..., None]
        z_cam = (hit_world - cam_centre) @ r[2]
        depth = np.where(valid & (z_cam > 0), z_cam, 0.0).astype(np.float32)
        conf = np.where(depth > 0, 0.9, 0.0).astype(np.float32)
        return depth, conf

    # -- textured colour rendering (for the frames stage / board detection) --

    def _board_texture_image(self) -> np.ndarray:
        """Cache a high-res ChArUco board image (grey), matching the printable board geometry."""
        from . import board as board_mod

        if self._board_texture is None:
            board = board_mod.charuco_board()
            # ~10 px per mm so the warped board stays crisp.
            sx, sy = self.board_squares
            px_w = int(sx * self.square_mm * 8)
            px_h = int(sy * self.square_mm * 8)
            self._board_texture = board.generateImage((px_w, px_h), marginSize=0)
        return self._board_texture

    def _render_color(self, w2c_scan: np.ndarray, k: np.ndarray, w: int, h: int) -> np.ndarray:
        """Render a colour image: the ChArUco board warped onto the Z=0 plane plus a grey object
        splat, so cv2.aruco.CharucoDetector can recover the board corners."""
        import cv2

        img = np.full((h, w), 200, np.uint8)  # light background
        tex = self._board_texture_image()
        th, tw = tex.shape
        sx, sy = self.board_squares
        bw = sx * self.square_mm
        bh = sy * self.square_mm
        # Board plane corners in scan coords (Z=0), matching getChessboardCorners origin.
        board_pts_scan = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], float)
        tex_pts = np.array([[0, th], [tw, th], [tw, 0], [0, 0]], np.float32)  # tex y-down
        img_pts = self._project(board_pts_scan, w2c_scan, k)
        if img_pts is None:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        hom, _ = cv2.findHomography(tex_pts, img_pts.astype(np.float32))
        warped = cv2.warpPerspective(tex, hom, (w, h), borderValue=255)
        # Board mask: where the quad projects.
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, img_pts.astype(np.int32), 255)
        img[mask > 0] = warped[mask > 0]
        # Object: a filled grey polygon where its gaussians project.
        gauss = self._object_gaussians_scan()
        obj_px = self._project(gauss.means.astype(float), w2c_scan, k)
        if obj_px is not None and len(obj_px) >= 3:
            hull = cv2.convexHull(obj_px.astype(np.int32))
            cv2.fillConvexPoly(img, hull, 120)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def _project(self, pts_scan: np.ndarray, w2c_scan: np.ndarray, k: np.ndarray):
        r = w2c_scan[:3, :3]
        t = w2c_scan[:3, 3]
        cam = pts_scan @ r.T + t
        if np.any(cam[:, 2] <= 1e-3):
            cam = cam[cam[:, 2] > 1e-3]
            if cam.shape[0] < 4:
                return None
        proj = (k @ cam.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        return uv

    def render_frames(self, scan, storage, frames_dir: Path):
        """Render the synthetic scene to full-res and model-grid frames + frames.json, so the
        pipeline's frames stage can run without ffmpeg (end-to-end CPU test)."""
        from PIL import Image

        from .frames import FrameEntry, FramesIndex, axis_mapping

        frames_dir = Path(frames_dir)
        model_dir = frames_dir / "model"
        frames_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        w, h = self.grid_w, self.grid_h
        full_w, full_h = w * 2, h * 2  # a "full-res" copy at 2x the model grid
        fx = fy = 0.9 * w
        k_model = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], float)
        k_full = k_model.copy()
        k_full[:2] *= 2.0

        sx, sy = self.board_squares
        target_scan = np.array([sx * self.square_mm + 60.0, (sy * self.square_mm) / 2.0, 40.0])
        radius = 400.0
        up = np.array([0, 0, 1.0])
        mapping = axis_mapping(full_w, full_h, w, h)

        entries = []
        for i in range(self.n_views):
            ang = 2 * np.pi * i / self.n_views
            eye = target_scan + radius * np.array([np.cos(ang), np.sin(ang), 0.4])
            w2c = look_at(eye, target_scan, up)
            full_img = self._render_color(w2c, k_full, full_w, full_h)
            model_img = self._render_color(w2c, k_model, w, h)
            name = f"{i + 1:04d}.jpg"
            Image.fromarray(full_img).save(frames_dir / name, quality=95)
            Image.fromarray(model_img).save(model_dir / name, quality=95)
            entries.append(FrameEntry(frame=name, sharpness=100.0,
                                      full_res=f"frames/{name}", model=f"frames/model/{name}",
                                      grid=mapping))
        index = FramesIndex(source="synthetic", fps=0.0, total_sampled=self.n_views,
                            kept=self.n_views, containment="synthetic (no ffmpeg)", frames=entries)
        (frames_dir / "frames.json").write_text(json.dumps(index.to_json(), indent=2))
        return index


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _cameras_scan_to_model(extr_scan: np.ndarray, r_sm: np.ndarray, t_sm: np.ndarray,
                           s_sm: float) -> np.ndarray:
    """Given world->camera extrinsics defined in the SCAN frame and the scan->model similarity
    (x_model = s R X + t), return world->camera extrinsics in the MODEL frame, keeping the rotation
    orthonormal (the scale lives in the camera centre / translation)."""
    out = np.zeros_like(extr_scan)
    for i in range(extr_scan.shape[0]):
        r_wc = extr_scan[i, :3, :3]
        t_wc = extr_scan[i, :3, 3]
        c_scan = -r_wc.T @ t_wc
        r_new = r_wc @ r_sm.T                   # orthonormal
        c_model = s_sm * (r_sm @ c_scan) + t_sm
        t_new = -r_new @ c_model
        out[i, :3, :3] = r_new
        out[i, :3, 3] = t_new
        out[i, 3, 3] = 1.0
    return out


def _gaussians_divide(cloud: GaussianCloud, factor: float) -> GaussianCloud:
    """Scale gaussian means and log-scales by 1/factor (used by the fake to move the gaussian frame
    away from the depth frame, reproducing a 'quirk present' case for the runtime aligner)."""
    out = cloud.select(np.ones(len(cloud), dtype=bool))
    out.means = (out.means / np.float32(factor)).astype(np.float32)
    out.log_scales = (out.log_scales - np.float32(np.log(factor))).astype(np.float32)
    return out
