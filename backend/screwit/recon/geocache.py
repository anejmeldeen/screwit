"""Video-keyed geometry cache (P2).

The geometry stage (DA3 inference) is the slowest part of the pipeline -- minutes on CPU. Its output
depends only on the input video and a few knobs (view cap, device, model id, infer_gs), so re-running
the same video can reuse the previous result instead of re-solving. This module serializes a
``GeometryResult`` to a single ``.npz`` keyed by a hash of (video bytes + those knobs) and reloads it.

The cache lives on the filesystem under ``SCREWIT_GEOCACHE_DIR`` (default: ``<SCREWIT_DATA_DIR>/
geometry-cache``), NOT through the Storage abstraction: the key is the video, not a scan, so a cache
entry is shared across scans of the same clip, and the storage key validator only permits ``scans/``
keys. Set ``SCREWIT_GEOCACHE=0`` to disable.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np

from screwit.recon.gaussians import GaussianCloud
from screwit.recon.geometry import GeometryResult


def cache_enabled() -> bool:
    return os.environ.get("SCREWIT_GEOCACHE", "1").strip().lower() not in ("0", "false", "off", "no")


def _cache_dir() -> Path:
    explicit = os.environ.get("SCREWIT_GEOCACHE_DIR")
    if explicit:
        d = Path(explicit)
    else:
        data_dir = os.environ.get("SCREWIT_DATA_DIR") or str(Path.cwd() / "data")
        d = Path(data_dir) / "geometry-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_key(video_bytes: bytes, *, model_id: str, view_cap: int, device: str,
              infer_gs: bool, extra: str = "") -> str:
    """A stable key for a geometry result: the video content plus everything that changes the output."""
    h = hashlib.sha256()
    h.update(video_bytes)
    h.update(f"|model={model_id}|views={view_cap}|device={device}|gs={int(infer_gs)}|{extra}".encode())
    return h.hexdigest()[:32]


def _path(key: str) -> Path:
    return _cache_dir() / f"{key}.npz"


def load(key: str) -> GeometryResult | None:
    """Return the cached GeometryResult for ``key``, or None on a miss or any decode problem."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            n = int(z["n_frames"])
            h, w = int(z["height"]), int(z["width"])
            frames = [str(x) for x in z["frames"].tolist()]
            depth = [z[f"depth_{i}"] for i in range(n)]
            conf = [z[f"conf_{i}"] for i in range(n)]
            g = GaussianCloud(
                means=z["g_means"], log_scales=z["g_log_scales"], quats_wxyz=z["g_quats"],
                opacities=z["g_opacities"], f_dc=z["g_f_dc"],
            )
            sf = z["scale_factor"]
            ls = z["learned_scale"]
            scale_factor = None if np.isnan(sf) else float(sf)
            learned_scale = None if np.isnan(ls) else float(ls)
            return GeometryResult(
                frames=frames, width=w, height=h,
                intrinsics=z["intrinsics"], extrinsics=z["extrinsics"],
                depth=depth, confidence=conf, gaussians=g,
                scale_factor=scale_factor, learned_scale=learned_scale,
            )
    except Exception:  # noqa: BLE001 - a corrupt/old cache entry must never break a scan
        return None


def store(key: str, result: GeometryResult) -> None:
    """Serialize ``result`` to the cache. Best-effort: a write failure must not break the scan."""
    try:
        arrays: dict[str, np.ndarray] = {
            "n_frames": np.asarray(len(result.frames)),
            "width": np.asarray(result.width),
            "height": np.asarray(result.height),
            "frames": np.asarray(result.frames, dtype=object if False else "U32"),
            "intrinsics": np.asarray(result.intrinsics, dtype=np.float64),
            "extrinsics": np.asarray(result.extrinsics, dtype=np.float64),
            "g_means": result.gaussians.means,
            "g_log_scales": result.gaussians.log_scales,
            "g_quats": result.gaussians.quats_wxyz,
            "g_opacities": result.gaussians.opacities,
            "g_f_dc": result.gaussians.f_dc,
            "scale_factor": np.asarray(np.nan if result.scale_factor is None else result.scale_factor),
            "learned_scale": np.asarray(np.nan if result.learned_scale is None else result.learned_scale),
        }
        for i, (d, c) in enumerate(zip(result.depth, result.confidence, strict=True)):
            arrays[f"depth_{i}"] = np.asarray(d, dtype=np.float32)
            arrays[f"conf_{i}"] = np.asarray(c, dtype=np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        tmp = _path(key).with_suffix(".npz.tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(_path(key))  # atomic: never leave a half-written entry
    except Exception:  # noqa: BLE001
        pass
