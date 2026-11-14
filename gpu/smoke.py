#!/usr/bin/env python3
"""GPU smoke test for Screw It: DA3, MapAnything, SAM 2, Grounding DINO on one video.

Run on a CUDA host after setup.sh. Writes gpu/out/<run>/report.json and report.md.
`--dry-run` exercises argument parsing, report writing and the VRAM-free logic with
no torch and no GPU (used by pytest on the dev machine).

Upstream APIs relied on (all checked against the projects' READMEs / source, see gpu/README.md):
  Depth Anything 3: depth_anything_3.api.DepthAnything3.from_pretrained, model.inference(...)
  MapAnything: mapanything.models.MapAnything.from_pretrained, mapanything.utils.image.load_images, model.infer(...)
  SAM 2: sam2.build_sam.build_sam2_video_predictor, init_state, add_new_points_or_box, propagate_in_video
  Grounding DINO: transformers AutoProcessor / AutoModelForZeroShotObjectDetection (as Grounded-SAM-2's HF demo)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_VIEWS = (32, 48, 64)
DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
MAPANYTHING_MODEL_ID = "facebook/map-anything-apache"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT_NAME = "sam2.1_hiera_large.pt"
STAGES = ("da3", "mapanything", "sam2", "gdino")
DEFAULT_VRAM_THRESHOLD_MB = 1024.0
GPU_ROOT = Path(__file__).resolve().parent


class VramNotFreed(RuntimeError):
    """Raised when GPU memory is still allocated after the release step."""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    run: str
    created_utc: str
    dry_run: bool
    args: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "created_utc": self.created_utc,
            "dry_run": self.dry_run,
            "args": self.args,
            "environment": self.environment,
            "frames": self.frames,
            "stages": self.stages,
            "notes": self.notes,
        }


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "item") and callable(o.item) and getattr(o, "ndim", 1) == 0:
        return _jsonable(o.item())
    if hasattr(o, "tolist"):
        return _jsonable(o.tolist())
    return o


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# GPU smoke report: {report['run']}", ""]
    lines.append(f"- created (UTC): {report['created_utc']}")
    lines.append(f"- dry run: {report['dry_run']}")
    env = report.get("environment", {})
    for k, v in env.items():
        lines.append(f"- {k}: {v}")
    frames = report.get("frames", {})
    if frames:
        lines += ["", "## Frames", ""]
        for k, v in frames.items():
            lines.append(f"- {k}: {v}")
    lines += ["", "## Stages", ""]
    for name, stage in report.get("stages", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        if isinstance(stage, dict) and "runs" in stage:
            lines.append("| views | status | wall s | peak alloc MB | peak reserved MB | notes |")
            lines.append("|---|---|---|---|---|---|")
            for r in stage["runs"]:
                note = r.get("error") or r.get("orbit_verdict") or ""
                lines.append(
                    f"| {r.get('views')} | {r.get('status')} | {_fmt(r.get('wall_s'))} | "
                    f"{_fmt(r.get('peak_allocated_mb'))} | {_fmt(r.get('peak_reserved_mb'))} | {note} |"
                )
            extra = {k: v for k, v in stage.items() if k != "runs"}
            if extra:
                lines += ["", "```json", json.dumps(_jsonable(extra), indent=2), "```"]
            lines += ["", "<details><summary>full per-run detail</summary>", "", "```json"]
            lines.append(json.dumps(_jsonable(stage["runs"]), indent=2))
            lines += ["```", "", "</details>"]
        else:
            lines += ["```json", json.dumps(_jsonable(stage), indent=2), "```"]
        lines.append("")
    if report.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in report["notes"]] + [""]
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return "" if v is None else str(v)


def write_report(report: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _jsonable(report.to_dict())
    jp = out_dir / "report.json"
    mp = out_dir / "report.md"
    jp.write_text(json.dumps(data, indent=2) + "\n")
    mp.write_text(render_markdown(data))
    return jp, mp


# ---------------------------------------------------------------------------
# VRAM handling (SPEC 6.1: free DA3 before SAM 2 loads, assert memory dropped)
# ---------------------------------------------------------------------------


def release_prediction(pred: Any) -> list[str]:
    """Drop the GPU-resident members DA3 leaves on a prediction. Returns what was cleared."""
    cleared: list[str] = []
    if pred is None:
        return cleared
    for attr in ("gaussians", "aux"):
        if getattr(pred, attr, None) is not None:
            try:
                setattr(pred, attr, None)
                cleared.append(attr)
            except (AttributeError, TypeError):
                pass
    return cleared


def free_gpu(torch_mod: Any, holders: dict[str, Any], threshold_mb: float) -> dict[str, Any]:
    """Delete holders, gc, empty the CUDA cache, assert allocated memory is under the threshold.

    `holders` is a mutable dict of name -> object; entries are removed (the caller must not keep
    other references). Returns before/after numbers. Raises VramNotFreed if still above threshold.
    """
    cuda = torch_mod.cuda
    before = cuda.memory_allocated() / 2**20
    cleared: dict[str, list[str]] = {}
    for name in list(holders):
        cleared[name] = release_prediction(holders[name]) if name.endswith("prediction") else []
        del holders[name]
    gc.collect()
    cuda.synchronize()
    cuda.empty_cache()
    after = cuda.memory_allocated() / 2**20
    result = {
        "allocated_before_mb": before,
        "allocated_after_mb": after,
        "threshold_mb": threshold_mb,
        "dropped": cleared,
        "reserved_after_mb": cuda.memory_reserved() / 2**20,
    }
    if after >= threshold_mb:
        raise VramNotFreed(
            f"{after:.0f} MB still allocated after release (threshold {threshold_mb:.0f} MB)"
        )
    return result


@contextmanager
def measure(torch_mod: Any | None) -> Iterator[dict[str, Any]]:
    """Wall time and peak CUDA memory for the enclosed block."""
    rec: dict[str, Any] = {}
    if torch_mod is not None:
        torch_mod.cuda.synchronize()
        torch_mod.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        if torch_mod is not None:
            torch_mod.cuda.synchronize()
            rec["peak_allocated_mb"] = torch_mod.cuda.max_memory_allocated() / 2**20
            rec["peak_reserved_mb"] = torch_mod.cuda.max_memory_reserved() / 2**20
        rec["wall_s"] = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Camera convention check (pure numpy, tested on CPU)
# ---------------------------------------------------------------------------


def _orbit_metrics(centres: Any, axes: Any | None = None) -> dict[str, Any]:
    import numpy as np

    c = np.asarray(centres, dtype=np.float64)
    mean = c.mean(axis=0)
    u, s, vt = np.linalg.svd(c - mean, full_matrices=False)
    normal = vt[2]
    e1, e2 = vt[0], vt[1]
    rel = c - mean
    x, y = rel @ e1, rel @ e2
    radius = np.hypot(x, y)
    r_mean = float(radius.mean()) if radius.size else 0.0
    ang = np.sort(np.arctan2(y, x))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
    span_deg = float(np.degrees(2 * math.pi - gaps.max())) if len(ang) > 1 else 0.0
    out = {
        "n": int(len(c)),
        "centroid": mean.tolist(),
        "mean_radius": r_mean,
        "radius_cv": float(radius.std() / r_mean) if r_mean > 1e-12 else None,
        "plane_thickness_ratio": float(np.abs(rel @ normal).std() / r_mean) if r_mean > 1e-12 else None,
        "angular_span_deg": span_deg,
    }
    if axes is not None:
        d = np.asarray(axes, dtype=np.float64)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        a = np.zeros((3, 3))
        b = np.zeros(3)
        for ci, di in zip(c, d, strict=True):
            m = np.eye(3) - np.outer(di, di)
            a += m
            b += m @ ci
        try:
            p = np.linalg.solve(a, b)
            out["look_point_offset_over_radius"] = float(np.linalg.norm(p - mean) / r_mean) if r_mean > 1e-12 else None
        except np.linalg.LinAlgError:
            out["look_point_offset_over_radius"] = None
    return out


def check_extrinsics(extrinsics: Any) -> dict[str, Any]:
    """Check an (N,3,4) or (N,4,4) world-to-camera stack.

    Camera centre for world-to-camera [R|t] is C = -R^T t. Also reports the metrics obtained if the
    matrices were camera-to-world instead (C = t), so a wrong convention shows as the worse orbit.
    """
    import numpy as np

    e = np.asarray(extrinsics, dtype=np.float64)
    if e.ndim != 3 or e.shape[2] != 4 or e.shape[1] not in (3, 4):
        return {"verdict": "unexpected_shape", "shape": list(e.shape)}
    r = e[:, :3, :3]
    t = e[:, :3, 3]
    ortho = float(np.abs(np.einsum("nij,nkj->nik", r, r) - np.eye(3)).max())
    dets = np.linalg.det(r)
    centres = -np.einsum("nji,nj->ni", r, t)
    forward_world = r[:, 2, :]
    m_w2c = _orbit_metrics(centres, forward_world)
    m_c2w = _orbit_metrics(t, r[:, :, 2])

    def plausible(m: dict[str, Any]) -> bool:
        return bool(
            m["radius_cv"] is not None
            and m["radius_cv"] < 0.35
            and m["plane_thickness_ratio"] < 0.35
            and m["angular_span_deg"] > 90
            and (m.get("look_point_offset_over_radius") or 0) < 0.5
        )

    ok_w2c, ok_c2w = plausible(m_w2c), plausible(m_c2w)
    if ok_w2c and not ok_c2w:
        verdict = "world_to_camera_consistent"
    elif ok_c2w and not ok_w2c:
        verdict = "looks_camera_to_world"
    elif ok_w2c and ok_c2w:
        verdict = "ambiguous_both_plausible"
    else:
        verdict = "no_plausible_orbit"
    return {
        "shape": list(e.shape),
        "rotation_orthonormality_max_err": ortho,
        "det_min": float(dets.min()),
        "det_max": float(dets.max()),
        "orbit_if_world_to_camera": m_w2c,
        "orbit_if_camera_to_world": m_c2w,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("no ffmpeg: install imageio-ffmpeg or put ffmpeg on PATH") from None
        return exe


def probe_duration_s(video: Path) -> float:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        raise RuntimeError(f"could not read duration from ffmpeg output for {video}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def pick_evenly(items: list[Any], n: int) -> list[Any]:
    """Pick n items spread evenly over the list, keeping order."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def extract_frames(video: Path, out_dir: Path, max_views: int, long_side: int = 1280) -> dict[str, Any]:
    """Minimal extraction: uniform fps, drop the blurriest 30% (variance of Laplacian), downscale.

    screwit.recon.frames is only probed for presence: no entry-point name is confirmed in the
    contracts, so this kit always uses its own extraction and records that fact.
    """
    info: dict[str, Any] = {"extractor": "gpu/smoke.py ffmpeg (own, minimal)"}
    try:
        importlib.import_module("screwit.recon.frames")
        info["screwit_recon_frames_importable"] = True
    except Exception:
        info["screwit_recon_frames_importable"] = False
    duration = probe_duration_s(video)
    candidates = math.ceil(max_views / 0.7) + 2
    fps = candidates / max(duration, 0.1)
    raw = out_dir / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for old in raw.glob("*.jpg"):
        old.unlink()
    vf = f"fps={fps:.5f},scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", vf,
         "-qscale:v", "2", str(raw / "f_%05d.jpg")],
        check=True,
    )
    files = sorted(raw.glob("f_*.jpg"))
    info.update({"duration_s": duration, "sample_fps": fps, "extracted": len(files)})
    try:
        import cv2

        scored = [(float(cv2.Laplacian(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()), f) for f in files]
        keep_n = max(1, int(len(files) * 0.7))
        keep = {f for _, f in sorted(scored, key=lambda x: -x[0])[:keep_n]}
        files = [f for f in files if f in keep]
        info["blur_filter"] = "dropped blurriest 30% by variance of Laplacian"
    except ImportError:
        info["blur_filter"] = "skipped (cv2 not installed)"
    info["kept"] = len(files)
    info["files"] = [str(f) for f in files]
    return info


# ---------------------------------------------------------------------------
# Stages (real; need CUDA)
# ---------------------------------------------------------------------------


def _tensor_summary(x: Any) -> dict[str, Any]:
    return {
        "shape": list(getattr(x, "shape", [])),
        "dtype": str(getattr(x, "dtype", "")),
        "device": str(getattr(x, "device", "cpu")),
    }


def run_da3(torch_mod: Any, frames: list[str], views_list: tuple[int, ...], out_dir: Path,
            threshold_mb: float) -> dict[str, Any]:
    import numpy as np
    from depth_anything_3.api import DepthAnything3  # README Quick Start

    stage: dict[str, Any] = {"model": DA3_MODEL_ID, "runs": []}
    t0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device=torch_mod.device("cuda"))
    stage["load_s"] = time.perf_counter() - t0
    stage["post_load_allocated_mb"] = torch_mod.cuda.memory_allocated() / 2**20
    for n in views_list:
        sel = _evenly(frames, n)
        run_dir = out_dir / f"da3_{n}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {"views": len(sel), "requested_views": n}
        pred = None
        try:
            with measure(torch_mod) as m:
                pred = model.inference(
                    sel,
                    infer_gs=True,
                    ref_view_strategy="middle",
                    export_dir=str(run_dir / "export"),
                    export_format="mini_npz-gs_ply",
                    export_kwargs={"gs_ply": {"gs_views_interval": 4}},
                )
            rec.update(m)
            rec["status"] = "ok"
            rec["shapes"] = {
                k: list(getattr(pred, k).shape)
                for k in ("depth", "conf", "extrinsics", "intrinsics", "processed_images")
                if getattr(pred, k, None) is not None
            }
            rec["is_metric"] = getattr(pred, "is_metric", None)
            rec["scale_factor"] = getattr(pred, "scale_factor", None)
            gs = getattr(pred, "gaussians", None)
            aux = getattr(pred, "aux", None)
            rec["gaussians_attribute"] = {
                "prediction.gaussians_present": gs is not None,
                "aux_keys": sorted(aux) if isinstance(aux, dict) else None,
                "aux_has_gaussians": bool(isinstance(aux, dict) and any("gauss" in k.lower() for k in aux)),
            }
            if gs is not None:
                rec["gaussians_attribute"]["fields"] = {
                    f: _tensor_summary(getattr(gs, f, None)) for f in ("means", "scales", "rotations", "harmonics", "opacities")  # noqa: E501
                    if getattr(gs, f, None) is not None
                }
            exp = run_dir / "export"
            rec["export_files"] = {p.name: p.stat().st_size for p in sorted(exp.rglob("*")) if p.is_file()}
            if pred.extrinsics is not None:
                rec["extrinsics_check"] = check_extrinsics(pred.extrinsics)
                rec["orbit_verdict"] = rec["extrinsics_check"]["verdict"]
            np.savez_compressed(
                run_dir / "outputs.npz",
                depth=pred.depth, conf=pred.conf if pred.conf is not None else np.zeros(0),
                extrinsics=pred.extrinsics, intrinsics=pred.intrinsics,
                scale_factor=np.float64(pred.scale_factor if pred.scale_factor is not None else np.nan),
            )
            rec["outputs_npz"] = str(run_dir / "outputs.npz")
        except torch_mod.cuda.OutOfMemoryError as e:
            rec["status"] = "oom"
            rec["error"] = str(e).splitlines()[0]
            rec["peak_allocated_mb"] = torch_mod.cuda.max_memory_allocated() / 2**20
        finally:
            release_prediction(pred)
            del pred
            gc.collect()
            torch_mod.cuda.empty_cache()
        stage["runs"].append(rec)
    holders = {"da3_model": model}
    del model
    stage["free"] = free_gpu(torch_mod, holders, threshold_mb)
    return stage


def _evenly(items: list[Any], n: int) -> list[Any]:
    return pick_evenly(items, n)


def run_mapanything(torch_mod: Any, frames: list[str], views_list: tuple[int, ...], out_dir: Path,
                    threshold_mb: float) -> dict[str, Any]:
    import numpy as np
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    stage: dict[str, Any] = {"model": MAPANYTHING_MODEL_ID, "runs": []}
    t0 = time.perf_counter()
    model = MapAnything.from_pretrained(MAPANYTHING_MODEL_ID).to("cuda")
    stage["load_s"] = time.perf_counter() - t0
    for n in views_list:
        sel = _evenly(frames, n)
        run_dir = out_dir / f"mapanything_{n}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {"views": len(sel), "requested_views": n}
        preds = None
        try:
            views = load_images(sel)
            with measure(torch_mod) as m:
                preds = model.infer(
                    views,
                    memory_efficient_inference=True,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                )
            rec.update(m)
            rec["status"] = "ok"
            c2w = np.stack([p["camera_poses"].float().cpu().numpy().reshape(4, 4) for p in preds])
            w2c = np.linalg.inv(c2w)
            rec["shapes"] = {
                "camera_poses": list(c2w.shape),
                "pts3d": list(preds[0]["pts3d"].shape),
                "depth_z": list(preds[0]["depth_z"].shape),
                "conf": list(preds[0]["conf"].shape),
                "intrinsics": list(preds[0]["intrinsics"].shape),
            }
            msf = [float(p["metric_scaling_factor"].float().mean().item()) for p in preds]
            rec["metric_scaling_factor"] = {"min": min(msf), "max": max(msf), "mean": sum(msf) / len(msf)}
            rec["extrinsics_check"] = check_extrinsics(w2c[:, :3, :])
            rec["orbit_verdict"] = rec["extrinsics_check"]["verdict"]
            np.savez_compressed(
                run_dir / "outputs.npz",
                extrinsics=w2c[:, :3, :],
                intrinsics=np.stack([p["intrinsics"].float().cpu().numpy().reshape(3, 3) for p in preds]),
                depth=np.stack([p["depth_z"].float().cpu().numpy().squeeze() for p in preds]),
                conf=np.stack([p["conf"].float().cpu().numpy().squeeze() for p in preds]),
                metric_scaling_factor=np.asarray(msf),
            )
            rec["outputs_npz"] = str(run_dir / "outputs.npz")
        except torch_mod.cuda.OutOfMemoryError as e:
            rec["status"] = "oom"
            rec["error"] = str(e).splitlines()[0]
            rec["peak_allocated_mb"] = torch_mod.cuda.max_memory_allocated() / 2**20
        finally:
            preds = None
            views = None
            gc.collect()
            torch_mod.cuda.empty_cache()
        stage["runs"].append(rec)
    holders = {"mapanything_model": model}
    del model
    stage["free"] = free_gpu(torch_mod, holders, threshold_mb)
    return stage


def run_sam2(torch_mod: Any, frames: list[str], out_dir: Path, ckpt: Path, views: int) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    sel = _evenly(frames, views)
    vdir = out_dir / "sam2_frames"
    vdir.mkdir(parents=True, exist_ok=True)
    for old in vdir.glob("*.jpg"):
        old.unlink()
    for i, f in enumerate(sel):
        os.symlink(os.path.abspath(f), vdir / f"{i:05d}.jpg")  # SAM 2 reads a folder of <int>.jpg
    w, h = Image.open(sel[0]).size
    box = np.array([w * 0.3, h * 0.3, w * 0.7, h * 0.7], dtype=np.float32)
    stage: dict[str, Any] = {"config": SAM2_CONFIG, "checkpoint": str(ckpt), "frames": len(sel), "prompt_box_xyxy": box.tolist()}  # noqa: E501
    t0 = time.perf_counter()
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(ckpt))
    stage["load_s"] = time.perf_counter() - t0
    areas: list[float] = []
    with measure(torch_mod) as m:
        with torch_mod.inference_mode(), torch_mod.autocast("cuda", dtype=torch_mod.bfloat16):
            state = predictor.init_state(str(vdir))
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
            for _fi, _ids, masks in predictor.propagate_in_video(state):
                areas.append(float((masks[0] > 0).float().mean().item()))
    stage.update(m)
    stage["status"] = "ok"
    stage["frames_propagated"] = len(areas)
    stage["mask_area_fraction"] = {"min": min(areas), "max": max(areas), "mean": sum(areas) / len(areas)} if areas else None  # noqa: E501
    del state, predictor
    gc.collect()
    torch_mod.cuda.empty_cache()
    stage["allocated_after_release_mb"] = torch_mod.cuda.memory_allocated() / 2**20
    return stage


def run_gdino(torch_mod: Any, frames: list[str], prompt: str, threshold_mb_unused: float = 0.0) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    text = prompt.strip().lower()
    if not text.endswith("."):
        text += "."  # Grounded-SAM-2 demo: queries must be lowercase and end with a dot
    stage: dict[str, Any] = {"model": GDINO_MODEL_ID, "text_prompt": text}
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to("cuda")
    stage["load_s"] = time.perf_counter() - t0
    image = Image.open(frames[len(frames) // 2]).convert("RGB")
    with measure(torch_mod) as m:
        inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
        with torch_mod.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
        )
    stage.update(m)
    stage["status"] = "ok"
    r = results[0]
    stage["detections"] = [
        {"label": lab, "score": float(s), "box_xyxy": [float(v) for v in b]}
        for lab, s, b in zip(r["labels"], r["scores"], r["boxes"], strict=False)
    ]
    del model, processor, outputs, inputs
    gc.collect()
    torch_mod.cuda.empty_cache()
    return stage


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_views(s: str) -> tuple[int, ...]:
    try:
        vals = tuple(int(x) for x in s.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--views must be comma-separated integers: {s!r}") from e
    if not vals or any(v < 3 for v in vals):
        raise argparse.ArgumentTypeError("--views needs at least one value and each must be >= 3")
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="phone video (mp4/mov/webm)")
    p.add_argument("--board-spec", type=Path, default=None, help="optional board spec JSON (recorded in the report only)")  # noqa: E501
    p.add_argument("--views", type=parse_views, default=DEFAULT_VIEWS, help="comma-separated view counts (default 32,48,64)")  # noqa: E501
    p.add_argument("--run", default=None, help="run name (default: UTC timestamp)")
    p.add_argument("--out-root", type=Path, default=GPU_ROOT / "out", help="output root (default gpu/out)")
    p.add_argument("--cache", type=Path, default=Path(os.environ.get("SCREWIT_GPU_CACHE", Path.home() / "screwit-gpu-cache")),  # noqa: E501
                   help="weights cache dir written by setup.sh")
    p.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    p.add_argument("--prompt", default="object.", help="Grounding DINO text prompt")
    p.add_argument("--vram-threshold-mb", type=float, default=DEFAULT_VRAM_THRESHOLD_MB,
                   help="max torch.cuda.memory_allocated (MB) after freeing a model; above it the run fails")
    p.add_argument("--dry-run", action="store_true", help="no torch, no GPU, no video decoding; exercises argument parsing, report writing and free-VRAM logic")  # noqa: E501
    return p


def environment_info(torch_mod: Any | None) -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    if torch_mod is not None:
        env["torch"] = torch_mod.__version__
        env["torch_cuda"] = torch_mod.version.cuda
        env["gpu"] = torch_mod.cuda.get_device_name(0)
        env["gpu_total_mb"] = torch_mod.cuda.get_device_properties(0).total_memory / 2**20
    return env


class _FakeCuda:
    def __init__(self) -> None:
        self.allocated = 20_000.0 * 2**20
        self.reserved = 22_000.0 * 2**20

    def memory_allocated(self) -> float:
        return self.allocated

    def memory_reserved(self) -> float:
        return self.reserved

    def synchronize(self) -> None:
        pass

    def empty_cache(self) -> None:
        self.reserved = self.allocated


class _FakeTorch:
    """Stand-in used only by --dry-run so the free-VRAM logic runs without torch."""

    def __init__(self, leak: bool = False) -> None:
        self.cuda = _FakeCuda()
        self.leak = leak


def dry_run_free_vram(threshold_mb: float, leak: bool = False) -> dict[str, Any]:
    fake = _FakeTorch(leak=leak)

    class _Pred:
        gaussians = object()
        aux = {"k": 1}

    holders = {"da3_prediction": _Pred(), "da3_model": object()}
    if not leak:
        fake.cuda.allocated = 100.0 * 2**20
    return free_gpu(fake, holders, threshold_mb)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HOME", str(args.cache / "hf"))
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"unknown stage(s): {bad}; choose from {STAGES}", file=sys.stderr)
        return 2
    run = args.run or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_root / run
    report = Report(
        run=run,
        created_utc=datetime.now(UTC).isoformat(),
        dry_run=args.dry_run,
        args={
            "video": str(args.video), "board_spec": str(args.board_spec) if args.board_spec else None,
            "views": list(args.views), "stages": list(stages), "prompt": args.prompt,
            "vram_threshold_mb": args.vram_threshold_mb,
        },
    )
    if args.board_spec:
        if not args.board_spec.is_file():
            print(f"board spec not found: {args.board_spec}", file=sys.stderr)
            return 2
        report.args["board_spec_content"] = json.loads(args.board_spec.read_text())

    if args.dry_run:
        report.environment = environment_info(None)
        report.stages["free_vram_logic"] = dry_run_free_vram(args.vram_threshold_mb)
        try:
            dry_run_free_vram(args.vram_threshold_mb, leak=True)
            report.stages["free_vram_leak_detected"] = False
        except VramNotFreed as e:
            report.stages["free_vram_leak_detected"] = True
            report.notes.append(f"leak case raised as expected: {e}")
        jp, mp = write_report(report, out_dir)
        print(f"dry run ok: {jp} {mp}")
        return 0

    if not args.video.is_file():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print("torch is not installed: run gpu/setup.sh first (or use --dry-run)", file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False: this kit needs a CUDA host", file=sys.stderr)
        return 3

    report.environment = environment_info(torch)
    out_dir.mkdir(parents=True, exist_ok=True)
    finfo = extract_frames(args.video, out_dir / "frames", max(args.views))
    frames = finfo.pop("files")
    report.frames = finfo
    ckpt = args.cache / "weights" / SAM2_CKPT_NAME
    ok = True

    def guarded(name: str, fn: Callable[[], dict[str, Any]]) -> None:
        nonlocal ok
        try:
            report.stages[name] = fn()
        except Exception as e:  # keep going so one broken model does not hide the others
            ok = False
            report.stages[name] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            write_report(report, out_dir)

    if "da3" in stages:
        guarded("da3", lambda: run_da3(torch, frames, args.views, out_dir, args.vram_threshold_mb))
    if "mapanything" in stages:
        guarded("mapanything", lambda: run_mapanything(torch, frames, args.views, out_dir, args.vram_threshold_mb))
    if "sam2" in stages:
        guarded("sam2", lambda: run_sam2(torch, frames, out_dir, ckpt, min(args.views)))
    if "gdino" in stages:
        guarded("gdino", lambda: run_gdino(torch, frames, args.prompt))
    jp, mp = write_report(report, out_dir)
    print(f"wrote {jp} and {mp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
