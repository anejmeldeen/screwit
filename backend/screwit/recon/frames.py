"""Frame extraction (SPEC 6.1 step 1).

Probe the input with ffmpeg (no ffprobe; parse ``ffmpeg -i`` stderr), pick a sampling rate from the
duration so that 48-64 keyframes survive after dropping the blurriest 30% (variance of Laplacian),
tone-map 10-bit HLG/PQ to 8-bit, and keep two copies of each keyframe: full resolution for board
detection and the agent's close-ups, and the model-grid copy (long side 504, each side rounded to a
multiple of 14, with the exact per-axis mapping recorded in frames.json).

All ffmpeg calls use ONE helper (``_run_ffmpeg``) that applies a wall-clock timeout, CPU/memory/pids
rlimits, and a no-network sandbox when one is available unprivileged (SPEC 3.3 containment). RLIMIT_CPU
counts CPU time over all of ffmpeg's threads, so the extraction's CPU and wall-clock limits scale with
the video duration (``extraction_budget``), and a run stopped by a limit says so in its error.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from screwit.storage.containment import (
    CPU_LIMIT_REACHED_FRACTION,
    MAX_BUDGETED_MEDIA_S,
    MediaBudget,
    media_budget,
    run_measured,
)

MODEL_LONG_SIDE = 504
GRID_MULTIPLE = 14
TARGET_KEYFRAMES = (48, 64)
BLUR_DROP_FRACTION = 0.30
# Cap the full-res working frames so an oversized phone capture (4K, even 8K) does not blow memory or
# stall the CPU geometry stage. The model grid is 504px anyway, so a 1920px long side loses nothing
# useful downstream. Inputs already smaller than this are left untouched.
FULL_RES_MAX_LONG_SIDE = 1920

# Resource ceilings for every contained ffmpeg invocation (SPEC 3.3): the limits of the probe and
# capability checks, and the floors of the extraction, whose budget scales with the video duration.
FFMPEG_TIMEOUT_S = 300
FFMPEG_CPU_S = 240
FFMPEG_MEM_BYTES = 4 * 1024 * 1024 * 1024
FFMPEG_THREADS = 4  # decoder threads (-threads before -i)
# Tone-map, resize and JPEG-encode one sampled 4K frame: about 14 CPU-s for better.mov's 80 frames.
SAMPLED_FRAME_CPU_S = 0.2
# RLIMIT_NPROC caps the user's TOTAL processes, not just this child's, so it must sit above the
# number already running or ffmpeg cannot spawn its decode threads ("Resource temporarily
# unavailable"). The real containment is the wall-clock timeout, the memory cap and the no-network
# sandbox; this is a generous backstop against a fork bomb.
FFMPEG_NPROC = 4096


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------- containment

def _detect_network_sandbox() -> list[str]:
    """Return an argv prefix that drops network access without privilege, or [] if none works.

    Tries ``unshare -rn`` (user+net namespace) then ``bwrap``. The chosen prefix is recorded in
    frames.json so the report can state the containment actually achieved.
    """
    for probe, prefix in (
        (["unshare", "-rn", "true"], ["unshare", "-rn"]),
        (["bwrap", "--unshare-net", "--dev-bind", "/", "/", "true"],
         ["bwrap", "--unshare-net", "--dev-bind", "/", "/"]),
    ):
        exe = shutil.which(probe[0])
        if not exe:
            continue
        try:
            r = subprocess.run([exe, *probe[1:]], capture_output=True, timeout=10)
            if r.returncode == 0:
                return [exe, *prefix[1:]]
        except (OSError, subprocess.SubprocessError):
            continue
    return []


_SANDBOX_PREFIX: list[str] | None = None


def network_sandbox_prefix() -> list[str]:
    global _SANDBOX_PREFIX
    if _SANDBOX_PREFIX is None:
        _SANDBOX_PREFIX = _detect_network_sandbox()
    return _SANDBOX_PREFIX


def containment_description(cpu_s: int = FFMPEG_CPU_S, timeout_s: int = FFMPEG_TIMEOUT_S) -> str:
    """The containment an ffmpeg run got, for frames.json; pass the extraction's own budget."""
    prefix = network_sandbox_prefix()
    if prefix and prefix[0].endswith("unshare"):
        net = "no-network via `unshare -rn`"
    elif prefix:
        net = "no-network via `bwrap --unshare-net`"
    else:
        net = "no unprivileged no-network sandbox available (ffmpeg is launched without it)"
    return (f"ffmpeg: wall-clock timeout {timeout_s}s, rlimits CPU {cpu_s}s / "
            f"RSS {FFMPEG_MEM_BYTES // (1024 * 1024)}MiB / {FFMPEG_NPROC} procs via preexec_fn; {net}")


def _child_limits(cpu_s: int):
    """preexec_fn applying the rlimits, with RLIMIT_CPU (summed over all threads) set to `cpu_s`."""

    def _apply() -> None:  # pragma: no cover - runs only in the child process
        def _try(which, soft, hard):
            # Best-effort: a platform that rejects or lacks a limit (macOS raises on RLIMIT_AS, where
            # a process reserves far more address space than any sane cap) must not crash the child,
            # which otherwise surfaces as "Exception occurred in preexec_fn" and fails every call.
            try:
                resource.setrlimit(which, (soft, hard))
            except (ValueError, OSError, AttributeError):
                pass

        _try(resource.RLIMIT_CPU, cpu_s, cpu_s)
        _try(resource.RLIMIT_AS, FFMPEG_MEM_BYTES, FFMPEG_MEM_BYTES)
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            # Cap processes as a fork-bomb backstop, but never raise above (or fight) the inherited
            # hard limit, and never below what the host already needs to run ffmpeg's decode threads.
            cap = FFMPEG_NPROC if hard == resource.RLIM_INFINITY else min(FFMPEG_NPROC, hard)
            resource.setrlimit(resource.RLIMIT_NPROC, (cap, hard))
        except (ValueError, OSError):
            pass

    return _apply


# ffmpeg redraws its progress meter ("frame= ... speed=1.8x") with carriage returns; in an error
# tail those lines crowd out the actual error.
_PROGRESS_LINE_RE = re.compile(r"^\s*(frame|size)=")


def _stderr_tail(stderr: bytes, limit: int = 800) -> str:
    lines = [ln for ln in re.split(r"[\r\n]+", stderr.decode("utf-8", "replace"))
             if ln.strip() and not _PROGRESS_LINE_RE.match(ln)]
    return "\n".join(lines)[-limit:]


def _run_ffmpeg(args: list[str], *, allow_fail: bool = False, cpu_s: int = FFMPEG_CPU_S,
                timeout_s: int = FFMPEG_TIMEOUT_S) -> subprocess.CompletedProcess:
    """The single contained ffmpeg entry point used for probing, tone-mapping and extraction.

    ``cpu_s`` is RLIMIT_CPU, summed over all of ffmpeg's threads, and ``timeout_s`` the wall clock;
    extraction passes its duration-scaled ``extraction_budget``. A run stopped by either limit raises
    RuntimeError naming the limit, even with ``allow_fail``: its output was cut short. The CPU limit
    is recognised from the measured CPU time, because ffmpeg catches SIGXCPU and exits 255.
    """
    prefix = network_sandbox_prefix()
    cmd = [*prefix, ffmpeg_exe(), "-nostdin", "-hide_banner", "-threads", str(FFMPEG_THREADS), *args]
    # glibc gives each thread its own malloc arena (64 MiB of address space each); on a many-core
    # host the HEVC/HDR tone-map path then exceeds the RLIMIT_AS cap and the encoder fails with
    # ENOMEM. Two arenas keep the address space inside the cap without changing the output.
    env = {**os.environ, "MALLOC_ARENA_MAX": "2"}
    run = run_measured(cmd, timeout_s=timeout_s, env=env, preexec_fn=_child_limits(cpu_s))
    if run.timed_out:
        raise RuntimeError(f"ffmpeg timed out after {timeout_s}s (its wall-clock limit)")
    if run.returncode != 0:
        if run.cpu_s is not None and run.cpu_s >= CPU_LIMIT_REACHED_FRACTION * cpu_s:
            raise RuntimeError(
                f"ffmpeg ran out of its CPU budget: it used {run.cpu_s:.0f} of its {cpu_s} CPU-seconds "
                f"in {run.wall_s:.0f} s and was stopped (exit {run.returncode}); "
                f"{_stderr_tail(run.stderr, 400)}")
        if run.returncode < 0:
            try:
                name = signal.Signals(-run.returncode).name
            except ValueError:
                name = f"signal {-run.returncode}"
            raise RuntimeError(f"ffmpeg was killed by {name}; {_stderr_tail(run.stderr, 400)}")
        if not allow_fail:
            raise RuntimeError(f"ffmpeg failed ({run.returncode}): {_stderr_tail(run.stderr)}")
    return subprocess.CompletedProcess(cmd, run.returncode, run.stdout, run.stderr)


def extraction_budget(duration_s: float, fps: float) -> MediaBudget:
    """CPU and wall-clock limits for extracting keyframes from a `duration_s` video sampled at `fps`.

    The work is decoding every frame (the validator's software-decode rate, 1.85 CPU-s per second of
    4K HDR HEVC) plus filtering and encoding each sampled frame; ``containment.media_budget`` adds the
    headroom for efficiency cores and safety. Never below FFMPEG_CPU_S / FFMPEG_TIMEOUT_S. better.mov
    (87.2 s) gets 1066 CPU-s and 334 s; it needed 17.7 CPU-s and 10.6 s with VideoToolbox decode,
    156 CPU-s and 40 s decoding in software, and 83 CPU-s and 48 s on the efficiency cores alone.
    """
    budgeted_s = MAX_BUDGETED_MEDIA_S
    if math.isfinite(duration_s):
        budgeted_s = min(max(duration_s, 0.0), MAX_BUDGETED_MEDIA_S)
    sampled = math.ceil(budgeted_s * fps) + 1
    return media_budget(duration_s, threads=FFMPEG_THREADS, fixed_cpu_s=sampled * SAMPLED_FRAME_CPU_S,
                        min_cpu_s=FFMPEG_CPU_S, min_wall_s=FFMPEG_TIMEOUT_S)


@lru_cache(maxsize=1)
def _hwaccel_decode_args() -> tuple[str, ...]:
    """Input-side ffmpeg flags for hardware-accelerated DECODE when the build supports it.

    Env override ``SCREWIT_FFMPEG_HWACCEL``: a method name (e.g. ``videotoolbox``, ``cuda``) to force,
    or ``none``/``0`` to disable. Otherwise auto-detect: on macOS prefer VideoToolbox. Decoded frames
    are downloaded to system memory automatically (no ``-hwaccel_output_format``), so the existing
    software tonemap/scale/fps filters run unchanged; if a clip cannot be hw-decoded ffmpeg falls
    back to software on its own. Returns the flags to place immediately before ``-i``.
    """
    override = os.environ.get("SCREWIT_FFMPEG_HWACCEL", "").strip().lower()
    if override in ("none", "0", "off", "false"):
        return ()
    try:
        proc = _run_ffmpeg(["-hwaccels"], allow_fail=True)
        available = proc.stdout.decode("utf-8", "replace").lower() + proc.stderr.decode("utf-8", "replace").lower()
    except Exception:  # noqa: BLE001 - detection must never break extraction
        return ()
    if override:
        return ("-hwaccel", override) if override in available else ()
    # Auto: VideoToolbox on macOS, otherwise leave decode on the CPU (CUDA needs explicit opt-in via
    # the env override, since NVENC/CUVID device setup is environment-specific).
    if sys.platform == "darwin" and "videotoolbox" in available:
        return ("-hwaccel", "videotoolbox")
    return ()


# --------------------------------------------------------------------------- probe

@dataclass
class Probe:
    duration_s: float
    width: int
    height: int
    codec: str | None
    pix_fmt: str | None
    hdr: bool


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
_STREAM_RE = re.compile(r"Stream.*Video:\s*([a-zA-Z0-9_]+).*?,\s*([a-z0-9]+)(?:\([^)]*\))?,\s*(\d+)x(\d+)")
_ROTATION_RE = re.compile(r"rotation of\s*(-?\d+(?:\.\d+)?)\s*degrees", re.IGNORECASE)


def probe_video(path: Path) -> Probe:
    """Parse ``ffmpeg -i`` stderr for duration, dimensions, codec and pixel format (no ffprobe).

    ``width``/``height`` are the DISPLAY size: phones store portrait video as landscape samples plus
    a display-matrix rotation, and ffmpeg applies that rotation when decoding, so a +-90 degree
    rotation swaps the stored dimensions.
    """
    proc = _run_ffmpeg(["-i", str(path)], allow_fail=True)
    txt = proc.stderr.decode("utf-8", "replace")
    dur = 0.0
    m = _DUR_RE.search(txt)
    if m:
        h, mm, s = m.groups()
        dur = int(h) * 3600 + int(mm) * 60 + float(s)
    sm = _STREAM_RE.search(txt)
    if not sm:
        raise ValueError("no video stream found by ffmpeg probe")
    codec, pix_fmt, w, h = sm.group(1), sm.group(2), int(sm.group(3)), int(sm.group(4))
    hdr = any(tag in txt.lower() for tag in ("bt2020", "smpte2084", "arib-std-b67", "10le", "10be", "p010"))
    hdr = hdr or (pix_fmt is not None and "10" in pix_fmt)
    rm = _ROTATION_RE.search(txt)
    if rm and round(abs(float(rm.group(1)))) % 180 == 90:
        w, h = h, w
    return Probe(duration_s=dur, width=w, height=h, codec=codec, pix_fmt=pix_fmt, hdr=hdr)


# --------------------------------------------------------------------------- grid mapping

def model_grid_size(width: int, height: int) -> tuple[int, int]:
    """Resize so the long side is 504, each side rounded to a multiple of 14 (SPEC 6.1 step 1)."""
    if width >= height:
        long_side, short_side = width, height
    else:
        long_side, short_side = height, width
    scale = MODEL_LONG_SIDE / long_side
    new_long = MODEL_LONG_SIDE
    new_short = int(round(short_side * scale))

    def round14(v: int) -> int:
        r = max(GRID_MULTIPLE, int(round(v / GRID_MULTIPLE)) * GRID_MULTIPLE)
        return r

    new_long = round14(new_long)
    new_short = round14(new_short)
    if width >= height:
        return new_long, new_short
    return new_short, new_long


def axis_mapping(src_w: int, src_h: int, dst_w: int, dst_h: int) -> dict:
    """Per-axis scale from full-res to model-grid pixels (they differ slightly after rounding)."""
    return {
        "src_width": src_w, "src_height": src_h,
        "dst_width": dst_w, "dst_height": dst_h,
        "scale_x": dst_w / src_w, "scale_y": dst_h / src_h,
    }


# --------------------------------------------------------------------------- sampling rate

def sampling_fps(duration_s: float) -> float:
    """Pick fps so that ~ (mid of 48-64)/0.7 raw frames are sampled, leaving 48-64 after the 30% blur
    drop. A 20 s clip lands near 4 fps and a 60 s clip near 1.5 fps (SPEC 6.1 step 1)."""
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    target_keyframes = sum(TARGET_KEYFRAMES) / 2  # 56
    raw_needed = target_keyframes / (1.0 - BLUR_DROP_FRACTION)  # ~80
    fps = raw_needed / duration_s
    return float(min(max(fps, 0.5), 8.0))


# --------------------------------------------------------------------------- blur

def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian (a cheap sharpness score). Pure-numpy 4-neighbour kernel."""
    g = gray.astype(np.float64)
    lap = (-4.0 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    lap = lap[1:-1, 1:-1]
    return float(lap.var())


# --------------------------------------------------------------------------- extraction

@dataclass
class FrameEntry:
    frame: str                 # "0001.jpg"
    sharpness: float
    full_res: str              # "frames/0001.jpg"
    model: str                 # "frames/model/0001.jpg"
    grid: dict                 # axis mapping


@dataclass
class FramesIndex:
    source: str
    fps: float
    total_sampled: int
    kept: int
    containment: str
    frames: list[FrameEntry]

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _tonemap_filter(hdr: bool) -> str | None:
    if not hdr:
        return None
    # zscale to linear, tonemap, back to bt709 8-bit (SPEC 6.1 step 1).
    return ("zscale=t=linear:npl=100,format=gbrpf32le,"
            "tonemap=tonemap=hable:desat=0,"
            "zscale=p=bt709:t=bt709:m=bt709:r=tv,format=yuv420p")


def _downscale_filter() -> str:
    # Downscale oversized captures so the long side is at most FULL_RES_MAX_LONG_SIDE, preserving
    # aspect and keeping both sides even. Smaller inputs pass through unchanged (the min()/max()
    # make the scale a no-op when the clip is already within the cap).
    cap = FULL_RES_MAX_LONG_SIDE
    return (f"scale='if(gte(iw,ih),min(iw,{cap}),-2)':'if(gte(iw,ih),-2,min(ih,{cap}))':"
            f"flags=lanczos")


def extract_video_frames(video_path: Path, out_dir: Path, *, duration_s: float | None = None) -> FramesIndex:
    """Extract keyframes from a video into ``out_dir`` (full-res under ``frames/`` and model-grid
    under ``frames/model/``), drop the blurriest 30%, and write ``frames.json``."""
    from PIL import Image

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    full_dir = out_dir
    model_dir = out_dir / "model"
    full_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_video(video_path)
    dur = duration_s if duration_s is not None else probe.duration_s
    fps = sampling_fps(dur)

    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    # Sample first: fps keeps ~80 of the clip's frames (80 of 2617 for better.mov), so the tone-map
    # and resize run on those and not on every decoded frame. It picks frames by timestamp alone,
    # so the frames are identical to filtering first (byte-for-byte on better.mov) at 1/20 of the CPU
    # time (17.7 instead of 367 CPU-s), which a 90 s 4K HDR clip otherwise needs and runs out of.
    chain = [f"fps={fps:.4f}"]
    tone = _tonemap_filter(probe.hdr)
    if tone:
        chain.append(tone)
    chain.append(_downscale_filter())
    vf = ",".join(chain)
    budget = extraction_budget(dur, fps)
    _run_ffmpeg([*_hwaccel_decode_args(), "-i", str(video_path), "-vf", vf, "-q:v", "2",
                 str(raw_dir / "raw_%05d.jpg")], cpu_s=budget.cpu_s, timeout_s=budget.wall_timeout_s)

    raw_frames = sorted(raw_dir.glob("raw_*.jpg"))
    if not raw_frames:
        raise RuntimeError("ffmpeg produced no frames")

    scored = []
    for p in raw_frames:
        img = np.asarray(Image.open(p).convert("RGB"))
        gray = img.mean(axis=2)
        scored.append((laplacian_variance(gray), p))

    scored.sort(key=lambda t: t[0], reverse=True)
    keep_n = max(1, int(round(len(scored) * (1.0 - BLUR_DROP_FRACTION))))
    # Cap at the upper target to avoid overloading the model.
    keep_n = min(keep_n, TARGET_KEYFRAMES[1])
    kept = sorted(scored[:keep_n], key=lambda t: t[1].name)  # chronological

    dec_w, dec_h = Image.open(raw_frames[0]).size  # what ffmpeg actually decoded (rotation + cap applied)
    # The extraction may downscale an oversized capture, so decoded size can differ from the probe.
    # What must still hold is the aspect ratio (a mismatch there means a wrong rotation/resample).
    probe_ar = probe.width / probe.height if probe.height else 0.0
    dec_ar = dec_w / dec_h if dec_h else 0.0
    if probe_ar and abs(dec_ar - probe_ar) > 0.02 * probe_ar:
        raise RuntimeError(f"decoded frames are {dec_w}x{dec_h} (aspect {dec_ar:.3f}) but the probe "
                           f"reported {probe.width}x{probe.height} (aspect {probe_ar:.3f}); "
                           f"refusing to resample with a wrong aspect")
    dst_w, dst_h = model_grid_size(dec_w, dec_h)
    mapping = axis_mapping(dec_w, dec_h, dst_w, dst_h)

    entries: list[FrameEntry] = []
    for i, (sharp, src) in enumerate(kept, start=1):
        name = f"{i:04d}.jpg"
        img = np.asarray(Image.open(src).convert("RGB"))
        Image.fromarray(img).save(full_dir / name, quality=95)
        model_img = _resize(img, dst_w, dst_h)
        Image.fromarray(model_img).save(model_dir / name, quality=95)
        entries.append(FrameEntry(frame=name, sharpness=sharp,
                                  full_res=f"frames/{name}", model=f"frames/model/{name}",
                                  grid=mapping))

    shutil.rmtree(raw_dir, ignore_errors=True)

    index = FramesIndex(source=video_path.name, fps=fps, total_sampled=len(raw_frames),
                        kept=len(entries),
                        containment=containment_description(budget.cpu_s, budget.wall_timeout_s),
                        frames=entries)
    (out_dir / "frames.json").write_text(json.dumps(index.to_json(), indent=2))
    return index


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image

    mode = "RGB" if img.ndim == 3 else "L"
    pil = Image.fromarray(img, mode=mode)
    return np.asarray(pil.resize((w, h), Image.LANCZOS))


# --------------------------------------------------------------------------- photos

def _exif_normalise(img_path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def extract_photo_frames(photo_paths: list[Path], out_dir: Path) -> FramesIndex:
    """Normalise EXIF orientation and resize every photo to ONE common size (the model stacks them
    into one tensor), then write the full-res and model-grid copies and frames.json."""
    from PIL import Image

    out_dir = Path(out_dir)
    full_dir = out_dir
    model_dir = out_dir / "model"
    full_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    imgs = [_exif_normalise(Path(p)) for p in photo_paths]
    if not imgs:
        raise ValueError("no photos provided")
    # One common full-res size: the median dimensions, so every photo stacks.
    common_w = int(np.median([im.shape[1] for im in imgs]))
    common_h = int(np.median([im.shape[0] for im in imgs]))
    dst_w, dst_h = model_grid_size(common_w, common_h)
    mapping = axis_mapping(common_w, common_h, dst_w, dst_h)

    entries: list[FrameEntry] = []
    for i, im in enumerate(imgs, start=1):
        name = f"{i:04d}.jpg"
        full = _resize(im, common_w, common_h)
        Image.fromarray(full).save(full_dir / name, quality=95)
        model_img = _resize(im, dst_w, dst_h)
        Image.fromarray(model_img).save(model_dir / name, quality=95)
        gray = full.mean(axis=2)
        entries.append(FrameEntry(frame=name, sharpness=laplacian_variance(gray),
                                  full_res=f"frames/{name}", model=f"frames/model/{name}",
                                  grid=mapping))

    index = FramesIndex(source="photos", fps=0.0, total_sampled=len(imgs), kept=len(entries),
                        containment=containment_description(), frames=entries)
    (out_dir / "frames.json").write_text(json.dumps(index.to_json(), indent=2))
    return index
