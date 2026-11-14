"""Frame extraction (SPEC 6.1 step 1).

Probe the input with ffmpeg (no ffprobe; parse ``ffmpeg -i`` stderr), pick a sampling rate from the
duration so that 48-64 keyframes survive after dropping the blurriest 30% (variance of Laplacian),
tone-map 10-bit HLG/PQ to 8-bit, and keep two copies of each keyframe: full resolution for board
detection and the agent's close-ups, and the model-grid copy (long side 504, each side rounded to a
multiple of 14, with the exact per-axis mapping recorded in frames.json).

All ffmpeg calls use ONE helper (``_run_ffmpeg``) that applies a wall-clock timeout, CPU/memory/pids
rlimits, and a no-network sandbox when one is available unprivileged (SPEC 3.3 containment).
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MODEL_LONG_SIDE = 504
GRID_MULTIPLE = 14
TARGET_KEYFRAMES = (48, 64)
BLUR_DROP_FRACTION = 0.30

# Resource ceilings for every contained ffmpeg invocation (SPEC 3.3).
FFMPEG_TIMEOUT_S = 300
FFMPEG_CPU_S = 240
FFMPEG_MEM_BYTES = 4 * 1024 * 1024 * 1024
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


def containment_description() -> str:
    prefix = network_sandbox_prefix()
    if prefix and prefix[0].endswith("unshare"):
        net = "no-network via `unshare -rn`"
    elif prefix:
        net = "no-network via `bwrap --unshare-net`"
    else:
        net = "no unprivileged no-network sandbox available (ffmpeg is launched without it)"
    return (f"ffmpeg: wall-clock timeout {FFMPEG_TIMEOUT_S}s, rlimits CPU {FFMPEG_CPU_S}s / "
            f"RSS {FFMPEG_MEM_BYTES // (1024 * 1024)}MiB / {FFMPEG_NPROC} procs via preexec_fn; {net}")


def _rlimits() -> None:  # pragma: no cover - runs only in the child process
    resource.setrlimit(resource.RLIMIT_CPU, (FFMPEG_CPU_S, FFMPEG_CPU_S))
    resource.setrlimit(resource.RLIMIT_AS, (FFMPEG_MEM_BYTES, FFMPEG_MEM_BYTES))
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        # Cap processes as a fork-bomb backstop, but never raise above (or fight) the inherited
        # hard limit, and never below what the host already needs to run ffmpeg's decode threads.
        cap = FFMPEG_NPROC if hard == resource.RLIM_INFINITY else min(FFMPEG_NPROC, hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (cap, hard))
    except (ValueError, OSError):
        pass


def _run_ffmpeg(args: list[str], *, allow_fail: bool = False) -> subprocess.CompletedProcess:
    """The single contained ffmpeg entry point used for probing, tone-mapping and extraction."""
    prefix = network_sandbox_prefix()
    cmd = [*prefix, ffmpeg_exe(), "-nostdin", "-hide_banner", "-threads", "4", *args]
    # glibc gives each thread its own malloc arena (64 MiB of address space each); on a many-core
    # host the HEVC/HDR tone-map path then exceeds the RLIMIT_AS cap and the encoder fails with
    # ENOMEM. Two arenas keep the address space inside the cap without changing the output.
    env = {**os.environ, "MALLOC_ARENA_MAX": "2"}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_S,
            preexec_fn=_rlimits,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s") from e
    if proc.returncode != 0 and not allow_fail:
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {tail}")
    return proc


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
    vf = f"fps={fps:.4f}"
    tone = _tonemap_filter(probe.hdr)
    if tone:
        vf = f"{tone},{vf}"
    _run_ffmpeg(["-i", str(video_path), "-vf", vf, "-q:v", "2",
                 str(raw_dir / "raw_%05d.jpg")])

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

    dec_w, dec_h = Image.open(raw_frames[0]).size  # what ffmpeg actually decoded (rotation applied)
    if (dec_w, dec_h) != (probe.width, probe.height):
        raise RuntimeError(f"decoded frames are {dec_w}x{dec_h} but the probe reported "
                           f"{probe.width}x{probe.height}; refusing to resample with a wrong aspect")
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
                        kept=len(entries), containment=containment_description(), frames=entries)
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
