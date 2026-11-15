"""Frame extraction: grid mapping, sampling rate, blur scoring, and a real synthetic-video run
through the single contained ffmpeg helper."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from screwit.recon import frames as F


def test_model_grid_long_side_and_multiple_of_14():
    w, h = F.model_grid_size(1920, 1080)
    assert max(w, h) == 504
    assert w % 14 == 0 and h % 14 == 0
    # Portrait input: long side still 504.
    w2, h2 = F.model_grid_size(1080, 1920)
    assert max(w2, h2) == 504
    assert w2 % 14 == 0 and h2 % 14 == 0


def test_axis_mapping_per_axis_scale():
    m = F.axis_mapping(1920, 1080, 504, 280)
    assert m["scale_x"] == pytest.approx(504 / 1920)
    assert m["scale_y"] == pytest.approx(280 / 1080)
    assert m["scale_x"] != m["scale_y"]  # per-axis differs after rounding


def test_sampling_fps_targets_keyframe_band():
    # ~56 keyframes target / 0.7 survive => raw count; fps = raw / duration.
    assert F.sampling_fps(20.0) == pytest.approx(80 / 20, rel=0.2)
    assert F.sampling_fps(60.0) == pytest.approx(80 / 60, rel=0.2)
    assert 0.5 <= F.sampling_fps(5.0) <= 8.0


def test_laplacian_variance_sharper_is_higher():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(64, 64)).astype(float)   # high-frequency noise
    blur = np.full((64, 64), 128.0)                              # flat
    assert F.laplacian_variance(sharp) > F.laplacian_variance(blur)


def _ffmpeg_ok() -> bool:
    try:
        exe = F.ffmpeg_exe()
        subprocess.run([exe, "-version"], capture_output=True, timeout=20)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ffmpeg_ok(), reason="imageio-ffmpeg binary unavailable")
def test_extract_video_frames_end_to_end(tmp_path):
    """Generate a short synthetic video with the ffmpeg binary, then extract keyframes and confirm
    frames.json, full-res and model-grid copies, and the recorded containment."""
    exe = F.ffmpeg_exe()
    video = tmp_path / "clip.mp4"
    # 6 s testsrc at 1280x720, 24 fps; a real decodable H.264 clip.
    subprocess.run([exe, "-nostdin", "-hide_banner", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=1280x720:rate=24:duration=6",
                    "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True, timeout=120)

    out = tmp_path / "frames"
    index = F.extract_video_frames(video, out, duration_s=6.0)
    assert index.kept >= 1
    assert index.total_sampled >= index.kept  # some dropped as blurriest 30%
    # Files exist in both resolutions.
    for e in index.frames:
        assert (out / e.frame).exists()
        assert (out / "model" / e.frame).exists()
    assert (out / "frames.json").exists()
    # Model-grid obeys the 504 / multiple-of-14 rule.
    e0 = index.frames[0]
    assert e0.grid["dst_width"] % 14 == 0 and e0.grid["dst_height"] % 14 == 0
    assert max(e0.grid["dst_width"], e0.grid["dst_height"]) == 504
    assert "ffmpeg" in index.containment


def test_containment_description_mentions_limits():
    d = F.containment_description()
    assert "timeout" in d and "rlimits" in d
    # Either a sandbox was found or it is explicitly recorded as unavailable.
    assert ("no-network" in d) or ("no unprivileged no-network sandbox" in d)


_MEDIA = __import__("pathlib").Path(__file__).resolve().parents[3] / "fixtures" / "media"


@pytest.mark.parametrize("name,portrait", [("android_h264.mp4", False),
                                           ("iphone_hevc_hlg_portrait.mov", True)])
def test_phone_media_extracts_with_display_orientation(tmp_path, name, portrait):
    """X2's phone clips: the HDR HEVC tone-map path fits the contained ffmpeg's address-space cap,
    and a portrait clip (landscape samples + 90 degree display matrix) yields portrait frames on
    the portrait model grid instead of being squashed to landscape."""
    from PIL import Image

    src = _MEDIA / name
    if not src.exists():
        pytest.skip(f"{src} not generated")
    probe = F.probe_video(src)
    assert (probe.height > probe.width) is portrait
    index = F.extract_video_frames(src, tmp_path)
    assert 48 <= index.kept <= 64
    first = index.frames[0].frame
    full = Image.open(tmp_path / first).size
    model = Image.open(tmp_path / "model" / first).size
    assert (full[1] > full[0]) is portrait
    assert model == ((280, 504) if portrait else (504, 280))


# -- extraction limits (the better.mov bug) ------------------------------------------------------

def test_extraction_budget_scales_with_duration_and_keeps_the_floors():
    short = F.extraction_budget(6.0, F.sampling_fps(6.0))
    assert (short.cpu_s, short.wall_timeout_s) == (F.FFMPEG_CPU_S, F.FFMPEG_TIMEOUT_S)
    # better.mov (87.2 s, 4K HDR HEVC): the flat 240 CPU-s ran out 62 s in. Extraction now needs
    # 17.7 CPU-s (VideoToolbox decode) or 156 CPU-s (software decode); it gets several times that.
    b = F.extraction_budget(87.23, F.sampling_fps(87.23))
    assert b.cpu_s >= 6 * 156 and b.wall_timeout_s >= 300
    assert F.extraction_budget(60.0, F.sampling_fps(60.0)).cpu_s < b.cpu_s
    # A bogus duration cannot buy more than the longest accepted (90 s) video gets.
    assert F.extraction_budget(1e9, 0.5).cpu_s <= F.extraction_budget(90.0, F.sampling_fps(90.0)).cpu_s


@pytest.mark.skipif(not _ffmpeg_ok(), reason="imageio-ffmpeg binary unavailable")
def test_running_out_of_the_cpu_budget_is_reported_as_such():
    # mandelbrot is CPU-bound; 1 CPU-second stops it long before its 60 s of output.
    with pytest.raises(RuntimeError, match=r"^ffmpeg ran out of its CPU budget: it used \d+ of its 1 CPU"):
        F._run_ffmpeg(["-f", "lavfi", "-i", "mandelbrot=size=1280x720:rate=25", "-t", "60",
                       "-f", "null", "-"], cpu_s=1, timeout_s=60)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="imageio-ffmpeg binary unavailable")
def test_wall_clock_timeout_is_reported_as_such():
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        F._run_ffmpeg(["-re", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10", "-t", "30",
                       "-f", "null", "-"], timeout_s=1)


@pytest.mark.skipif(not _ffmpeg_ok(), reason="imageio-ffmpeg binary unavailable")
def test_extraction_samples_first_and_runs_under_its_budget(tmp_path, monkeypatch):
    exe = F.ffmpeg_exe()
    video = tmp_path / "clip.mp4"
    subprocess.run([exe, "-nostdin", "-hide_banner", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=640x360:rate=24:duration=12",
                    "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True, timeout=120)
    calls = []
    real = F._run_ffmpeg

    def spy(args, **kw):
        calls.append((args, kw))
        return real(args, **kw)

    monkeypatch.setattr(F, "_run_ffmpeg", spy)
    index = F.extract_video_frames(video, tmp_path / "frames", duration_s=12.0)
    [(args, kw)] = [c for c in calls if "-vf" in c[0]]
    assert args[args.index("-vf") + 1].startswith("fps=")
    budget = F.extraction_budget(12.0, F.sampling_fps(12.0))
    assert (kw["cpu_s"], kw["timeout_s"]) == (budget.cpu_s, budget.wall_timeout_s)
    assert f"rlimits CPU {budget.cpu_s}s" in index.containment


@pytest.mark.parametrize("name", ["iphone_hevc_hlg_portrait.mov", "android_h264.mp4"])
def test_sampling_before_filtering_gives_byte_identical_frames(tmp_path, name):
    """The fps filter picks frames by timestamp alone, so moving it ahead of the tone-map and resize
    (which only saves CPU) must not change a single output byte, for HDR and SDR, portrait and landscape."""
    src = _MEDIA / name
    if not src.exists():
        pytest.skip(f"{src} not generated")
    probe = F.probe_video(src)
    fps = f"fps={F.sampling_fps(probe.duration_s):.4f}"
    filters = [f for f in (F._tonemap_filter(probe.hdr), F._downscale_filter()) if f]
    outputs = {}
    for order, chain in (("filter_first", [*filters, fps]), ("sample_first", [fps, *filters])):
        out = tmp_path / order
        out.mkdir()
        F._run_ffmpeg([*F._hwaccel_decode_args(), "-i", str(src), "-vf", ",".join(chain), "-q:v", "2",
                       str(out / "raw_%05d.jpg")])
        outputs[order] = [p.read_bytes() for p in sorted(out.glob("raw_*.jpg"))]
    assert len(outputs["sample_first"]) >= 48
    assert outputs["sample_first"] == outputs["filter_first"]
