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
