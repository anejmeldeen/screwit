"""MediaValidator: contained probing of generated good/bad media (SPEC 3.3).

Media files are generated with the imageio-ffmpeg binary (no system ffmpeg, AGENTS.md) and with
Pillow for photos. Each file is written into a LocalStorage dev store and validated through the real
containment path.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import imageio_ffmpeg
import pytest

from screwit.contracts import Scan, ScanInput, Stage
from screwit.contracts.models import STAGE_ORDER
from screwit.storage.local import LocalStorage
from screwit.storage.validation import MediaValidator

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _make_video(path: Path, seconds: float, codec: str = "libx264", fmt: str = "mp4",
                extra: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={seconds}",
        "-c:v", codec, "-pix_fmt", "yuv420p", *(extra or []), "-f", fmt, str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _scan(kind: str, key: str) -> Scan:
    stages = [Stage(name=n, state=("skipped" if kind == "mesh" and n in
                                   ("frames", "geometry", "object_mask", "splat") else "pending"))
              for n in STAGE_ORDER]
    return Scan(scan_id=key.split("/")[1], created_at=datetime.now(UTC),
                input=ScanInput(type=kind, uri=key), status="validating",
                reconstruction_revision=0, stages=stages)


@pytest.fixture
def store(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path / "data", signing_key=b"k" * 32)


@pytest.fixture
def validator() -> MediaValidator:
    return MediaValidator(ffmpeg_exe=FFMPEG)


# -- good media -----------------------------------------------------------

def test_valid_h264_mp4(tmp_path, store, validator):
    key = "scans/scn_v/input/clip.mp4"
    f = tmp_path / "clip.mp4"
    _make_video(f, 12, codec="libx264", fmt="mp4")
    store.put_file(key, f, "video/mp4")
    info = validator.validate(_scan("video", key), store)
    assert info.kind == "video"
    assert info.codec == "h264"
    assert 11.5 <= info.duration_s <= 12.5
    assert info.width == 320 and info.height == 240
    # Containment was attempted; the mechanism is recorded honestly.
    assert validator.last_containment is not None
    assert validator.last_containment.rlimits is True


def test_valid_photo(tmp_path, store, validator):
    from PIL import Image

    key = "scans/scn_p/input/photo.jpg"
    f = tmp_path / "photo.jpg"
    Image.new("RGB", (800, 600), (120, 130, 140)).save(f, "JPEG")
    store.put_file(key, f, "image/jpeg")
    info = validator.validate(_scan("photos", key), store)
    assert info.kind == "photos"
    assert info.width == 800 and info.height == 600


# -- bad media ------------------------------------------------------------

def test_too_short_video_rejected(tmp_path, store, validator):
    key = "scans/scn_s/input/short.mp4"
    f = tmp_path / "short.mp4"
    _make_video(f, 3, codec="libx264")
    store.put_file(key, f, "video/mp4")
    with pytest.raises(ValueError, match="accepted range"):
        validator.validate(_scan("video", key), store)


def test_too_long_video_rejected(tmp_path, store, validator):
    key = "scans/scn_l/input/long.mp4"
    f = tmp_path / "long.mp4"
    _make_video(f, 95, codec="libx264")
    store.put_file(key, f, "video/mp4")
    with pytest.raises(ValueError, match="accepted range"):
        validator.validate(_scan("video", key), store)


def test_wrong_codec_rejected(tmp_path, store, validator):
    # mpeg4 (not H.264/HEVC/VP8/VP9) in an mp4 container.
    key = "scans/scn_c/input/wrong.mp4"
    f = tmp_path / "wrong.mp4"
    _make_video(f, 12, codec="mpeg4", fmt="mp4")
    store.put_file(key, f, "video/mp4")
    with pytest.raises(ValueError, match="codec"):
        validator.validate(_scan("video", key), store)


def test_non_video_bytes_rejected(tmp_path, store, validator):
    key = "scans/scn_n/input/notvideo.mp4"
    store.put_bytes(key, b"this is definitely not a video file" * 10, "video/mp4")
    with pytest.raises(ValueError, match="supported video"):
        validator.validate(_scan("video", key), store)


def test_truncated_video_rejected(tmp_path, store, validator):
    key = "scans/scn_t/input/trunc.mp4"
    f = tmp_path / "ok.mp4"
    _make_video(f, 12, codec="libx264")
    data = f.read_bytes()
    # Keep the ftyp header so magic passes, but truncate the stream so duration/stream parsing fails.
    store.put_bytes(key, data[:200], "video/mp4")
    with pytest.raises(ValueError):
        validator.validate(_scan("video", key), store)


def test_truncated_photo_rejected(tmp_path, store, validator):
    from PIL import Image

    key = "scans/scn_tp/input/bad.jpg"
    f = tmp_path / "ok.jpg"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(f, "JPEG")
    data = f.read_bytes()
    store.put_bytes(key, data[:len(data) // 3], "image/jpeg")  # keeps JPEG magic, breaks the body
    with pytest.raises(ValueError):
        validator.validate(_scan("photos", key), store)


def test_non_image_bytes_rejected(tmp_path, store, validator):
    key = "scans/scn_ni/input/x.jpg"
    store.put_bytes(key, b"nope not an image", "image/jpeg")
    with pytest.raises(ValueError, match="supported image"):
        validator.validate(_scan("photos", key), store)


def test_mesh_glb_accepted(tmp_path, store, validator):
    key = "scans/scn_m/input/model.glb"
    store.put_bytes(key, b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 16, "model/gltf-binary")
    info = validator.validate(_scan("mesh", key), store)
    assert info.kind == "mesh" and info.container == "glb"


def test_non_glb_mesh_rejected(tmp_path, store, validator):
    key = "scans/scn_mo/input/model.glb"
    store.put_bytes(key, b"OBJ file not glb", "model/gltf-binary")
    with pytest.raises(ValueError, match="glTF"):
        validator.validate(_scan("mesh", key), store)
