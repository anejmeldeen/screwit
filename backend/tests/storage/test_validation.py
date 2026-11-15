"""MediaValidator: contained probing of generated good/bad media (SPEC 3.3).

Media files are generated with the imageio-ffmpeg binary (no system ffmpeg, AGENTS.md) and with
Pillow for photos. Each file is written into a LocalStorage dev store and validated through the real
containment path.
"""

from __future__ import annotations

import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import imageio_ffmpeg
import pytest

from screwit.contracts import Scan, ScanInput, Stage
from screwit.contracts.models import STAGE_ORDER
from screwit.storage import validation as V
from screwit.storage.containment import Containment, Limits, MediaBudget, RunResult, docker_available, media_budget
from screwit.storage.local import LocalStorage
from screwit.storage.validation import MediaValidator

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA = Path(__file__).resolve().parents[3] / "fixtures" / "media"


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


def test_validation_leaves_no_scratch_copy_behind(tmp_path, store, validator, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    key = "scans/scn_clean/input/clip.mp4"
    f = tmp_path / "clip.mp4"
    _make_video(f, 12)
    store.put_file(key, f, "video/mp4")
    validator.validate(_scan("video", key), store)
    assert list((tmp_path / "tmp").iterdir()) == []


# -- the decode check under containment limits (the better.mov bug) -------------

_RL = Containment(network_isolated=False, mechanism="rlimits-only", rlimits=True)
_OVER_BUDGET = "over"  # the fake reports CPU time just past the limit it was given


class _FakeDecoder:
    """Stands in for run_contained in the decode check: records each run's argv and limits, writes the
    progress report ffmpeg would have written, and returns the scripted outcome."""

    def __init__(self, *outcomes: dict) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[list[str], Limits]] = []

    def __call__(self, argv, scratch, *, limits=None, **_):
        o = self.outcomes.pop(0)
        self.calls.append((argv, limits))
        readable = o.get("readable_s")
        if readable is not None:
            prog = Path(argv[argv.index("-progress") + 1])
            prog.write_text(f"out_time_us={int(readable * 1e6)}\nprogress=end\n")
        cpu = o.get("cpu_s", 1.0)
        if cpu == _OVER_BUDGET:
            cpu = limits.cpu_s + 4.0
        return RunResult(o.get("rc", 0), b"", o.get("stderr", b""), _RL, cpu_s=cpu, wall_s=1.0,
                         timed_out=o.get("timed_out", False), limits=limits)


def _decode_check(monkeypatch, tmp_path, fake, *, declared_s=87.23, hwaccel=()):
    monkeypatch.setattr(V, "run_contained", fake)
    monkeypatch.setattr(V, "_decode_hwaccel_args", lambda: hwaccel)
    MediaValidator(ffmpeg_exe=FFMPEG)._decode_check(tmp_path / "upload.mov", tmp_path, declared_s=declared_s)


def test_decode_check_cpu_limit_exit_255_is_not_a_file_verdict(tmp_path, monkeypatch):
    # The better.mov failure on macOS: SIGXCPU, which ffmpeg turns into exit 255 with an empty stderr
    # at -v error, and a progress report that stops part-way (73.8 of 87.2 s).
    fake = _FakeDecoder({"rc": 255, "readable_s": 73.8, "cpu_s": _OVER_BUDGET})
    with pytest.raises(RuntimeError, match="CPU budget.*not a file verdict"):
        _decode_check(monkeypatch, tmp_path, fake)


@pytest.mark.parametrize("outcome,reason", [
    ({"rc": 255, "readable_s": 14.8, "cpu_s": None}, "code 255"),          # docker: no CPU time
    ({"rc": -9, "readable_s": 30.0, "cpu_s": _OVER_BUDGET}, "CPU budget"),  # Linux hard limit: SIGKILL
    ({"rc": -9, "readable_s": 30.0, "cpu_s": 2.0}, "SIGKILL"),              # killed for another reason
    ({"rc": 124, "readable_s": 50.0, "timed_out": True}, "wall-clock"),
])
def test_decode_check_stopped_runs_are_not_file_verdicts(tmp_path, monkeypatch, outcome, reason):
    with pytest.raises(RuntimeError, match=f"{reason}.*not a file verdict"):
        _decode_check(monkeypatch, tmp_path, _FakeDecoder(outcome))


def test_decode_check_short_read_that_finished_is_still_truncated(tmp_path, monkeypatch):
    fake = _FakeDecoder({"rc": 0, "readable_s": 14.8})
    with pytest.raises(ValueError, match="only 14.8 s of the declared 87.2 s"):
        _decode_check(monkeypatch, tmp_path, fake)


def test_decode_check_limits_scale_with_the_declared_duration(tmp_path, monkeypatch):
    budgets = {}
    for declared in (12.0, 87.23):
        fake = _FakeDecoder({"rc": 0, "readable_s": declared})
        _decode_check(monkeypatch, tmp_path, fake, declared_s=declared)
        limits = fake.calls[0][1]
        expected = media_budget(declared, threads=V.DECODE_THREADS, min_cpu_s=120, min_wall_s=180)
        assert (limits.cpu_s, limits.wall_timeout_s) == (expected.cpu_s, expected.wall_timeout_s)
        budgets[declared] = limits
    # A 12 s clip keeps the old floor; better.mov gets about 6x the 161 CPU-s it was measured to need.
    assert budgets[12.0].cpu_s >= 120 and budgets[12.0].wall_timeout_s == 180
    assert budgets[87.23].cpu_s >= 6 * 161 and budgets[87.23].wall_timeout_s >= 300


_VT = ("-hwaccel", "videotoolbox")


def test_spotless_hardware_decode_is_accepted_without_a_software_pass(tmp_path, monkeypatch):
    fake = _FakeDecoder({"rc": 0, "readable_s": 87.23, "cpu_s": 4.0})
    _decode_check(monkeypatch, tmp_path, fake, hwaccel=_VT)
    assert len(fake.calls) == 1 and "-hwaccel" in fake.calls[0][0]


@pytest.mark.parametrize("hw_outcome", [
    {"rc": 0, "readable_s": 87.23, "stderr": b"[hevc] hardware accelerator failed to decode picture\n"},
    {"rc": 0, "readable_s": 40.0},
    {"rc": 255, "readable_s": 10.0, "cpu_s": _OVER_BUDGET},
    {"rc": 69, "readable_s": 2.3, "stderr": b"[hevc] Invalid NAL unit size\n"},
])
def test_any_flawed_hardware_decode_is_decided_by_software(tmp_path, monkeypatch, hw_outcome):
    fake = _FakeDecoder(hw_outcome, {"rc": 0, "readable_s": 87.23})
    _decode_check(monkeypatch, tmp_path, fake, hwaccel=_VT)  # the clean software pass accepts
    assert len(fake.calls) == 2 and "-hwaccel" not in fake.calls[1][0]

    fake = _FakeDecoder(hw_outcome, {"rc": 0, "readable_s": 40.0})
    with pytest.raises(ValueError, match="truncated"):  # and the software verdict is the verdict
        _decode_check(monkeypatch, tmp_path, fake, hwaccel=_VT)


_IPHONE_HDR = MEDIA / "iphone_hevc_hlg_portrait.mov"


@pytest.mark.skipif(docker_available(), reason="RLIMIT_CPU applies to the rlimits runner")
@pytest.mark.skipif(not _IPHONE_HDR.exists(), reason="fixtures/media not generated")
def test_real_decode_cut_short_by_its_cpu_limit_is_not_a_file_verdict(store, monkeypatch):
    """The original bug with a real ffmpeg and a real RLIMIT_CPU: a valid HDR HEVC phone clip (5.5 CPU-s
    to decode in software) given 1 CPU-second must fail as "not a file verdict", not "truncated"."""
    monkeypatch.setenv("SCREWIT_FFMPEG_HWACCEL", "none")  # software decode is what spends CPU
    monkeypatch.setattr(V, "media_budget", lambda *a, **k: MediaBudget(cpu_s=1, wall_timeout_s=60))
    key = "scans/scn_cpu/input/clip.mov"
    store.put_file(key, _IPHONE_HDR, "video/quicktime")
    validator = MediaValidator(ffmpeg_exe=FFMPEG, limits=Limits(cpu_s=1, wall_timeout_s=60))
    with pytest.raises(RuntimeError, match="CPU budget.*not a file verdict"):
        validator.validate(_scan("video", key), store)


def _hevc_clip(path: Path) -> Path:
    _make_video(path, 12, codec="libx265", fmt="mov",
                extra=["-tag:v", "hvc1", "-x265-params", "log-level=error", "-movflags", "+faststart"])
    return path


def _has_encoder(name: str) -> bool:
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {name} " in out


def _truncate(src: Path, dst: Path) -> Path:
    data = src.read_bytes()
    dst.write_bytes(data[: len(data) // 2])
    return dst


def _corrupt(src: Path, dst: Path) -> Path:
    data = bytearray(src.read_bytes())
    rnd = random.Random(1)
    for frac in (0.35, 0.6):
        off = int(len(data) * frac)
        data[off:off + 4096] = bytes(rnd.randrange(256) for _ in range(4096))
    dst.write_bytes(bytes(data))
    return dst


@pytest.mark.skipif(not _has_encoder("libx265"), reason="this ffmpeg build has no HEVC encoder")
@pytest.mark.parametrize("hwaccel_env", ["", "none"])  # default (VideoToolbox on macOS) and software
@pytest.mark.parametrize("damage", [_truncate, _corrupt])
def test_damaged_hevc_gets_the_same_verdict_with_or_without_hardware_decode(
        tmp_path, store, validator, monkeypatch, hwaccel_env, damage):
    monkeypatch.setenv("SCREWIT_FFMPEG_HWACCEL", hwaccel_env)
    good = _hevc_clip(tmp_path / "good.mov")
    key_ok, key_bad = "scans/scn_hg/input/good.mov", "scans/scn_hb/input/bad.mov"
    store.put_file(key_ok, good, "video/quicktime")
    store.put_file(key_bad, damage(good, tmp_path / "bad.mov"), "video/quicktime")
    assert validator.validate(_scan("video", key_ok), store).codec == "hevc"
    with pytest.raises(ValueError):
        validator.validate(_scan("video", key_bad), store)
