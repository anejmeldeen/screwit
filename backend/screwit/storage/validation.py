"""Worker-side upload validation job (SPEC 3.3, P5 workload 1).

Reads the uploaded object under containment and decides whether it is acceptable media, raising
ValueError with a user-facing message otherwise. ffmpeg is the imageio-ffmpeg binary parsed from
`ffmpeg -i` (no ffprobe, common.md); photos are read with Pillow headers/EXIF in the same
containment. The media file is always copied into the per-job directory first, because in docker
containment that directory is the only mount the tool can see.

Beyond the header probe, every video is partly DECODED inside the containment: a header can declare a
25 s H.264 stream while the bitstream is truncated, so the probe alone accepts a broken file. The
validator decodes the stream to null and rejects on decoder errors or when the readable duration
falls far short of the declared one.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from screwit.interfaces import MediaInfo, Storage
from screwit.storage.containment import Containment, Limits, run_contained

# Magic bytes we accept for the container families in SPEC 2.2 / 6.1.
_VIDEO_MAGIC = (
    (4, b"ftyp"),      # ISO-BMFF: MP4 / MOV (box type at offset 4)
    (0, b"\x1aE\xdf\xa3"),  # EBML: WebM / Matroska
)
_PHOTO_MAGIC = (
    (0, b"\xff\xd8\xff"),            # JPEG
    (0, b"\x89PNG\r\n\x1a\n"),       # PNG
    (0, b"RIFF"),                    # WebP (RIFF....WEBP)
)

_ACCEPTED_VIDEO_CODECS = {"h264", "hevc", "vp8", "vp9"}
_MIN_DURATION_S = 10.0
_MAX_DURATION_S = 90.0
_MAX_DIM = 7680  # 8K ceiling
_MIN_DIM = 64
# A readable duration shorter than this fraction of the declared duration means the stream is cut off.
_READABLE_DURATION_MIN_FRACTION = 0.85

# ffmpeg -i parse patterns.
_DURATION_RE = re.compile(rb"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_STREAM_RE = re.compile(rb"Stream #\d+:\d+.*: Video:\s*([a-zA-Z0-9_]+).*?,\s*[^,]*?(\d{2,5})x(\d{2,5})")
_STREAM_RE = re.compile(rb"Stream #\d+:\d+.*?:\s*(Video|Audio|Subtitle|Data):")
# Decoder error markers in ffmpeg stderr that indicate a BAD BITSTREAM (not a containment/resource
# failure such as "Resource temporarily unavailable" or "Error while opening decoder").
_DECODE_ERROR_RE = re.compile(
    rb"corrupt|invalid data|invalid nal|non existing pps|error splitting|decoding error|"
    rb"truncat|concealing|error while decoding|co located|missing reference",
    re.IGNORECASE,
)
# A containment/resource failure (not corruption): decode could not run, so we cannot judge the file.
_RESOURCE_FAIL_RE = re.compile(
    rb"resource temporarily unavailable|error while opening decoder|cannot allocate memory|"
    rb"out of memory|error opening output",
    re.IGNORECASE,
)
_OUT_TIME_US_RE = re.compile(rb"out_time_us=(\d+)")


class MediaValidator:
    """MediaValidator protocol: contained validation of an uploaded scan input."""

    def __init__(self, ffmpeg_exe: str | None = None, limits: Limits | None = None) -> None:
        self._ffmpeg = ffmpeg_exe
        self._limits = limits
        self.last_containment: Containment | None = None

    def _ffmpeg_path(self) -> str:
        if self._ffmpeg:
            return self._ffmpeg
        import imageio_ffmpeg

        self._ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        return self._ffmpeg

    def validate(self, scan, storage: Storage) -> MediaInfo:
        key = scan.input.uri
        head = storage.head(key)
        if head is None:
            raise ValueError("uploaded file not found in storage")

        scratch = Path(tempfile.mkdtemp(prefix="media_"))
        # Copy the media INTO the job dir (the only path visible in docker containment).
        src = storage.local_path(key, scratch)
        local = scratch / ("upload" + Path(key).suffix)
        if src.resolve() != local.resolve():
            local.write_bytes(src.read_bytes())
        data_head = local.read_bytes()[:64]

        kind = scan.input.type
        if kind == "mesh":
            return self._validate_mesh(local, scratch)
        if kind == "photos":
            return self._validate_photo(local, data_head, scratch)
        return self._validate_video(local, data_head, scratch)

    # -- video -------------------------------------------------------------

    def _validate_video(self, local: Path, head: bytes, scratch: Path) -> MediaInfo:
        if not any(head[off:off + len(sig)] == sig for off, sig in _VIDEO_MAGIC):
            raise ValueError("file does not look like a supported video (MP4/MOV/WebM)")

        res = run_contained([self._ffmpeg_path(), "-hide_banner", "-i", str(local)], scratch,
                            limits=self._limits)
        self.last_containment = res.containment
        text = res.stderr  # ffmpeg writes stream info to stderr

        dur = self._parse_duration(text)
        if dur is None:
            raise ValueError("could not read the video duration; the file may be truncated or corrupt")
        if not (_MIN_DURATION_S <= dur <= _MAX_DURATION_S):
            raise ValueError(
                f"video is {dur:.1f} s; the accepted range is {int(_MIN_DURATION_S)}-{int(_MAX_DURATION_S)} s"
            )

        vmatch = _VIDEO_STREAM_RE.search(text)
        if vmatch is None:
            raise ValueError("no video stream found in the file")
        codec = vmatch.group(1).decode("ascii", "replace").lower()
        width = int(vmatch.group(2))
        height = int(vmatch.group(3))
        if codec not in _ACCEPTED_VIDEO_CODECS:
            raise ValueError(f"video codec {codec!r} is not supported (need H.264/HEVC/VP8/VP9)")
        self._check_dims(width, height)

        n_streams = len(_STREAM_RE.findall(text))
        if n_streams == 0:
            raise ValueError("no media streams found")
        if n_streams > 6:
            raise ValueError(f"too many streams ({n_streams}); expected a single video (plus audio)")

        # Decode the stream to null: catch a truncated/undecodable bitstream the header hides.
        self._decode_check(local, scratch, declared_s=dur)

        hdr = self._detect_hdr(text)
        container = self._container_from_magic(head)
        return MediaInfo(kind="video", duration_s=round(dur, 2), container=container, codec=codec,
                         width=width, height=height, hdr=hdr)

    def _decode_check(self, local: Path, scratch: Path, declared_s: float) -> None:
        """Decode the video stream to null inside containment; reject on decoder errors or a readable
        duration far short of the declared one (SPEC 3.3, truncated phone video).

        Decode single-threaded (`-threads 1`) so the containment's process-count limit does not make a
        valid file look broken, with a dedicated (larger) decode budget. A resource/containment failure
        is re-raised so it is not mistaken for corruption."""
        from dataclasses import replace

        base = self._limits or Limits()
        decode_limits = replace(base, cpu_s=max(base.cpu_s, 120),
                                wall_timeout_s=max(base.wall_timeout_s, 180),
                                nproc=max(base.nproc, 512))

        prog = scratch / "progress.txt"
        res = run_contained(
            [self._ffmpeg_path(), "-v", "error", "-threads", "1", "-i", str(local),
             "-map", "0:v:0", "-an", "-threads", "1", "-progress", str(prog), "-f", "null", "-"],
            scratch, limits=decode_limits,
        )
        self.last_containment = res.containment
        if _RESOURCE_FAIL_RE.search(res.stderr) and not _DECODE_ERROR_RE.search(res.stderr):
            raise RuntimeError(
                "media decode could not run under containment (resource limit); not a file verdict"
            )
        if _DECODE_ERROR_RE.search(res.stderr):
            raise ValueError("the video could not be fully decoded; it is truncated or corrupt")

        readable_us = None
        if prog.exists():
            matches = _OUT_TIME_US_RE.findall(prog.read_bytes())
            if matches:
                readable_us = int(matches[-1])
        if readable_us is None:
            raise ValueError("the video stream could not be read to the end; it may be truncated")
        readable_s = readable_us / 1_000_000
        if readable_s < declared_s * _READABLE_DURATION_MIN_FRACTION:
            raise ValueError(
                f"only {readable_s:.1f} s of the declared {declared_s:.1f} s video could be decoded; "
                f"the file is truncated"
            )

    @staticmethod
    def _parse_duration(text: bytes) -> float | None:
        m = _DURATION_RE.search(text)
        if not m:
            return None
        h, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        total = h * 3600 + mm * 60 + ss
        # ffmpeg prints N/A as 00:00:00.00 only when it truly has it; a zero here means unknown.
        return total if total > 0 else None

    @staticmethod
    def _detect_hdr(text: bytes) -> bool:
        lowered = text.lower()
        return any(tag in lowered for tag in (b"bt2020", b"smpte2084", b"arib-std-b67", b"pq"))

    @staticmethod
    def _container_from_magic(head: bytes) -> str:
        if head[0:4] == b"\x1aE\xdf\xa3":
            return "webm"
        if head[4:8] == b"ftyp":
            brand = head[8:12]
            if brand.startswith((b"qt", b"\x00\x00\x00")):  # QuickTime major brand "qt  "
                return "mov"
            return "mp4"
        return "unknown"

    def _check_dims(self, width: int, height: int) -> None:
        if not (_MIN_DIM <= width <= _MAX_DIM and _MIN_DIM <= height <= _MAX_DIM):
            raise ValueError(f"frame size {width}x{height} is out of bounds")

    # -- photos ------------------------------------------------------------

    def _validate_photo(self, local: Path, head: bytes, scratch: Path) -> MediaInfo:
        if not any(head[off:off + len(sig)] == sig for off, sig in _PHOTO_MAGIC):
            raise ValueError("file does not look like a supported image (JPEG/PNG/WebP)")
        # Read headers/EXIF with Pillow inside the same containment, and fully load() the pixels so a
        # truncated image is rejected (verify() alone misses a cut-off body).
        script = (
            "import sys\n"
            "from PIL import Image, ImageOps\n"
            "p = sys.argv[1]\n"
            "Image.open(p).verify()\n"
            "im = ImageOps.exif_transpose(Image.open(p))\n"
            "im.load()\n"
            "print(f'{im.format}|{im.width}|{im.height}')\n"
        )
        res = run_contained([self._python_for_photos(), "-c", script, str(local)], scratch,
                            limits=self._limits)
        self.last_containment = res.containment
        if res.returncode != 0:
            raise ValueError("image could not be decoded; it may be truncated or not a real image")
        try:
            fmt, w, h = res.stdout.decode("ascii", "replace").strip().split("|")
            width, height = int(w), int(h)
        except Exception:  # noqa: BLE001
            raise ValueError("could not read image dimensions") from None
        self._check_dims(width, height)
        return MediaInfo(kind="photos", duration_s=None, container=fmt.lower(), codec=None,
                         width=width, height=height, hdr=False)

    def _python_for_photos(self) -> str:
        # In docker containment the image's own python3 is used (argv[0] basename); in rlimits mode the
        # current interpreter (which has Pillow) runs it.
        from screwit.storage.containment import docker_available

        return "python3" if docker_available() else sys.executable

    # -- mesh (stretch input) ---------------------------------------------

    def _validate_mesh(self, local: Path, scratch: Path) -> MediaInfo:
        head = local.read_bytes()[:12]
        if head[:4] != b"glTF":
            raise ValueError("mesh input must be a self-contained binary glTF (.glb)")
        return MediaInfo(kind="mesh", duration_s=None, container="glb", codec=None,
                         width=None, height=None, hdr=None)
