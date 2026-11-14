"""Contained subprocess runner for untrusted-media tools (SPEC 3.3).

Every ffmpeg/image-header read of uploaded media runs here, isolated. Two mechanisms, in order:

1. **docker** (preferred when `docker info` works): a per-job container, no network, read-only root,
   all caps dropped, no-new-privileges, a non-root user, pid/memory/cpu caps, and the per-job
   directory bind-mounted as the ONLY writable path. This is the isolation the untrusted-code team
   uses on this host, and it works where unprivileged user namespaces do not.
2. **rlimits-only** fallback: a clean minimal env, the job dir as the only writable cwd, CPU/AS/NPROC/
   FSIZE rlimits and a wall-clock timeout, with `unshare -rn`/`bwrap` for no-network when they work
   unprivileged. No network namespace is available when neither does.

The job directory is mounted at /work inside the container; any argument that points inside the job
dir is rewritten to its /work path, so callers build commands with ordinary host paths.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Pinned image carrying ffmpeg (+ python3/Pillow for photo headers), built from deploy/Dockerfile.media.
MEDIA_IMAGE = os.environ.get("SCREWIT_MEDIA_IMAGE", "screwit-media:latest")
_CONTAINER_WORK = "/work"


@dataclass(frozen=True)
class Containment:
    """What isolation the runner achieved on this host."""

    network_isolated: bool
    mechanism: str  # "docker", "unshare", "bwrap", or "rlimits-only"
    rlimits: bool


@dataclass(frozen=True)
class Limits:
    cpu_s: int = 20
    address_space_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB (RLIMIT_AS)
    nproc: int = 64
    fsize_bytes: int = 512 * 1024 * 1024  # 512 MiB written output (RLIMIT_FSIZE)
    wall_timeout_s: int = 60
    memory_bytes: int = 2 * 1024 * 1024 * 1024  # docker --memory
    cpus: str = "1.0"                            # docker --cpus
    pids: int = 128                              # docker --pids-limit


def _probe(cmd: list[str]) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def docker_available() -> bool:
    """True when `docker info` works AND the media image is present (build it from deploy/)."""
    if not shutil.which("docker"):
        return False
    if not _probe(["docker", "info"]):
        return False
    r = subprocess.run(["docker", "image", "inspect", MEDIA_IMAGE], capture_output=True)
    return r.returncode == 0


@lru_cache(maxsize=1)
def detect_network_isolation() -> tuple[bool, str]:
    """Probe once whether unprivileged network isolation is available (rlimits fallback path)."""
    if shutil.which("unshare") and _probe(["unshare", "-rn", "true"]):
        return True, "unshare"
    if shutil.which("bwrap") and _probe(
        ["bwrap", "--unshare-net", "--dev", "/dev", "--ro-bind", "/", "/", "true"]
    ):
        return True, "bwrap"
    return False, "rlimits-only"


def _preexec(limits: Limits):
    def _try_setrlimit(which: int, soft: int, hard: int) -> None:
        # Each limit is best-effort: a platform that rejects or lacks it must not crash the child.
        # macOS, in particular, raises on RLIMIT_AS (processes reserve far more than any sane cap),
        # which previously surfaced as "Exception occurred in preexec_fn" and failed every ffmpeg run.
        try:
            resource.setrlimit(which, (soft, hard))
        except (ValueError, OSError, AttributeError):
            pass

    def set_limits() -> None:
        _try_setrlimit(resource.RLIMIT_CPU, limits.cpu_s, limits.cpu_s)
        _try_setrlimit(resource.RLIMIT_AS, limits.address_space_bytes, limits.address_space_bytes)
        if hasattr(resource, "RLIMIT_NPROC"):
            _try_setrlimit(resource.RLIMIT_NPROC, limits.nproc, limits.nproc)
        _try_setrlimit(resource.RLIMIT_FSIZE, limits.fsize_bytes, limits.fsize_bytes)
        os.setsid()

    return set_limits


@dataclass
class RunResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    containment: Containment


def _rewrite_for_container(argv: list[str], scratch: Path) -> list[str]:
    """Rewrite any argument that points inside the job dir to its /work container path.

    `ffmpeg`/`python3` (argv[0]) run from the image, so a leading absolute host interpreter/binary
    path is replaced with the bare tool name.
    """
    scratch_s = str(scratch)
    out: list[str] = []
    for i, a in enumerate(argv):
        if i == 0:
            # Tool name: use the basename so the image's own ffmpeg/python3 is used.
            out.append(Path(a).name if ("/" in a) else a)
            continue
        if a.startswith(scratch_s):
            rel = os.path.relpath(a, scratch_s)
            out.append(f"{_CONTAINER_WORK}/{rel}")
        else:
            out.append(a)
    return out


def _run_docker(argv: list[str], scratch: Path, limits: Limits,
                input_bytes: bytes | None, max_output_bytes: int) -> RunResult:
    cargv = _rewrite_for_container(argv, scratch)
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--pids-limit", str(limits.pids),
        "--memory", str(limits.memory_bytes),
        "--cpus", str(limits.cpus),
        "--mount", f"type=bind,src={scratch},dst={_CONTAINER_WORK}",
        "--workdir", _CONTAINER_WORK,
        MEDIA_IMAGE, *cargv,
    ]
    try:
        proc = subprocess.run(docker_cmd, input=input_bytes, capture_output=True,
                              timeout=limits.wall_timeout_s, check=False)
        out, err, rc = proc.stdout[:max_output_bytes], proc.stderr[:max_output_bytes], proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")[:max_output_bytes]
        err = (e.stderr or b"")[:max_output_bytes] + b"\n[contained docker run timed out]"
        rc = 124
    return RunResult(rc, out, err, Containment(network_isolated=True, mechanism="docker", rlimits=True))


def _run_rlimits(argv: list[str], scratch: Path, limits: Limits,
                 input_bytes: bytes | None, max_output_bytes: int) -> RunResult:
    net_ok, mechanism = detect_network_isolation()
    wrapped = argv
    if mechanism == "unshare":
        wrapped = ["unshare", "-rn", "--", *argv]
    elif mechanism == "bwrap":
        wrapped = [
            "bwrap", "--unshare-net", "--die-with-parent", "--dev", "/dev",
            "--ro-bind", "/", "/", "--bind", str(scratch), str(scratch),
            "--chdir", str(scratch), "--", *argv,
        ]
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch), "HOME": str(scratch), "LC_ALL": "C"}
    try:
        proc = subprocess.run(wrapped, cwd=str(scratch), env=env, input=input_bytes,
                              capture_output=True, timeout=limits.wall_timeout_s,
                              preexec_fn=_preexec(limits), check=False)
        out, err, rc = proc.stdout[:max_output_bytes], proc.stderr[:max_output_bytes], proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")[:max_output_bytes]
        err = (e.stderr or b"")[:max_output_bytes] + b"\n[contained run timed out]"
        rc = 124
    return RunResult(rc, out, err,
                     Containment(network_isolated=net_ok, mechanism=mechanism, rlimits=True))


def run_contained(argv: list[str], scratch: Path, *, limits: Limits | None = None,
                  input_bytes: bytes | None = None, max_output_bytes: int = 4 * 1024 * 1024) -> RunResult:
    """Run `argv` isolated, preferring docker, then the rlimits fallback. Output is truncated."""
    limits = limits or Limits()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if docker_available():
        return _run_docker(argv, scratch, limits, input_bytes, max_output_bytes)
    return _run_rlimits(argv, scratch, limits, input_bytes, max_output_bytes)
