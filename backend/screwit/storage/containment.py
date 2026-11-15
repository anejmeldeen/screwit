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

RLIMIT_CPU counts CPU time summed over all of a process's threads, so one fixed CPU budget that suits
a short clip stops a long one part-way through. Budgets for work over a video therefore scale with
its duration (``media_budget``). Each rlimits run also records the tool's own CPU time (``wait4``), so
callers can tell a run its limits stopped from one that failed on its own
(``RunResult.containment_failure``) and never report the first as a verdict on the file.
"""

from __future__ import annotations

import math
import os
import resource
import shutil
import signal
import subprocess
import threading
import time
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


# --------------------------------------------------------------------------- media CPU budgets
#
# Measured on the dev host, an Apple M3 Pro (6 performance + 6 efficiency cores) with other work
# running, software-decoding better.mov (87.2 s of 3840x2160 30 fps 10-bit HLG HEVC at 25 Mbit/s, an
# iPhone capture) to null exactly as storage.validation does:
#   4 threads, default QoS: 122-161 CPU-s (1.40-1.85 per second of video), 34-45 s wall,
#                           3.3-3.6 of the 4 threads busy on average
#   1 thread:               139 CPU-s (1.59 per second of video), 139 s wall
#   4 threads held to the efficiency cores (taskpolicy -c background): 655 CPU-s (7.5 per second of
#                           video), 199 s wall
# The CPU work barely depends on the thread count; more threads spend it faster.
SOFTWARE_DECODE_CPU_S_PER_MEDIA_S = 1.85  # the worst default-QoS sample above
EFFICIENCY_CORE_SLOWDOWN = 4.0  # 7.5 / 1.85 measured on efficiency cores at background QoS
BUDGET_SAFETY = 1.5  # heavier captures (500 MB in 90 s is 44 Mbit/s) and load spikes
BUSY_THREAD_FRACTION = 0.8  # 3.3-3.6 of 4 decode threads busy, measured
MAX_BUDGETED_MEDIA_S = 90.0  # the longest video storage.validation accepts
# A failed run whose CPU time reached this fraction of its RLIMIT_CPU was stopped by that limit.
CPU_LIMIT_REACHED_FRACTION = 0.95


@dataclass(frozen=True)
class MediaBudget:
    """Limits for one contained run over a video: RLIMIT_CPU (summed over threads) and wall clock."""

    cpu_s: int
    wall_timeout_s: int


def media_budget(media_s: float, *, threads: int,
                 cpu_s_per_media_s: float = SOFTWARE_DECODE_CPU_S_PER_MEDIA_S,
                 fixed_cpu_s: float = 0.0, min_cpu_s: int = 0, min_wall_s: int = 0) -> MediaBudget:
    """CPU-time and wall-clock limits for a tool working through `media_s` seconds of video.

    The CPU budget is the measured need, ``fixed_cpu_s + media_s * cpu_s_per_media_s``, as if every
    thread ran on an efficiency core (``EFFICIENCY_CORE_SLOWDOWN``), times ``BUDGET_SAFETY``.
    ``threads`` busy threads use up RLIMIT_CPU that many times faster than the wall clock runs, so the
    wall-clock timeout is that CPU budget spread over ``threads * BUSY_THREAD_FRACTION`` threads.
    Both stay finite: ``media_s`` is capped at ``MAX_BUDGETED_MEDIA_S`` (a missing or absurd duration
    counts as that cap), so no header can buy more than the longest accepted video gets. ``min_*``
    are floors.

    better.mov (87.2 s, 4 threads) gets 969 CPU-s and 303 s; it needed 122-161 CPU-s and 34-45 s
    (655 CPU-s and 199 s on the efficiency cores alone).
    """
    if threads < 1:
        raise ValueError("threads must be at least 1")
    media_s = float(media_s)
    if not math.isfinite(media_s):
        media_s = MAX_BUDGETED_MEDIA_S
    media_s = min(max(media_s, 0.0), MAX_BUDGETED_MEDIA_S)
    need_cpu_s = fixed_cpu_s + media_s * cpu_s_per_media_s
    cpu_s = math.ceil(need_cpu_s * EFFICIENCY_CORE_SLOWDOWN * BUDGET_SAFETY)
    wall_s = math.ceil(cpu_s / (threads * BUSY_THREAD_FRACTION))
    return MediaBudget(cpu_s=max(int(min_cpu_s), cpu_s), wall_timeout_s=max(int(min_wall_s), wall_s))


# --------------------------------------------------------------------------- measured child runs

@dataclass(frozen=True)
class MeasuredRun:
    """A finished child process, with the CPU time it used."""

    returncode: int  # negative: killed by that signal (the subprocess convention)
    stdout: bytes
    stderr: bytes
    cpu_s: float | None  # user + system CPU seconds over all the child's threads (wait4)
    wall_s: float
    timed_out: bool  # the wall-clock timeout killed it


# After the child exits, how long to wait for its pipes to close (a grandchild may hold them open).
_PIPE_DRAIN_GRACE_S = 5.0


def _drain(stream, sink: bytearray, cap: int) -> None:
    """Read a pipe to EOF, keeping at most `cap` bytes, so the child never blocks on a full pipe."""
    try:
        with stream:
            while chunk := stream.read1(1 << 16):
                room = cap - len(sink)
                if room > 0:
                    sink += chunk[:room]
    except (OSError, ValueError):
        pass


def _feed(stream, data: bytes) -> None:
    try:
        with stream:
            stream.write(data)
    except (OSError, ValueError):  # the child exited without reading all of it (BrokenPipeError)
        pass


def _kill_tree(pid: int) -> None:
    """SIGKILL a child, with its process group when it leads one (the rlimits preexec setsid()s).

    A child that did not start its own group shares ours, so only the child itself is signalled."""
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
            return
    except OSError:
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def run_measured(argv: list[str], *, timeout_s: float, cwd: str | None = None,
                 env: dict[str, str] | None = None, preexec_fn=None, input_bytes: bytes | None = None,
                 max_output_bytes: int = 4 * 1024 * 1024) -> MeasuredRun:
    """Like ``subprocess.run(argv, capture_output=True, timeout=timeout_s)``, plus the child's CPU time.

    RLIMIT_CPU stops a process with SIGXCPU, which a tool may catch and turn into an ordinary exit code
    (ffmpeg exits 255), so the exit code alone cannot tell "stopped by its CPU limit" from "failed".
    The child's own CPU time from ``wait4`` can, and it stays exact while other threads of this
    process run subprocesses (``RUSAGE_CHILDREN`` would mix them in). On timeout the child is killed
    with SIGKILL, and ``timed_out`` is set. Output past ``max_output_bytes`` per stream is dropped.
    """
    start = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, preexec_fn=preexec_fn,
                            stdin=subprocess.PIPE if input_bytes is not None else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = bytearray(), bytearray()
    io_threads = [threading.Thread(target=_drain, args=(proc.stdout, out, max_output_bytes), daemon=True),
                  threading.Thread(target=_drain, args=(proc.stderr, err, max_output_bytes), daemon=True)]
    if input_bytes is not None:
        io_threads.append(threading.Thread(target=_feed, args=(proc.stdin, input_bytes), daemon=True))
    for t in io_threads:
        t.start()

    lock = threading.Lock()
    state = {"reaped": False, "timed_out": False}

    def _on_timeout() -> None:
        with lock:
            if not state["reaped"]:
                state["timed_out"] = True
                _kill_tree(proc.pid)

    timer = threading.Timer(timeout_s, _on_timeout)
    timer.daemon = True
    timer.start()
    usage = None
    try:
        _, status, usage = os.wait4(proc.pid, 0)
    except ChildProcessError:  # reaped elsewhere; the status is lost, as in subprocess itself
        status = 0
    except BaseException:  # e.g. KeyboardInterrupt while waiting: never leave the child running
        _kill_tree(proc.pid)
        try:
            os.wait4(proc.pid, 0)
        except ChildProcessError:
            pass
        proc.returncode = -signal.SIGKILL
        raise
    finally:
        with lock:
            state["reaped"] = True
        timer.cancel()
    wall_s = time.monotonic() - start
    proc.returncode = os.waitstatus_to_exitcode(status)
    for t in io_threads:
        t.join(timeout=_PIPE_DRAIN_GRACE_S)
    cpu_s = usage.ru_utime + usage.ru_stime if usage is not None else None
    return MeasuredRun(proc.returncode, bytes(out), bytes(err), cpu_s, wall_s, state["timed_out"])


# --------------------------------------------------------------------------- contained runs

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


# `docker run` exit codes that come from docker or from a signal inside the container, not the tool.
_DOCKER_CONTAINMENT_EXITS = {
    125: "could not be started by docker (exit 125)",
    126: "could not be run inside its container (exit 126)",
    127: "was not found inside its container (exit 127)",
    137: "was killed inside its container (SIGKILL, usually the memory cap)",
    143: "was terminated inside its container (SIGTERM)",
    152: "ran out of CPU time inside its container (SIGXCPU)",
    153: "exceeded its output size limit inside its container (SIGXFSZ)",
}


def _signal_name(signum: int) -> str:
    try:
        return f"{signal.Signals(signum).name} ({signum})"
    except ValueError:
        return f"signal {signum}"


@dataclass
class RunResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    containment: Containment
    cpu_s: float | None = None  # the tool's own CPU time; None under docker (it runs in the daemon)
    wall_s: float | None = None
    timed_out: bool = False
    limits: Limits | None = None

    def containment_failure(self) -> str | None:
        """Why the containment, not the tool, ended this run, or None if the tool finished by itself.

        Hitting the wall-clock or CPU-time limit, being killed by a signal, or docker failing to run
        the container all cut a run short, so its output says nothing about the input file. The CPU
        limit is recognised from the measured CPU time, because ffmpeg catches SIGXCPU and exits 255,
        which looks like an ordinary failure.
        """
        if self.timed_out:
            limit = f" of {self.limits.wall_timeout_s} s" if self.limits is not None else ""
            return f"hit its wall-clock limit{limit}"
        if self.returncode == 0:
            return None
        if (self.limits is not None and self.cpu_s is not None
                and self.cpu_s >= CPU_LIMIT_REACHED_FRACTION * self.limits.cpu_s):
            return f"ran out of its CPU budget (used {self.cpu_s:.1f} of {self.limits.cpu_s} CPU-seconds)"
        if self.returncode < 0:
            return f"was killed by {_signal_name(-self.returncode)}"
        if self.containment.mechanism == "docker":
            return _DOCKER_CONTAINMENT_EXITS.get(self.returncode)
        return None


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
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(docker_cmd, input=input_bytes, capture_output=True,
                              timeout=limits.wall_timeout_s, check=False)
        out, err, rc = proc.stdout[:max_output_bytes], proc.stderr[:max_output_bytes], proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")[:max_output_bytes]
        err = (e.stderr or b"")[:max_output_bytes] + b"\n[contained docker run timed out]"
        rc = 124
        timed_out = True
    return RunResult(rc, out, err, Containment(network_isolated=True, mechanism="docker", rlimits=True),
                     wall_s=time.monotonic() - start, timed_out=timed_out, limits=limits)


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
    run = run_measured(wrapped, timeout_s=limits.wall_timeout_s, cwd=str(scratch), env=env,
                       preexec_fn=_preexec(limits), input_bytes=input_bytes,
                       max_output_bytes=max_output_bytes)
    rc, err = run.returncode, run.stderr
    if run.timed_out:
        err += b"\n[contained run timed out]"
        rc = 124
    return RunResult(rc, run.stdout, err,
                     Containment(network_isolated=net_ok, mechanism=mechanism, rlimits=True),
                     cpu_s=run.cpu_s, wall_s=run.wall_s, timed_out=run.timed_out, limits=limits)


def run_contained(argv: list[str], scratch: Path, *, limits: Limits | None = None,
                  input_bytes: bytes | None = None, max_output_bytes: int = 4 * 1024 * 1024) -> RunResult:
    """Run `argv` isolated, preferring docker, then the rlimits fallback. Output is truncated."""
    limits = limits or Limits()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if docker_available():
        return _run_docker(argv, scratch, limits, input_bytes, max_output_bytes)
    return _run_rlimits(argv, scratch, limits, input_bytes, max_output_bytes)
