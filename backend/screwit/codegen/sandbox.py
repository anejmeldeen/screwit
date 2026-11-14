"""Run one untrusted CadQuery codegen job in a disposable, locked-down container (SPEC 3.3, X7 brief).

The container is the security boundary. Every job runs with:
    docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges
               --user 65534:65534 --pids-limit 128 --memory 1g --cpus 1 --tmpfs /tmp:size=64m
               -v <fresh per-job host dir>:/out:rw -i <image>
with a wall-clock timeout (default 30 s), then `docker kill`. Nothing else is mounted: no repo, no
data directory, no home, no environment passthrough, no secrets. The job (source + params) is written
to the container's stdin as JSON; the runner (deploy/codegen-sandbox/runner.py) writes model.stl and
model.step into /out and prints a one-line JSON summary on stdout.

An AST pre-check (codegen/astcheck.py) runs first as a convenience filter only; it is NOT the boundary.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from screwit.codegen.astcheck import check_source

DEFAULT_IMAGE = "screwit-codegen-sandbox:latest"
DEFAULT_TIMEOUT_S = 30.0
# The container writes into a host bind mount, so everything it leaves there is untrusted: only these
# two fixed names are read back, each must be a regular file (never a symlink, which would resolve on
# the HOST), and each is capped in size.
OUTPUT_NAMES = {"stl": "model.stl", "step": "model.step"}
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandbox run.

    `ok` means the program built and exported a mesh. On failure, `error` (and `traceback` when the
    failure was inside the container) explain why; `stage` says where it failed:
      - "ast"      rejected by the AST pre-check, container never started
      - "timeout"  killed at the wall-clock limit
      - "docker"   docker itself failed (image missing, daemon down)
      - "runner"   the container ran but the job failed (build/export/measure error)
    """

    ok: bool
    stage: str = ""
    error: str | None = None
    traceback: str | None = None
    volume_mm3: float | None = None
    bbox_mm: list | None = None
    watertight: bool | None = None
    stl_path: Path | None = None
    step_path: Path | None = None
    elapsed_s: float = 0.0
    summary: dict = field(default_factory=dict)


def _docker_run_argv(image: str, out_dir: Path, name: str) -> list[str]:
    return [
        "docker", "run", "--rm",
        "--name", name,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--pids-limit", "128",
        "--memory", "1g",
        "--cpus", "1",
        "--tmpfs", "/tmp:size=64m",
        "-v", f"{out_dir}:/out:rw",
        "-i", image,
    ]


def docker_available() -> bool:
    """True if `docker info` succeeds. Tests that need the daemon skip when this is False."""
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_job(source: str, params: dict, *, out_root: Path, image: str = DEFAULT_IMAGE,
            timeout_s: float = DEFAULT_TIMEOUT_S, skip_ast: bool = False) -> SandboxResult:
    """Run one codegen job. Creates a fresh per-job host directory under `out_root` for the mount.

    `out_root` must be a host-writable directory; a unique subdirectory is made world-writable so the
    container's uid 65534 can write into it. Returns a SandboxResult; never executes `source` in this
    process.
    """
    t0 = time.monotonic()

    if not skip_ast:
        ast_res = check_source(source)
        if not ast_res.ok:
            return SandboxResult(ok=False, stage="ast", error=ast_res.reason,
                                 elapsed_s=time.monotonic() - t0)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"job_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True)
    # uid 65534 inside the container must be able to write here. 0o777 via os.chmod (a shell
    # `chmod 777` is a different, blocked path); this dir is a throwaway per-job scratch mount.
    out_dir.chmod(0o777)

    job = json.dumps({"source": source, "params": params})
    name = f"screwit-cg-{out_dir.name[len('job_'):]}"
    argv = _docker_run_argv(image, out_dir, name)

    try:
        proc = subprocess.run(argv, input=job.encode("utf-8"), capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # Killing the docker CLI does not stop the container; kill it by its unique name.
        _kill_container(name)
        return SandboxResult(ok=False, stage="timeout",
                             error=f"job exceeded the {timeout_s:g}s wall-clock limit and was killed",
                             elapsed_s=time.monotonic() - t0)
    except FileNotFoundError:
        return SandboxResult(ok=False, stage="docker", error="docker executable not found",
                             elapsed_s=time.monotonic() - t0)

    elapsed = time.monotonic() - t0
    stdout = proc.stdout.decode("utf-8", "replace").strip()

    if proc.returncode != 0 and not stdout:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        return SandboxResult(ok=False, stage="docker",
                             error=f"docker exited {proc.returncode}: {stderr[:500]}",
                             elapsed_s=elapsed)

    try:
        summary = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except Exception as e:  # noqa: BLE001
        return SandboxResult(ok=False, stage="runner",
                             error=f"could not parse runner summary: {e!r}; raw={stdout[:500]!r}",
                             elapsed_s=elapsed)

    if not summary.get("ok"):
        return SandboxResult(ok=False, stage="runner", error=summary.get("error"),
                             traceback=summary.get("traceback"), elapsed_s=elapsed, summary=summary)

    try:
        stl = _trusted_output(out_dir, "stl")
        step = _trusted_output(out_dir, "step")
    except ValueError as e:
        return SandboxResult(ok=False, stage="runner", error=str(e), elapsed_s=elapsed,
                             summary=summary)
    return SandboxResult(
        ok=True, stage="", volume_mm3=summary.get("volume_mm3"), bbox_mm=summary.get("bbox_mm"),
        watertight=summary.get("watertight"), stl_path=stl, step_path=step, elapsed_s=elapsed,
        summary=summary,
    )


def _trusted_output(out_dir: Path, kind: str) -> Path:
    """The fixed output file for ``kind``, checked without following links. The container's summary
    only says that the job succeeded; its file names are ignored."""
    import os
    import stat

    path = out_dir / OUTPUT_NAMES[kind]
    try:
        st = os.lstat(path)
    except FileNotFoundError as e:
        raise ValueError(f"sandbox reported success but wrote no {OUTPUT_NAMES[kind]}") from e
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"sandbox output {OUTPUT_NAMES[kind]} is not a regular file")
    if st.st_size > MAX_OUTPUT_BYTES:
        raise ValueError(f"sandbox output {OUTPUT_NAMES[kind]} is {st.st_size} bytes "
                         f"(limit {MAX_OUTPUT_BYTES})")
    return path


def _kill_container(name: str) -> None:
    """Best-effort kill of a timed-out job's container (``--rm`` then removes it)."""
    for cmd in (["docker", "kill", name], ["docker", "rm", "-f", name]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception:  # noqa: BLE001
            pass
