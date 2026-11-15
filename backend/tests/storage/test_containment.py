"""Containment isolation (SPEC 3.3, defect 3).

Asserts that a contained run cannot open a network socket and cannot read the repo when the
mechanism provides that isolation (docker, or unshare/bwrap network namespaces). On a host where only
the rlimits fallback is available, network and filesystem isolation are NOT provided, so those
assertions are skipped with an explicit reason rather than asserting something false.
"""

from __future__ import annotations

import signal
import sys
import tempfile
from pathlib import Path

import pytest

from screwit.storage.containment import (
    MAX_BUDGETED_MEDIA_S,
    Containment,
    Limits,
    RunResult,
    detect_network_isolation,
    docker_available,
    media_budget,
    run_contained,
    run_measured,
)

REPO_MARKER = str(Path(__file__).resolve().parents[3] / "backend" / "screwit" / "contracts" / "models.py")


def _mechanism() -> str:
    if docker_available():
        return "docker"
    return detect_network_isolation()[1]


def _python() -> str:
    return "python3" if docker_available() else sys.executable


def test_contained_run_executes():
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    res = run_contained([_python(), "-c", "print('ok')"], scratch)
    assert res.returncode == 0
    assert res.stdout.strip() == b"ok"
    assert res.containment.rlimits is True
    assert res.containment.mechanism in ("docker", "unshare", "bwrap", "rlimits-only")


def test_contained_run_cannot_open_socket():
    mech = _mechanism()
    if mech == "rlimits-only":
        pytest.skip("rlimits-only host: no network namespace, socket isolation not provided here")
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    script = (
        "import socket, sys\n"
        "try:\n"
        "    s = socket.create_connection(('10.255.255.1', 80), timeout=2)\n"
        "    print('CONNECTED'); sys.exit(0)\n"
        "except OSError as e:\n"
        "    print('BLOCKED', e); sys.exit(3)\n"
    )
    res = run_contained([_python(), "-c", script], scratch)
    assert res.returncode != 0, f"socket was not blocked: {res.stdout!r}"
    assert b"CONNECTED" not in res.stdout


def test_contained_run_cannot_read_repo():
    mech = _mechanism()
    if mech != "docker":
        pytest.skip(f"{mech} host: filesystem isolation is only provided by docker containment")
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    script = (
        "import sys\n"
        f"try:\n    open({REPO_MARKER!r}).read(); print('READ'); sys.exit(0)\n"
        "except OSError:\n    print('DENIED'); sys.exit(3)\n"
    )
    res = run_contained([_python(), "-c", script], scratch)
    assert res.returncode != 0, "repo file was readable from the contained run"
    assert b"READ" not in res.stdout


def test_mechanism_is_recorded():
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    res = run_contained([_python(), "-c", "pass"], scratch)
    # docker/unshare/bwrap isolate the network; rlimits-only does not and says so.
    if res.containment.mechanism in ("docker", "unshare", "bwrap"):
        assert res.containment.network_isolated is True
    else:
        assert res.containment.network_isolated is False


# -- CPU budgets for media work -----------------------------------------------

def test_media_budget_scales_with_duration():
    short, long_ = media_budget(10, threads=4), media_budget(90, threads=4)
    assert long_.cpu_s == pytest.approx(9 * short.cpu_s, rel=0.01)
    assert long_.wall_timeout_s == pytest.approx(9 * short.wall_timeout_s, rel=0.01)


def test_media_budget_cpu_is_the_work_and_threads_set_the_wall_clock():
    # RLIMIT_CPU sums every thread: the work, and so the CPU budget, is the same whatever the thread
    # count, while fewer threads need proportionally longer on the wall clock.
    one, four = media_budget(60, threads=1), media_budget(60, threads=4)
    assert one.cpu_s == four.cpu_s
    assert one.wall_timeout_s == pytest.approx(4 * four.wall_timeout_s, rel=0.01)


def test_media_budget_stays_finite_for_bogus_durations():
    cap = media_budget(MAX_BUDGETED_MEDIA_S, threads=4)
    for bogus in (10_000.0, float("inf"), float("nan")):
        assert media_budget(bogus, threads=4) == cap
    assert media_budget(-5, threads=4) == media_budget(0, threads=4)
    with pytest.raises(ValueError):
        media_budget(10, threads=0)


def test_media_budget_floors():
    b = media_budget(1, threads=4, min_cpu_s=120, min_wall_s=180)
    assert (b.cpu_s, b.wall_timeout_s) == (120, 180)


def test_media_budget_covers_the_longest_4k_hdr_clip_with_headroom():
    # Measured worst case on the dev host (M3 Pro): better.mov, 87.2 s of 4K HDR HEVC, decoded in
    # software with 4 threads all held to efficiency cores (taskpolicy -c background) took 655 CPU-s
    # and 199 s. The longest accepted video (90 s) must fit with at least 1.4x to spare.
    b = media_budget(90, threads=4)
    assert b.cpu_s >= 1.4 * 655 * 90 / 87.23
    assert b.wall_timeout_s >= 1.4 * 199 * 90 / 87.23
    # The flat 120 CPU-s the decode check used to get ran out 73.8 s into that clip.
    assert media_budget(87.23, threads=4).cpu_s > 5 * 120


# -- measured runs and containment failures -------------------------------------

def test_run_measured_reports_the_childs_cpu_time():
    script = "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.5:\n    pass\n"
    run = run_measured([sys.executable, "-c", script], timeout_s=30)
    assert run.returncode == 0 and not run.timed_out
    assert 0.45 <= run.cpu_s < 5.0
    assert run.wall_s >= 0.45


def test_run_measured_timeout_kills_the_child():
    run = run_measured([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=1)
    assert run.timed_out
    assert run.returncode == -signal.SIGKILL
    assert run.wall_s < 10


def test_run_measured_feeds_input_and_caps_output():
    script = "import sys; sys.stdout.write(sys.stdin.read().upper() * 1000); sys.stderr.write('e')"
    run = run_measured([sys.executable, "-c", script], timeout_s=30, input_bytes=b"abc",
                       max_output_bytes=100)
    assert run.returncode == 0
    assert run.stdout == b"ABC" * 33 + b"A"
    assert run.stderr == b"e"


_rlimits_runner = pytest.mark.skipif(docker_available(),
                                     reason="CPU time is measured by the rlimits runner only")
_SPIN = "while True:\n    pass\n"


@_rlimits_runner
def test_cpu_limit_stop_is_a_containment_failure():
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    res = run_contained([sys.executable, "-c", _SPIN], scratch, limits=Limits(cpu_s=1, wall_timeout_s=30))
    assert res.returncode != 0 and not res.timed_out
    assert res.cpu_s >= 0.95
    assert "CPU budget" in res.containment_failure()


@_rlimits_runner
def test_cpu_limit_caught_and_turned_into_exit_255_is_still_a_containment_failure():
    # ffmpeg catches SIGXCPU and exits 255, which looks like an ordinary failure; the measured CPU
    # time gives it away. (Linux sends SIGKILL instead when the soft and hard limits are equal.)
    script = "import signal, sys\nsignal.signal(signal.SIGXCPU, lambda *_: sys.exit(255))\n" + _SPIN
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    res = run_contained([sys.executable, "-c", script], scratch, limits=Limits(cpu_s=1, wall_timeout_s=30))
    assert res.returncode in (255, -signal.SIGKILL)
    assert "CPU budget" in res.containment_failure()


def test_wall_clock_timeout_is_a_containment_failure():
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    res = run_contained([_python(), "-c", "import time; time.sleep(30)"], scratch,
                        limits=Limits(wall_timeout_s=1))
    assert res.returncode == 124 and res.timed_out
    assert "wall-clock limit" in res.containment_failure()


def test_ordinary_failure_is_not_a_containment_failure():
    scratch = Path(tempfile.mkdtemp(prefix="contain_"))
    res = run_contained([_python(), "-c", "import sys; sys.exit(3)"], scratch)
    assert res.returncode == 3
    assert res.containment_failure() is None


def test_containment_failure_classifies_exit_codes():
    rl = Containment(network_isolated=False, mechanism="rlimits-only", rlimits=True)
    dk = Containment(network_isolated=True, mechanism="docker", rlimits=True)
    assert "SIGKILL" in RunResult(-9, b"", b"", rl).containment_failure()
    assert RunResult(1, b"", b"", rl, cpu_s=0.1, limits=Limits()).containment_failure() is None
    # A run that finished is not a failure, however close to its budget it came.
    assert RunResult(0, b"", b"", rl, cpu_s=500.0, limits=Limits(cpu_s=20)).containment_failure() is None
    # Exit 137 means SIGKILL (usually the memory cap) only under docker; elsewhere it is the tool's own.
    assert "memory" in RunResult(137, b"", b"", dk).containment_failure()
    assert RunResult(137, b"", b"", rl).containment_failure() is None
    assert RunResult(1, b"", b"", dk).containment_failure() is None
