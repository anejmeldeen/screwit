"""Containment isolation (SPEC 3.3, defect 3).

Asserts that a contained run cannot open a network socket and cannot read the repo when the
mechanism provides that isolation (docker, or unshare/bwrap network namespaces). On a host where only
the rlimits fallback is available, network and filesystem isolation are NOT provided, so those
assertions are skipped with an explicit reason rather than asserting something false.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from screwit.storage.containment import (
    detect_network_isolation,
    docker_available,
    run_contained,
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
