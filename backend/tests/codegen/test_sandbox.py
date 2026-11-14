"""Container isolation + a valid build (X7 brief step 5). Docker-dependent; skips cleanly without it.

Every property the brief names is proven here against the real container:
  - a program that imports socket is rejected by the AST filter;
  - a variant that evades the AST filter still cannot connect from inside (--network none);
  - the host filesystem (the repo) is not visible inside (nothing mounted but /out);
  - a program that runs forever is killed at the wall-clock timeout;
  - a program cannot write outside /out (read-only root, tmpfs /tmp only);
  - the process runs as uid 65534;
  - a valid program produces a watertight STL with the expected bbox.

Untrusted source only ever runs inside the container; this module never execs it.
"""

from __future__ import annotations

from screwit.codegen.sandbox import run_job

# A build that pokes a hole at runtime via a dunder chain would be caught by the AST filter, so these
# "evasion" programs avoid dunders and disallowed imports; they rely on the container to contain them.


def test_valid_box_builds_watertight_with_expected_bbox(docker_or_skip, tmp_path):
    src = (
        "def build(params):\n"
        "    import cadquery as cq\n"
        "    return cq.Workplane('XY').box(12, 24, 36)\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip)
    assert res.ok, res.error
    assert res.watertight is True
    # box centred on origin: extents 12/24/36 -> bbox +-6/+-12/+-18
    lo, hi = res.bbox_mm
    assert abs((hi[0] - lo[0]) - 12) < 1e-3
    assert abs((hi[1] - lo[1]) - 24) < 1e-3
    assert abs((hi[2] - lo[2]) - 36) < 1e-3
    assert res.stl_path.exists() and res.step_path.exists()


def test_socket_import_rejected_by_ast(docker_or_skip, tmp_path):
    src = (
        "import socket\n"
        "def build(params):\n"
        "    socket.socket().connect(('1.1.1.1', 53))\n"
        "    import cadquery as cq\n"
        "    return cq.Workplane('XY').box(1, 1, 1)\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip)
    assert not res.ok
    assert res.stage == "ast"


def test_network_blocked_inside_even_when_ast_evaded(docker_or_skip, tmp_path):
    # Evade the import filter by using cadquery's own transitive deps is hard; instead prove the
    # boundary directly: skip the AST filter and try to open a socket from inside the container.
    # --network none means even a program that reaches `socket` cannot connect.
    src = (
        "def build(params):\n"
        "    import socket\n"
        "    s = socket.socket()\n"
        "    s.settimeout(3)\n"
        "    s.connect(('1.1.1.1', 53))\n"  # would succeed with a network; must fail here
        "    import cadquery as cq\n"
        "    return cq.Workplane('XY').box(1, 1, 1)\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip, skip_ast=True)
    assert not res.ok
    assert res.stage == "runner"
    # The failure is a network error (unreachable / permission), not a build error.
    blob = (res.error or "") + (res.traceback or "")
    assert "build(params) raised" in blob
    assert any(w in blob for w in ("Network", "unreachable", "Errno", "OSError", "timed out"))


def test_host_repo_not_visible_inside(docker_or_skip, tmp_path):
    # Nothing is mounted but /out. A program that lists the host repo path finds it absent.
    repo = "/local/home/brancli/.kiro/crew/workspace/screw-it"
    src = (
        "def build(params):\n"
        "    import os, cadquery as cq\n"  # os import -> blocked by AST; run with skip_ast
        f"    assert not os.path.exists({repo!r}), 'host repo is visible inside the container!'\n"
        "    return cq.Workplane('XY').box(2, 2, 2)\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip, skip_ast=True)
    # The build succeeds precisely because the assert holds (repo not visible).
    assert res.ok, res.error


def test_killed_at_timeout(docker_or_skip, tmp_path):
    src = (
        "def build(params):\n"
        "    while True:\n"
        "        pass\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip, skip_ast=True, timeout_s=8.0)
    assert not res.ok
    assert res.stage == "timeout"
    assert res.elapsed_s >= 7.0
    # The container itself is gone, not only the docker CLI process that waited on it.
    import subprocess
    import time

    deadline = time.monotonic() + 20
    while True:
        left = subprocess.run(["docker", "ps", "-aq", "--filter", "name=screwit-cg-"],
                              capture_output=True, text=True, timeout=15).stdout.split()
        if not left or time.monotonic() > deadline:
            break
        time.sleep(1)
    assert not left, f"timed-out sandbox containers still present: {left}"


def test_outputs_read_back_only_as_fixed_regular_files(tmp_path):
    """Whatever the container leaves in the bind mount is untrusted: a symlink named model.stl (which
    would resolve on the host) and an oversized file are both refused."""
    import pytest

    from screwit.codegen import sandbox as sb

    (tmp_path / "model.stl").symlink_to("/etc/hostname")
    with pytest.raises(ValueError, match="not a regular file"):
        sb._trusted_output(tmp_path, "stl")
    with pytest.raises(ValueError, match="wrote no model.step"):
        sb._trusted_output(tmp_path, "step")
    big = tmp_path / "model.step"
    with big.open("wb") as f:
        f.truncate(sb.MAX_OUTPUT_BYTES + 1)
    with pytest.raises(ValueError, match="limit"):
        sb._trusted_output(tmp_path, "step")


def test_cannot_write_outside_out(docker_or_skip, tmp_path):
    # Root fs is read-only and only /out + tmpfs /tmp are writable. Writing to /etc or /app fails.
    src = (
        "def build(params):\n"
        "    import cadquery as cq\n"
        "    ok = False\n"
        "    for path in ('/model.stl', '/app/evil.txt', '/etc/evil.txt'):\n"
        "        try:\n"
        "            open(path, 'w').write('x')\n"
        "        except OSError:\n"
        "            ok = True\n"
        "        else:\n"
        "            raise RuntimeError('wrote outside /out: ' + path)\n"
        "    assert ok\n"
        "    return cq.Workplane('XY').box(2, 2, 2)\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip, skip_ast=True)
    assert res.ok, res.error


def test_runs_as_uid_65534(docker_or_skip, tmp_path):
    src = (
        "def build(params):\n"
        "    import os, cadquery as cq\n"
        "    assert os.getuid() == 65534, 'ran as uid ' + str(os.getuid())\n"
        "    return cq.Workplane('XY').box(2, 2, 2)\n"
    )
    res = run_job(src, {}, out_root=tmp_path, image=docker_or_skip, skip_ast=True)
    assert res.ok, res.error
