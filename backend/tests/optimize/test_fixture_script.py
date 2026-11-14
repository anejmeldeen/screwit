"""Test scripts/make_fixture_mobile.py on a synthetic scan fixture layout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "make_fixture_mobile.py"


def _run(root: Path) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO_ROOT / "backend")}
    import os

    env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True, text=True, env=env, check=False,
    )


def test_exits_zero_when_no_meshes(tmp_path):
    (tmp_path / "fixtures" / "scans" / "scn_empty").mkdir(parents=True)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "no mesh/object.glb" in result.stdout or "no fixture meshes" in result.stdout


def test_builds_mobile_mesh(tmp_path):
    try:
        from screwit.optimize.mesh import gltfpack_binary

        gltfpack_binary()
    except Exception:
        pytest.skip("gltfpack not installed")

    scan = tmp_path / "fixtures" / "scans" / "scn_synth"
    (scan / "mesh").mkdir(parents=True)
    m = trimesh.creation.icosphere(subdivisions=6, radius=50.0)  # ~80k tris
    pos = np.asarray(m.vertices)
    span = np.ptp(pos, axis=0) + 1e-9
    rgba = np.c_[((pos - pos.min(0)) / span * 255).astype(np.uint8),
                 np.full((len(pos), 1), 255, np.uint8)]
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
    m.export(str(scan / "mesh" / "object.glb"), file_type="glb")

    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    out = scan / "mobile" / "object.glb"
    assert out.exists(), result.stdout
    assert "scn_synth" in result.stdout
