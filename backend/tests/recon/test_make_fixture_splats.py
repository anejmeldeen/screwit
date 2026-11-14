"""scripts/make_fixture_splats.py: sample a mesh into a splat and exit 0 when no mesh exists."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "make_fixture_splats.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("make_fixture_splats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sample_mesh_to_cloud(tmp_path):
    import trimesh

    mod = _load_script()
    box = trimesh.creation.box(extents=(80.0, 80.0, 100.0))
    box.visual.vertex_colors = np.tile([200, 100, 50, 255], (len(box.vertices), 1))
    mesh_path = tmp_path / "object.glb"
    box.export(mesh_path)
    cloud = mod.sample_mesh_to_cloud(mesh_path, max_gaussians=5000)
    assert 0 < len(cloud) <= 5000
    # Points lie on the box surface: within its bbox (trimesh centres the box at the origin).
    assert cloud.means[:, 2].min() >= -51 and cloud.means[:, 2].max() <= 51
    assert cloud.means[:, 0].min() >= -41 and cloud.means[:, 0].max() <= 41


def test_main_exits_zero_without_meshes(tmp_path, capsys):
    mod = _load_script()
    (tmp_path / "fixtures" / "scans" / "scn_x").mkdir(parents=True)
    rc = mod.main(["--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no mesh" in out or "no fixture meshes" in out


def test_main_writes_splat_when_mesh_exists(tmp_path):
    import trimesh

    mod = _load_script()
    scan = tmp_path / "fixtures" / "scans" / "scn_y"
    (scan / "mesh").mkdir(parents=True)
    trimesh.creation.box(extents=(50, 50, 50)).export(scan / "mesh" / "object.glb")
    rc = mod.main(["--root", str(tmp_path), "--max", "3000"])
    assert rc == 0
    assert (scan / "splat" / "scene.spz").exists()
