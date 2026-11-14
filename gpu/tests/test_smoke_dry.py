import json
import subprocess
import sys
from pathlib import Path

import compare
import numpy as np
import pytest
import smoke

GPU = Path(__file__).resolve().parents[1]


def test_dry_run_writes_reports(tmp_path):
    rc = smoke.main(["x.mp4", "--dry-run", "--run", "t", "--out-root", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "t" / "report.json").read_text())
    assert data["dry_run"] is True
    assert data["stages"]["free_vram_leak_detected"] is True
    assert data["stages"]["free_vram_logic"]["allocated_after_mb"] < 1024
    assert "free_vram_logic" in (tmp_path / "t" / "report.md").read_text()


def test_dry_run_as_script_without_torch(tmp_path):
    out = subprocess.run(
        [sys.executable, str(GPU / "smoke.py"), "v.mp4", "--dry-run", "--run", "s",
         "--out-root", str(tmp_path), "--views", "32,48"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads((tmp_path / "s" / "report.json").read_text())["args"]["views"] == [32, 48]


def test_bad_args():
    with pytest.raises(SystemExit):
        smoke.main(["v.mp4", "--views", "a,b", "--dry-run"])
    with pytest.raises(SystemExit):
        smoke.main(["v.mp4", "--views", "2", "--dry-run"])
    assert smoke.main(["v.mp4", "--stages", "nope", "--dry-run"]) == 2


def test_missing_video_and_board(tmp_path):
    assert smoke.main(["nope.mp4", "--out-root", str(tmp_path)]) in (2, 3)
    missing = str(tmp_path / "none.json")
    assert smoke.main(["v.mp4", "--dry-run", "--board-spec", missing, "--out-root", str(tmp_path)]) == 2


def test_board_spec_recorded(tmp_path):
    spec = tmp_path / "b.json"
    spec.write_text('{"squares": [5, 7]}')
    rc = smoke.main(["v.mp4", "--dry-run", "--run", "b", "--board-spec", str(spec), "--out-root", str(tmp_path)])
    assert rc == 0
    got = json.loads((tmp_path / "b" / "report.json").read_text())
    assert got["args"]["board_spec_content"] == {"squares": [5, 7]}


class FakeCuda:
    def __init__(self, allocated_mb, drops_to_mb):
        self.a = allocated_mb * 2**20
        self.drops = drops_to_mb * 2**20

    def memory_allocated(self):
        return self.a

    def memory_reserved(self):
        return self.a

    def synchronize(self):
        pass

    def empty_cache(self):
        self.a = self.drops


class FakeTorch:
    def __init__(self, allocated_mb, drops_to_mb):
        self.cuda = FakeCuda(allocated_mb, drops_to_mb)


def test_free_gpu_ok_clears_holders_and_prediction():
    class Pred:
        gaussians = object()
        aux = {"x": 1}

    pred = Pred()
    holders = {"da3_prediction": pred, "model": object()}
    res = smoke.free_gpu(FakeTorch(20000, 200), holders, 1024)
    assert holders == {}
    assert pred.gaussians is None and pred.aux is None
    assert res["dropped"]["da3_prediction"] == ["gaussians", "aux"]
    assert res["allocated_before_mb"] == 20000 and res["allocated_after_mb"] == 200


def test_free_gpu_raises_when_memory_not_freed():
    with pytest.raises(smoke.VramNotFreed):
        smoke.free_gpu(FakeTorch(20000, 5000), {"m": object()}, 1024)


def _orbit(n=48, r=1.0, c2w=False):
    ex = []
    for k in range(n):
        th = 2 * np.pi * k / n
        c = np.array([r * np.cos(th), r * np.sin(th), 0.3])
        z = -c / np.linalg.norm(c)
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rot_c2w = np.stack([x, y, z], axis=1)
        if c2w:
            ex.append(np.hstack([rot_c2w, c[:, None]]))
        else:
            rw = rot_c2w.T
            ex.append(np.hstack([rw, (-rw @ c)[:, None]]))
    return np.stack(ex)


def test_extrinsics_world_to_camera_orbit():
    res = smoke.check_extrinsics(_orbit())
    assert res["verdict"] == "world_to_camera_consistent"
    assert res["rotation_orthonormality_max_err"] < 1e-9
    assert abs(res["orbit_if_world_to_camera"]["mean_radius"] - 1.0) < 0.1
    assert res["orbit_if_world_to_camera"]["angular_span_deg"] > 300


def test_extrinsics_4x4_and_wrong_convention_detected():
    ex = _orbit()
    ex4 = np.concatenate([ex, np.tile([0, 0, 0, 1.0], (len(ex), 1, 1))], axis=1)
    assert smoke.check_extrinsics(ex4)["verdict"] == "world_to_camera_consistent"
    assert smoke.check_extrinsics(_orbit(c2w=True))["verdict"] != "world_to_camera_consistent"
    assert smoke.check_extrinsics(np.zeros((3, 2, 2)))["verdict"] == "unexpected_shape"


def test_pick_evenly():
    assert smoke.pick_evenly(list(range(10)), 3) == [0, 4, 9]
    assert smoke.pick_evenly([1, 2], 5) == [1, 2]
    assert smoke.pick_evenly([1, 2, 3], 1) == [2]


def test_release_prediction_none_safe():
    assert smoke.release_prediction(None) == []


def test_markdown_renders_runs():
    md = smoke.render_markdown({
        "run": "r", "created_utc": "t", "dry_run": False, "environment": {"gpu": "A10G"},
        "frames": {"kept": 3},
        "stages": {"da3": {"runs": [{"views": 32, "status": "ok", "wall_s": 1.5, "peak_allocated_mb": 100.0}]}},
        "notes": ["n"],
    })
    assert "| 32 | ok | 1.5 | 100.0 |" in md


def test_compare_errors_and_cli(tmp_path):
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"stages": {"da3": {"runs": [{"views": 48, "status": "ok", "scale_factor": 1.2}]}}}))
    cal = tmp_path / "cal.json"
    cal.write_text('{"od": 31.75, "len": 100.0, "missing": 5}')
    scan = tmp_path / "scan.json"
    scan.write_text('{"od": 33.0, "len": 99.0, "extra": 1}')
    assert compare.main([str(rep), "--caliper", str(cal), "--scan-mm", str(scan)]) == 0
    res = json.loads((tmp_path / "compare.json").read_text())
    summ = res["errors"]["summary"]
    assert summ["paired"] == 2 and summ["unpaired_caliper"] == 1
    assert summ["extra_scan_names"] == ["extra"]
    rows = {r["name"]: r for r in res["errors"]["rows"]}
    assert abs(rows["od"]["error_mm"] - 1.25) < 1e-9
    assert res["model_scale"] == {"da3_48": 1.2}
    assert set(res["repo_modules"]) == set(compare.REPO_MODULES)
    assert "Repo modules" in (tmp_path / "compare.md").read_text()


def test_compare_bad_input(tmp_path):
    rep = tmp_path / "r.json"
    rep.write_text("{}")
    bad = tmp_path / "c.json"
    bad.write_text('{"a": -1}')
    assert compare.main([str(rep), "--caliper", str(bad)]) == 2
    assert compare.main([str(rep), "--caliper", str(tmp_path / "none.json")]) == 2
