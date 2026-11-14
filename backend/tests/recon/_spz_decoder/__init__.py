"""Independent SPZ decoder for golden verification (Niantic spz C++ loadSpz, MIT, vendored).

``decode(path)`` runs the ``spz_dump`` CLI on a .spz file and returns the first gaussian's
attributes as decoded by Niantic's own ``loadSpz`` -- an INDEPENDENT reader, not our writer. The
binary is prebuilt and committed; ``ensure_decoder()`` rebuilds it from the vendored spz sources
with g++ if it is missing and a compiler is available (otherwise the test skips).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE / "spz_dump"
SRC = HERE / "spz_src"


def ensure_decoder() -> Path | None:
    """Return the path to a working spz_dump binary, building it if needed, or None if impossible."""
    if BINARY.exists() and _works(BINARY):
        return BINARY
    gpp = shutil.which("g++") or shutil.which("c++")
    if not gpp:
        return None
    cmd = [gpp, "-std=c++17", "-O2", "-I", str(SRC), str(HERE / "spz_dump.cc"),
           str(SRC / "load-spz.cc"), str(SRC / "zstd-batch.cc"), str(SRC / "splat-types.cc"),
           "-lz", "-lzstd", "-lpthread", "-o", str(BINARY)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    return BINARY if BINARY.exists() else None


def _works(binary: Path) -> bool:
    try:
        r = subprocess.run([str(binary)], capture_output=True, timeout=10)
        return r.returncode in (0, 2)  # 2 = usage (no file); means it runs
    except (OSError, subprocess.SubprocessError):
        return False


def decode(spz_path: Path, n: int = 1) -> dict:
    """Decode a .spz with the independent C++ loader; return {'num_points', 'sh_degree', 'gaussians'}
    where each gaussian is a dict of pos/scale/rot(xyzw)/alpha/color."""
    binary = ensure_decoder()
    if binary is None:
        raise RuntimeError("no spz decoder available")
    out = subprocess.run([str(binary), str(spz_path), str(n)], capture_output=True, text=True,
                         timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"spz_dump failed: {out.stderr}")
    lines = out.stdout.strip().splitlines()
    header = None
    header_idx = 0
    for i, ln in enumerate(lines):
        m = re.search(r"numPoints (\d+) shDegree (\d+)", ln)
        if m and ln.startswith("numPoints"):
            header = m
            header_idx = i
            break
    if header is None:
        raise RuntimeError(f"could not parse spz_dump output: {out.stdout!r}")
    result: dict = {"num_points": int(header.group(1)), "sh_degree": int(header.group(2)),
                    "gaussians": []}
    for ln in lines[header_idx + 1:]:
        if not ln.startswith("g"):
            continue
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", ln)]
        result["gaussians"].append({
            "pos": nums[0:3], "scale": nums[3:6], "rot_xyzw": nums[6:10],
            "alpha": nums[10], "color": nums[11:14],
        })
    return result
