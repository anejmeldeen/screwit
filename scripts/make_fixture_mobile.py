#!/usr/bin/env python3
"""Build phone-sized fixture variants from the full-size fixture assets (owner: X5).

For each ``fixtures/scans/<id>/mesh/object.glb`` (P3 produces these later) write
``fixtures/scans/<id>/mobile/object.glb`` (phone mesh, at most 50k triangles, meshopt-compressed).
When a full-size splat PLY ``fixtures/scans/<id>/splat/scene.ply`` also exists, write
``fixtures/scans/<id>/mobile/scene.spz`` (phone splat, at most 400k gaussians).

If no fixture mesh exists yet, print a note and exit 0 so the fixture build does not fail before P3
lands.

Run: ``python scripts/make_fixture_mobile.py [--root <repo>]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from screwit.optimize import make_mobile_variants  # noqa: E402


def _gaussian_io():
    """P2's real gaussian IO when available (integrated main tree), else the local test IO.

    The optimize library never imports from another worktree; this script is the integration seam,
    so it prefers P2's ``screwit.recon.gaussians`` and degrades to the local reader/writer only so
    the fixture build still works in a worktree that lacks P2's module.
    """
    try:
        from screwit.recon import gaussians as g  # type: ignore

        return g
    except Exception:
        from screwit.optimize.splat import LocalGaussianIO

        return LocalGaussianIO()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phone-sized fixture variants.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root (default: this repo)")
    args = parser.parse_args()

    scans_dir = args.root / "fixtures" / "scans"
    if not scans_dir.is_dir():
        print(f"no fixtures/scans under {args.root}; nothing to do")
        return 0

    built = 0
    for scan_dir in sorted(p for p in scans_dir.iterdir() if p.is_dir()):
        full_mesh = scan_dir / "mesh" / "object.glb"
        if not full_mesh.exists():
            print(f"{scan_dir.name}: no mesh/object.glb yet (P3 produces it); skipping")
            continue

        full_splat = scan_dir / "splat" / "scene.ply"
        out_dir = scan_dir / "mobile"
        io = _gaussian_io() if full_splat.exists() else None

        variants = make_mobile_variants(
            full_mesh,
            full_splat if full_splat.exists() else None,
            out_dir,
            gaussian_io=io,
        )
        m = variants.mesh_stats
        print(
            f"{scan_dir.name}: mesh {m.in_triangles} -> {m.out_triangles} tris, "
            f"{m.in_bytes} -> {m.out_bytes} bytes -> {variants.mesh_mobile_path}"
        )
        if variants.splat_stats is not None:
            s = variants.splat_stats
            print(
                f"{scan_dir.name}: splat {s.in_gaussians} -> {s.out_gaussians} gaussians, "
                f"{s.in_bytes} -> {s.out_bytes} bytes -> {variants.splat_mobile_path}"
            )
        built += 1

    if built == 0:
        print("no fixture meshes found; exiting 0 (run again once P3 has written them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
