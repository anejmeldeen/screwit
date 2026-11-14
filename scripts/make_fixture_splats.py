#!/usr/bin/env python3
"""Build fixture splats from the fixture meshes (owner: P2).

For each ``fixtures/scans/<id>/mesh/object.glb`` (P3 produces these later), sample up to 200k
gaussians on the mesh surface, colour them from the surface, and write
``fixtures/scans/<id>/splat/scene.spz`` with the project's own SPZ v4 writer. If no mesh exists yet,
print a note and exit 0 so the fixture build does not fail before P3 lands.

Run: ``python scripts/make_fixture_splats.py [--max N] [--root <repo>]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from screwit.recon import gaussians as G  # noqa: E402

MAX_GAUSSIANS = 200_000
MOBILE_GAUSSIANS = 20_000


def sample_mesh_to_cloud(mesh_path: Path, max_gaussians: int = MAX_GAUSSIANS) -> G.GaussianCloud:
    """Surface-sample a mesh into a gaussian cloud (scan-frame millimetres)."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    # Density from surface area, not face count (the fixture meshes are low-poly): about one gaussian
    # per square millimetre, capped, so the splat view reads as a continuous surface.
    area = float(mesh.area)
    n = int(min(max_gaussians, max(20_000, area / 1.0)))
    points, face_idx = trimesh.sample.sample_surface(mesh, n, seed=0)
    points = np.asarray(points, np.float32)

    # Colour from vertex colours (nearest vertex) or a neutral grey.
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors):
        from scipy.spatial import cKDTree

        tree = cKDTree(mesh.vertices)
        _, vidx = tree.query(points, k=1)
        rgb = np.asarray(mesh.visual.vertex_colors)[vidx, :3].astype(np.float32) / 255.0
    else:
        rgb = np.full((points.shape[0], 3), 0.6, np.float32)

    # f_dc so that 0.5 + SH_C0 * f_dc == rgb.
    f_dc = ((rgb - 0.5) / G.SH_C0).astype(np.float32)

    # Scale ~ the sample spacing so neighbouring gaussians overlap into a solid surface.
    spacing = float(np.sqrt(area / max(n, 1)))
    log_scale = np.log(max(spacing * 0.8, 0.3))
    log_scales = np.full((points.shape[0], 3), log_scale, np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], np.float32), (points.shape[0], 1))  # identity [w,x,y,z]
    opac = np.full((points.shape[0],), 4.0, np.float32)  # sigmoid ~0.98
    return G.GaussianCloud(means=points, log_scales=log_scales, quats_wxyz=quats,
                           opacities=opac, f_dc=f_dc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--max", type=int, default=MAX_GAUSSIANS)
    args = ap.parse_args(argv)

    scans_dir = args.root / "fixtures" / "scans"
    if not scans_dir.is_dir():
        print(f"no fixtures/scans directory at {scans_dir}; nothing to do")
        return 0

    made = 0
    for scan_dir in sorted(scans_dir.iterdir()):
        mesh_path = scan_dir / "mesh" / "object.glb"
        if not mesh_path.exists():
            print(f"{scan_dir.name}: no mesh/object.glb yet (P3 produces it); skipping")
            continue
        cloud = sample_mesh_to_cloud(mesh_path, args.max)
        out = scan_dir / "splat" / "scene.spz"
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = G.write_spz(cloud, out)
        print(f"{scan_dir.name}: wrote {out.relative_to(args.root)} "
              f"({manifest['gaussians']} gaussians, {manifest['bytes']} bytes, spz v{manifest['spz_version']})")
        # Phone splat: a sparser sample of the same surface, well under the 400k phone budget and
        # small enough for a slow-4G link. Referenced by reconstruction.json splat_mobile_uri.
        mobile = scan_dir / "mobile" / "scene.spz"
        mobile.parent.mkdir(parents=True, exist_ok=True)
        m_manifest = G.write_spz(sample_mesh_to_cloud(mesh_path, MOBILE_GAUSSIANS), mobile)
        print(f"{scan_dir.name}: wrote {mobile.relative_to(args.root)} "
              f"({m_manifest['gaussians']} gaussians, {m_manifest['bytes']} bytes)")
        made += 1

    if made == 0:
        print("no fixture meshes found; exited without writing any splats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
