"""X5 phone asset optimizer (SPEC 6.1 steps 4 and 6, AGENTS.md "Performance budgets").

Produces the phone-sized variants of a scan's full-size assets, as a library P2's pipeline calls:
- mesh: full-size scan-frame GLB/PLY -> phone GLB, at most 50k triangles, meshopt-compressed.
- splat: full-size 3DGS PLY -> phone ``.spz``, at most 400k gaussians, chosen by importance.

Both stay in scan-frame millimetres with no axis or unit change (SPEC 3.1). These feed the optional
``mesh_mobile_uri`` and ``splat_mobile_uri`` on the Reconstruction contract (SPEC 4.2); the web client
picks the full or mobile asset by device capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from screwit.optimize.mesh import MAX_MOBILE_TRIANGLES, MeshStats, optimize_mesh
from screwit.optimize.splat import (
    MAX_MOBILE_GAUSSIANS,
    GaussianIO,
    SplatStats,
    optimize_splat,
)

__all__ = [
    "MAX_MOBILE_GAUSSIANS",
    "MAX_MOBILE_TRIANGLES",
    "GaussianIO",
    "MobileVariants",
    "make_mobile_variants",
]


@dataclass(frozen=True)
class MobileVariants:
    """Result of :func:`make_mobile_variants`: output paths plus size and count stats."""

    mesh_mobile_path: Path
    splat_mobile_path: Path | None
    mesh_stats: MeshStats
    splat_stats: SplatStats | None

    def as_dict(self) -> dict:
        return {
            "mesh_mobile_path": str(self.mesh_mobile_path),
            "splat_mobile_path": str(self.splat_mobile_path) if self.splat_mobile_path else None,
            "mesh_stats": asdict(self.mesh_stats),
            "splat_stats": asdict(self.splat_stats) if self.splat_stats else None,
        }


def make_mobile_variants(
    full_mesh_path: str | Path,
    full_splat_path: str | Path | None,
    out_dir: str | Path,
    *,
    gaussian_io: GaussianIO | None = None,
) -> MobileVariants:
    """Write phone variants of a scan's full-size assets into ``out_dir``.

    Writes ``out_dir/object.glb`` (phone mesh) always, and ``out_dir/scene.spz`` (phone splat) when
    ``full_splat_path`` is given. The splat path needs a :class:`GaussianIO` to read the PLY and write
    the ``.spz``; P2's pipeline passes ``screwit.recon.gaussians``. Tests pass
    ``screwit.optimize.splat.LocalGaussianIO``. Returns a :class:`MobileVariants` with output paths and
    per-asset stats (triangle/gaussian counts and byte sizes).

    Raises :class:`ValueError` if a splat was requested without a ``gaussian_io``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_out = out_dir / "object.glb"
    mesh_stats = optimize_mesh(full_mesh_path, mesh_out)

    splat_out: Path | None = None
    splat_stats: SplatStats | None = None
    if full_splat_path is not None:
        if gaussian_io is None:
            raise ValueError("a splat was requested but no gaussian_io was provided")
        splat_out = out_dir / "scene.spz"
        splat_stats = optimize_splat(full_splat_path, splat_out, gaussian_io)

    return MobileVariants(
        mesh_mobile_path=mesh_out,
        splat_mobile_path=splat_out,
        mesh_stats=mesh_stats,
        splat_stats=splat_stats,
    )
