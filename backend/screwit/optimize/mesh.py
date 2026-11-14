"""Full-size scan-frame mesh -> phone-sized GLB (X5, SPEC 6.1 step 6, AGENTS.md "Performance budgets").

Two steps, both in scan-frame millimetres with no axis or unit change (SPEC 3.1):

1. Quadric decimation to at most ``MAX_MOBILE_TRIANGLES`` (50k) with Open3D
   ``simplify_quadric_decimation``. Open3D is already a pinned project dependency (0.19.0), its
   quadric decimation preserves per-vertex colour (verified: a 327k-triangle coloured sphere
   decimates to 50k in ~2.2 s with colours kept and the bounding box within 0.005 mm), and it needs
   no extra package. pyfqmr is the alternative but is not installed and would add a dependency for
   no gain here.
2. meshopt compression with the official ``gltfpack`` binary (the pinned npm ``gltfpack`` package,
   which ships meshoptimizer's own cross-platform WASM build; see ``optimize/README.md`` and
   ``optimize/toolchain``). gltfpack is run with ``-cc -noq``: ``-cc`` writes ``EXT_meshopt_compression``,
   and ``-noq`` turns OFF ``KHR_mesh_quantization`` so vertex positions stay at their exact
   millimetre values (quantization would snap them to a 14-bit grid and bake a dequantize transform
   that not every loader honours, which would break the "coordinates unchanged" rule). The output
   loads in three.js only with ``MeshoptDecoder`` wired into ``GLTFLoader`` (see the README).

Coordinates are never rescaled or re-axised here; the phone GLB is the full mesh's geometry with
fewer triangles, in the same scan frame.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

# Phone mesh budget (AGENTS.md "Performance budgets": mesh at most 50k triangles).
MAX_MOBILE_TRIANGLES = 50_000
# Largest bounding-box corner shift we accept from decimation: 0.5 mm, or 1% of the box diagonal
# for large noisy scans. The guard exists to catch an axis or unit change, which moves the box by
# tens of millimetres or more; a real CPU scan of a 100 mm cube moved it 0.95 mm at 176k -> 50k.
BBOX_TOLERANCE_MM = 0.5
BBOX_TOLERANCE_FRACTION = 0.01


def bbox_tolerance_mm(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    return max(BBOX_TOLERANCE_MM, BBOX_TOLERANCE_FRACTION * float(np.linalg.norm(bbox_max - bbox_min)))


class GltfpackNotFound(RuntimeError):
    """Raised when the pinned gltfpack binary cannot be located."""


@dataclass(frozen=True)
class MeshStats:
    """Reported by :func:`optimize_mesh` for the optimizer's stats dict."""

    in_triangles: int
    out_triangles: int
    in_bytes: int
    out_bytes: int
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    bbox_shift_mm: float
    had_vertex_colours: bool


def gltfpack_binary() -> Path:
    """Locate the pinned gltfpack binary.

    Order: the ``SCREWIT_GLTFPACK`` override, the npm install under ``optimize/toolchain``,
    then anything on ``PATH``. Raises :class:`GltfpackNotFound` if none is found so callers can fall
    back to Draco or fail loudly rather than silently skipping compression.
    """
    import os

    override = os.environ.get("SCREWIT_GLTFPACK")
    if override and Path(override).exists():
        return Path(override)
    # The pinned npm package lives beside this module, under optimize/toolchain.
    local = Path(__file__).resolve().parent / "toolchain" / "node_modules" / ".bin" / "gltfpack"
    if local.exists():
        return local
    on_path = shutil.which("gltfpack")
    if on_path:
        return Path(on_path)
    raise GltfpackNotFound(
        "gltfpack not found. Install the pinned package with "
        "`npm install` in screwit/optimize/toolchain, or set SCREWIT_GLTFPACK."
    )


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    """Load a GLB/PLY as a single trimesh, concatenating a scene into one geometry."""
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no triangle geometry in {path}")
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"unsupported mesh type {type(loaded).__name__} in {path}")
    return mesh


def _vertex_colours(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Per-vertex RGB in 0..1 if the mesh carries vertex colours, else None."""
    visual = mesh.visual
    if getattr(visual, "kind", None) == "vertex":
        colours = np.asarray(visual.vertex_colors)
        if colours.shape[0] == len(mesh.vertices):
            return colours[:, :3].astype(np.float64) / 255.0
    return None


def decimate(mesh: trimesh.Trimesh, max_triangles: int = MAX_MOBILE_TRIANGLES) -> trimesh.Trimesh:
    """Quadric-decimate to at most ``max_triangles``, preserving vertex colour and coordinates.

    A mesh already under the cap is returned unchanged (apart from being re-wrapped).
    """
    colours = _vertex_colours(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(faces) <= max_triangles:
        out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if colours is not None:
            out.visual = trimesh.visual.ColorVisuals(
                mesh=out, vertex_colors=(colours * 255).astype(np.uint8)
            )
        return out

    o3m = o3d.geometry.TriangleMesh()
    o3m.vertices = o3d.utility.Vector3dVector(vertices)
    o3m.triangles = o3d.utility.Vector3iVector(faces)
    if colours is not None:
        o3m.vertex_colors = o3d.utility.Vector3dVector(colours)

    dec = o3m.simplify_quadric_decimation(target_number_of_triangles=int(max_triangles))

    out_vertices = np.asarray(dec.vertices, dtype=np.float64)
    out_faces = np.asarray(dec.triangles, dtype=np.int64)
    out = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
    if dec.has_vertex_colors():
        out_cols = (np.asarray(dec.vertex_colors) * 255).astype(np.uint8)
        out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=out_cols)
    return out


def compress_glb(src_glb: Path, dst_glb: Path) -> None:
    """meshopt-compress a GLB with gltfpack ``-cc -noq`` (lossless positions, no quantization)."""
    binary = gltfpack_binary()
    dst_glb.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(binary), "-i", str(src_glb), "-o", str(dst_glb), "-cc", "-noq"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not dst_glb.exists():
        raise RuntimeError(
            f"gltfpack failed (exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )


def decompress_glb(src_glb: Path, dst_glb: Path) -> None:
    """Reverse gltfpack's meshopt+``-noq`` encoding so a loader without MeshoptDecoder (e.g. trimesh
    or the gltf-validator path in tests) can read the geometry back for verification."""
    binary = gltfpack_binary()
    dst_glb.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(binary), "-i", str(src_glb), "-o", str(dst_glb), "-noq"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not dst_glb.exists():
        raise RuntimeError(
            f"gltfpack decompress failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _bbox(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    return v.min(axis=0), v.max(axis=0)


def optimize_mesh(
    full_mesh_path: str | Path,
    out_glb_path: str | Path,
    max_triangles: int = MAX_MOBILE_TRIANGLES,
) -> MeshStats:
    """Full-size scan-frame GLB/PLY -> phone GLB (decimated + meshopt-compressed).

    Writes the compressed GLB to ``out_glb_path`` and returns :class:`MeshStats`. Raises
    :class:`ValueError` if decimation moves the bounding box by more than ``BBOX_TOLERANCE_MM``
    (a guard against an accidental axis or unit change).
    """
    full_mesh_path = Path(full_mesh_path)
    out_glb_path = Path(out_glb_path)
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)

    source = _load_trimesh(full_mesh_path)
    in_tris = len(source.faces)
    in_min, in_max = _bbox(source.vertices)
    had_colours = _vertex_colours(source) is not None

    decimated = decimate(source, max_triangles=max_triangles)
    out_min, out_max = _bbox(decimated.vertices)
    bbox_shift = float(
        max(np.abs(out_min - in_min).max(), np.abs(out_max - in_max).max())
    )
    tolerance = bbox_tolerance_mm(in_min, in_max)
    if bbox_shift > tolerance:
        raise ValueError(
            f"decimation moved the bounding box by {bbox_shift:.3f} mm "
            f"(> {tolerance:.2f} mm); refusing to write a shifted mesh"
        )

    # Write the decimated mesh uncompressed, then meshopt-compress in place.
    uncompressed = out_glb_path.with_suffix(".uncompressed.glb")
    decimated.export(str(uncompressed), file_type="glb")
    try:
        compress_glb(uncompressed, out_glb_path)
    finally:
        uncompressed.unlink(missing_ok=True)

    return MeshStats(
        in_triangles=in_tris,
        out_triangles=len(decimated.faces),
        in_bytes=full_mesh_path.stat().st_size,
        out_bytes=out_glb_path.stat().st_size,
        bbox_min_mm=tuple(float(x) for x in out_min),
        bbox_max_mm=tuple(float(x) for x in out_max),
        bbox_shift_mm=bbox_shift,
        had_vertex_colours=had_colours,
    )
