"""Full-size splat -> phone-sized splat (X5, SPEC 6.1 step 4, AGENTS.md "Performance budgets").

Selection, not re-fitting: keep at most ``MAX_MOBILE_GAUSSIANS`` (400k) of the input gaussians,
chosen by importance = ``sigmoid(opacity) * volume`` (volume from the three log scales), with
**spatial stratification** so the kept set covers the object evenly instead of clustering on the
densest region. Positions, rotations, scales, opacities and colours of the kept gaussians are
written through unchanged; nothing is rescaled or re-axised (SPEC 3.1).

**No cross-worktree import.** P2 owns the real SPZ writer and the 3DGS PLY reader
(``screwit.recon.gaussians``). The brief forbids importing from another worktree, so this module
talks to those capabilities through the :class:`GaussianIO` Protocol below and ships a minimal local
PLY reader / SPZ writer good enough for this stage and its tests. Integration wires P2's
``read_ply`` / ``write_spz`` in as the :class:`GaussianIO` implementation; the field names and
quaternion-order contract here match P2's ``GaussianCloud`` exactly (means, log_scales,
quats_wxyz = [w, x, y, z] PLY order, opacities = pre-sigmoid logits, f_dc = SH degree-0), so the
swap is a one-line wiring change with no data reshaping.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# Phone splat budget (AGENTS.md "Performance budgets": splat at most 400k gaussians).
MAX_MOBILE_GAUSSIANS = 400_000


@dataclass
class MobileGaussians:
    """The subset of fields this stage needs, in PLY order.

    Mirrors P2's ``screwit.recon.gaussians.GaussianCloud`` field-for-field so the integration swap
    needs no translation. Rotations are [w, x, y, z]; scales are natural-log scales; opacities are
    pre-sigmoid logits; ``f_dc`` is the SH degree-0 DC coefficient per channel.
    """

    means: np.ndarray        # (N, 3) float32, millimetres, scan frame
    log_scales: np.ndarray   # (N, 3) float32
    quats_wxyz: np.ndarray   # (N, 4) float32, [w, x, y, z], unit
    opacities: np.ndarray    # (N,) float32, inverse-sigmoid logit
    f_dc: np.ndarray         # (N, 3) float32

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def select(self, idx: np.ndarray) -> MobileGaussians:
        return MobileGaussians(
            means=self.means[idx].copy(),
            log_scales=self.log_scales[idx].copy(),
            quats_wxyz=self.quats_wxyz[idx].copy(),
            opacities=self.opacities[idx].copy(),
            f_dc=self.f_dc[idx].copy(),
        )


@runtime_checkable
class GaussianIO(Protocol):
    """Read a 3DGS PLY into, and write an SPZ from, a cloud with the fields of :class:`MobileGaussians`.

    P2's ``screwit.recon.gaussians`` satisfies this (``read_ply`` returns its ``GaussianCloud`` and
    ``write_spz`` consumes one; both share these field names). Integration passes that module-like
    object; tests pass :class:`LocalGaussianIO`.
    """

    def read_ply(self, path: Path): ...

    def write_spz(self, cloud, path: Path) -> dict: ...


# --------------------------------------------------------------------------- importance selection

def _volumes(log_scales: np.ndarray) -> np.ndarray:
    """Per-gaussian volume proxy from log scales: exp(sum of the three log scales) = product of the
    per-axis sigmas (proportional to the ellipsoid volume)."""
    return np.exp(np.sum(log_scales.astype(np.float64), axis=1))


def importance(cloud: MobileGaussians) -> np.ndarray:
    """Importance = sigmoid(opacity) * volume (SPEC 6.1 step 4 "chosen by importance")."""
    alpha = 1.0 / (1.0 + np.exp(-cloud.opacities.astype(np.float64)))
    return alpha * _volumes(cloud.log_scales)


def _stratified_indices(means: np.ndarray, score: np.ndarray, keep: int) -> np.ndarray:
    """Pick ``keep`` indices spread evenly over space: bin the means into a grid sized so there are
    roughly a handful of gaussians per cell, then round-robin the highest-scoring member of each
    non-empty cell until the budget is filled. This keeps thin or low-density regions represented
    instead of letting the densest patch win every slot.
    """
    n = means.shape[0]
    if keep >= n:
        return np.arange(n)

    lo = means.min(axis=0)
    span = np.maximum(means.max(axis=0) - lo, 1e-6)
    # Aim for ~2 gaussians per occupied cell on average: cells ~= keep/2 total.
    target_cells = max(keep // 2, 1)
    cells_per_axis = max(int(round(target_cells ** (1.0 / 3.0))), 1)
    idx3 = np.minimum(
        (means - lo) / span * cells_per_axis, cells_per_axis - 1e-6
    ).astype(np.int64)
    cell = (idx3[:, 0] * cells_per_axis + idx3[:, 1]) * cells_per_axis + idx3[:, 2]

    # Order every gaussian by (cell, -score) so each cell's members are best-first and contiguous.
    order = np.lexsort((-score, cell))
    sorted_cell = cell[order]
    # Rank within cell: 0 for the best in each cell, 1 for the next, ...
    new_cell = np.empty(n, dtype=bool)
    new_cell[0] = True
    new_cell[1:] = sorted_cell[1:] != sorted_cell[:-1]
    rank_in_cell = np.arange(n) - np.maximum.accumulate(np.where(new_cell, np.arange(n), 0))

    # Round-robin: take all rank-0 first (one per cell), then rank-1, ... Within a rank, best score
    # first so ties break toward the more important gaussian.
    priority = np.lexsort((-score[order], rank_in_cell))
    chosen = order[priority][:keep]
    return np.sort(chosen)


def select_gaussians(cloud: MobileGaussians, max_count: int = MAX_MOBILE_GAUSSIANS) -> MobileGaussians:
    """Keep at most ``max_count`` gaussians by importance with spatial stratification."""
    n = len(cloud)
    if n <= max_count:
        return cloud
    score = importance(cloud)
    idx = _stratified_indices(cloud.means.astype(np.float64), score, max_count)
    return cloud.select(idx)


# --------------------------------------------------------------------------- local PLY reader (tests)

def read_ply_local(path: Path) -> MobileGaussians:
    """Minimal 3DGS binary-little-endian PLY reader (INRIA layout: x y z, nx ny nz, f_dc_0..2,
    opacity, scale_0..2 log, rot_0..3 = [w, x, y, z]). For this stage's tests only; integration
    uses P2's reader through :class:`GaussianIO`.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header = b""
        while b"end_header\n" not in header:
            chunk = f.read(1)
            if not chunk:
                raise ValueError("PLY header not terminated")
            header += chunk
        text = header.decode("ascii")
        if "binary_little_endian" not in text:
            raise ValueError("only binary_little_endian PLY is supported")
        n = 0
        props: list[str] = []
        for line in text.splitlines():
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] != "float":
                    raise ValueError(f"only float properties are supported, got {line!r}")
                props.append(parts[-1])
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))

    idx = {name: i for i, name in enumerate(props)}

    def col(name: str) -> np.ndarray:
        return data[:, idx[name]].astype(np.float32)

    means = np.stack([col("x"), col("y"), col("z")], axis=1)
    f_dc = np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1)
    opac = col("opacity")
    log_scales = np.stack([col(f"scale_{i}") for i in range(3)], axis=1)
    quats = np.stack([col(f"rot_{i}") for i in range(4)], axis=1)  # [w, x, y, z]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    return MobileGaussians(means=means, log_scales=log_scales, quats_wxyz=quats.astype(np.float32),
                           opacities=opac, f_dc=f_dc)


def write_ply_local(cloud: MobileGaussians, path: Path) -> None:
    """Write a 3DGS binary PLY in the INRIA layout (used by tests to build fixtures)."""
    path = Path(path)
    n = len(cloud)
    props = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in props)
    header += "end_header\n"
    cols = np.zeros((n, len(props)), dtype=np.float32)
    cols[:, 0:3] = cloud.means
    cols[:, 6:9] = cloud.f_dc
    cols[:, 9] = cloud.opacities
    cols[:, 10:13] = cloud.log_scales
    cols[:, 13:17] = cloud.quats_wxyz
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(cols.astype("<f4").tobytes())


# --------------------------------------------------------------------------- minimal local SPZ (tests)

# Mirror P2's SPZ v4 constants so a file written here is byte-compatible with Spark >= 2.4.0.
_SPZ_MAGIC = 0x5053474E  # "NGSP"
_SPZ_VERSION = 4
_SPZ_FRACTIONAL_BITS = 12
_SH_C0 = 0.28209479177387814
_SPZ_COLOR_SCALE = 0.15


def _pack_quat_smallest_three(quats_xyzw: np.ndarray) -> np.ndarray:
    q = quats_xyzw.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    n = q.shape[0]
    i_largest = np.argmax(np.abs(q), axis=1)
    negate = q[np.arange(n), i_largest] < 0
    sqrt1_2 = np.sqrt(0.5)
    comp = i_largest.astype(np.uint32).copy()
    for i in range(4):
        sel = i != i_largest
        negbit = ((q[:, i] < 0) ^ negate) & sel
        mag = np.floor(((1 << 9) - 1) * (np.abs(q[:, i]) / sqrt1_2) + 0.5).astype(np.uint32)
        mag = np.clip(mag, 0, (1 << 9) - 1)
        step = (comp << np.uint32(10)) | (negbit.astype(np.uint32) << np.uint32(9)) | mag
        comp = np.where(sel, step, comp).astype(np.uint32)
    out = np.zeros((n, 4), dtype=np.uint8)
    out[:, 0] = comp & 0xFF
    out[:, 1] = (comp >> 8) & 0xFF
    out[:, 2] = (comp >> 16) & 0xFF
    out[:, 3] = (comp >> 24) & 0xFF
    return out


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def write_spz_local(cloud: MobileGaussians, path: Path) -> dict:
    """Minimal SPZ v4 writer (SH degree 0, from=UNSPECIFIED) for this stage's tests. Integration
    uses P2's ``write_spz`` through :class:`GaussianIO`; this mirrors its packing so the output is a
    valid Spark-readable file either way.
    """
    import zstandard as zstd

    path = Path(path)
    n = len(cloud)

    scale = float(1 << _SPZ_FRACTIONAL_BITS)
    fixed = np.round(cloud.means.astype(np.float64) * scale).astype(np.int64) & 0xFFFFFF
    pos = np.zeros((n, 3, 3), dtype=np.uint8)
    pos[:, :, 0] = fixed & 0xFF
    pos[:, :, 1] = (fixed >> 8) & 0xFF
    pos[:, :, 2] = (fixed >> 16) & 0xFF
    positions = pos.reshape(n * 9).tobytes()

    alpha = 1.0 / (1.0 + np.exp(-cloud.opacities.astype(np.float64)))
    alphas = _to_uint8(alpha * 255.0).tobytes()

    colour = 0.5 + _SH_C0 * cloud.f_dc.astype(np.float64)
    colors = _to_uint8(colour * (_SPZ_COLOR_SCALE * 255.0) + 0.5 * 255.0).reshape(n * 3).tobytes()

    scales = _to_uint8((cloud.log_scales.astype(np.float64) + 10.0) * 16.0).reshape(n * 3).tobytes()

    quats_xyzw = cloud.quats_wxyz[:, [1, 2, 3, 0]]
    rotations = _pack_quat_smallest_three(quats_xyzw).reshape(n * 4).tobytes()

    streams = [positions, alphas, colors, scales, rotations]
    cctx = zstd.ZstdCompressor(level=12)
    compressed = [cctx.compress(s) for s in streams]

    header = struct.pack(
        "<IIIBBBBI12s",
        _SPZ_MAGIC, _SPZ_VERSION, n,
        0, _SPZ_FRACTIONAL_BITS, 0, len(streams), 32, b"\x00" * 12,
    )
    toc = b"".join(struct.pack("<QQ", len(compressed[i]), len(streams[i])) for i in range(len(streams)))
    out = header + toc + b"".join(compressed)
    path.write_bytes(out)
    return {"spz_version": _SPZ_VERSION, "gaussians": n, "bytes": len(out), "pack_from": "UNSPECIFIED"}


class LocalGaussianIO:
    """:class:`GaussianIO` backed by this module's local PLY reader and SPZ writer (tests)."""

    def read_ply(self, path: Path) -> MobileGaussians:
        return read_ply_local(path)

    def write_spz(self, cloud, path: Path) -> dict:
        return write_spz_local(cloud, path)


# --------------------------------------------------------------------------- stage entry point

@dataclass(frozen=True)
class SplatStats:
    in_gaussians: int
    out_gaussians: int
    in_bytes: int
    out_bytes: int


def optimize_splat(
    full_splat_path: str | Path,
    out_spz_path: str | Path,
    io: GaussianIO,
    max_count: int = MAX_MOBILE_GAUSSIANS,
) -> SplatStats:
    """Full-size splat (3DGS PLY) -> phone ``.spz`` with at most ``max_count`` gaussians.

    ``io`` reads the input and writes the output; integration passes P2's ``screwit.recon.gaussians``
    module, tests pass :class:`LocalGaussianIO`.
    """
    full_splat_path = Path(full_splat_path)
    out_spz_path = Path(out_spz_path)
    out_spz_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = io.read_ply(full_splat_path)
    in_count = len(cloud)
    reduced = select_gaussians(cloud, max_count=max_count)
    io.write_spz(reduced, out_spz_path)

    return SplatStats(
        in_gaussians=in_count,
        out_gaussians=len(reduced),
        in_bytes=full_splat_path.stat().st_size,
        out_bytes=out_spz_path.stat().st_size,
    )
