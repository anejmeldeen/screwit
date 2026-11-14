"""3D Gaussian handling: 3DGS PLY read/write, the scan-frame similarity transform, the DA3-nested
scale_factor correction, cropping/pruning/downsampling, and a pure-Python SPZ v4 writer.

Quaternion orders (SPEC 3.1, AGENTS.md):
- 3DGS PLY stores rotations as [w, x, y, z] (rot_0 is w).
- The contracts and the spz GaussianCloud use [x, y, z, w].
This module keeps gaussians internally in PLY order [w, x, y, z] and reorders only at the spz
boundary, where the C++ library expects xyzw.

The SPZ writer reproduces Niantic spz's packGaussians exactly (verified byte-for-byte against the
library's own loadSpz): version 4, fractionalBits=12, scales (s+10)*16, colours c*0.15*255+127.5,
alpha sigmoid, smallest-three 10-bit quaternion, zstd level 12 per stream, stream order positions,
alphas, colours, scales, rotations (SH omitted at degree 0). Packed with from=UNSPECIFIED so the
numeric scan-frame coordinates are stored unchanged (SPEC 3.1 "SPZ packing, no conversion").
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SH degree-0 constant: f_dc -> colour via c = 0.5 + SH_C0 * f_dc.
SH_C0 = 0.28209479177387814
# spz colour scale (splat-utils.h colorScale).
SPZ_COLOR_SCALE = 0.15
# spz fixed packing parameters (load-spz.cc packGaussians).
SPZ_FRACTIONAL_BITS = 12
SPZ_VERSION = 4
SPZ_ZSTD_LEVEL = 12
# Spark version that reads spz v4 (SPEC 3.1 table); pinned with the pack version.
SPARK_VERSION = "2.4.0"


@dataclass
class GaussianCloud:
    """A set of 3D gaussians in the scan frame (or the model frame before the similarity is applied).

    Rotations are [w, x, y, z] (PLY order). Scales are natural-log scales. Opacity is the raw
    pre-sigmoid logit. Colours are f_dc SH degree-0 coefficients (one RGB triple per gaussian).
    """

    means: np.ndarray        # (N, 3) float32, millimetres in the scan frame
    log_scales: np.ndarray   # (N, 3) float32
    quats_wxyz: np.ndarray   # (N, 4) float32, [w, x, y, z], unit
    opacities: np.ndarray    # (N,) float32, inverse-sigmoid logit
    f_dc: np.ndarray         # (N, 3) float32, SH degree-0 DC coefficients

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        assert self.means.shape == (n, 3), self.means.shape
        assert self.log_scales.shape == (n, 3), self.log_scales.shape
        assert self.quats_wxyz.shape == (n, 4), self.quats_wxyz.shape
        assert self.opacities.shape == (n,), self.opacities.shape
        assert self.f_dc.shape == (n, 3), self.f_dc.shape

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def select(self, mask: np.ndarray) -> GaussianCloud:
        return GaussianCloud(
            means=self.means[mask].copy(),
            log_scales=self.log_scales[mask].copy(),
            quats_wxyz=self.quats_wxyz[mask].copy(),
            opacities=self.opacities[mask].copy(),
            f_dc=self.f_dc[mask].copy(),
        )


# --------------------------------------------------------------------------- quaternion helpers

def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b for [w, x, y, z] quaternions. `a` is (4,), `b` is (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def rotation_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Proper rotation matrix (3x3) -> unit quaternion [w, x, y, z]."""
    m = np.asarray(r, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return (q / np.linalg.norm(q)).astype(np.float32)


# --------------------------------------------------------------------------- transforms

def apply_scale_factor(cloud: GaussianCloud, scale_factor: float) -> GaussianCloud:
    """DA3-nested metric-alignment correction (SPEC 3.1 "Transforming gaussians").

    The nested model multiplies depth and extrinsic translations by prediction.scale_factor but
    leaves the gaussians unscaled, so before the scan-frame similarity we multiply the means by
    scale_factor and add ln(scale_factor) to the log scales. Skipping this offsets the splat from
    the mesh by exactly that factor.
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    out = cloud.select(np.ones(len(cloud), dtype=bool))
    out.means *= np.float32(scale_factor)
    out.log_scales += np.float32(np.log(scale_factor))
    return out


def apply_similarity(cloud: GaussianCloud, r: np.ndarray, t: np.ndarray, s: float) -> GaussianCloud:
    """Apply the similarity (R, t, s): mean' = s R mean + t, quat' = quat(R)*quat,
    log_scale' = log_scale + ln(s), opacity and colour unchanged (SPEC 3.1)."""
    if s <= 0:
        raise ValueError(f"similarity scale must be positive, got {s}")
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    means = (s * (cloud.means.astype(np.float64) @ r.T) + t).astype(np.float32)
    qr = rotation_to_quat_wxyz(r)
    quats = quat_mul_wxyz(qr.astype(np.float64), cloud.quats_wxyz.astype(np.float64))
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    log_scales = cloud.log_scales + np.float32(np.log(s))
    return GaussianCloud(
        means=means,
        log_scales=log_scales.astype(np.float32),
        quats_wxyz=quats.astype(np.float32),
        opacities=cloud.opacities.copy(),
        f_dc=cloud.f_dc.copy(),
    )


def translate(cloud: GaussianCloud, t: np.ndarray) -> GaussianCloud:
    """Shift means by t (used to fix the no-board origin before writing anything)."""
    out = cloud.select(np.ones(len(cloud), dtype=bool))
    out.means += np.asarray(t, dtype=np.float32).reshape(3)
    return out


# --------------------------------------------------------------------------- crop / prune / downsample

def crop_to_bbox(cloud: GaussianCloud, bbox_min: np.ndarray, bbox_max: np.ndarray,
                 margin_mm: float = 10.0) -> GaussianCloud:
    """Keep gaussians whose means fall inside the object bbox plus a margin (SPEC 6.1 step 4/5)."""
    lo = np.asarray(bbox_min, dtype=np.float32) - margin_mm
    hi = np.asarray(bbox_max, dtype=np.float32) + margin_mm
    m = np.all((cloud.means >= lo) & (cloud.means <= hi), axis=1)
    return cloud.select(m)


def prune_opacity(cloud: GaussianCloud, min_opacity: float = 0.1) -> GaussianCloud:
    """Drop gaussians whose sigmoid(opacity) is below the threshold (SPEC 6.1 step 4)."""
    alpha = 1.0 / (1.0 + np.exp(-cloud.opacities.astype(np.float64)))
    return cloud.select(alpha >= min_opacity)


def voxel_downsample(cloud: GaussianCloud, max_count: int = 1_500_000) -> GaussianCloud:
    """Voxel-downsample to the 1.5 M hard cap (SPEC 6.1 step 4). Keeps the highest-opacity gaussian
    per occupied voxel; grows the voxel size until the count is under the cap."""
    if len(cloud) <= max_count:
        return cloud
    means = cloud.means.astype(np.float64)
    lo = means.min(axis=0)
    span = np.maximum(means.max(axis=0) - lo, 1e-6)
    # Start from a voxel that would, if uniformly filled, yield ~max_count cells, then grow.
    voxel = float(np.cbrt(np.prod(span) / max(max_count, 1)))
    voxel = max(voxel, 1e-4)
    alpha = cloud.opacities.astype(np.float64)
    for _ in range(40):
        keys = np.floor((means - lo) / voxel).astype(np.int64)
        # Rank voxels; keep the max-opacity member of each.
        order = np.lexsort((-alpha, keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        kept_idx = order[first]
        if len(kept_idx) <= max_count:
            keep = np.zeros(len(cloud), dtype=bool)
            keep[kept_idx] = True
            return cloud.select(keep)
        voxel *= 1.26  # ~2x volume each step
    # Fallback: keep the top max_count by opacity.
    keep_idx = np.argsort(-alpha)[:max_count]
    keep = np.zeros(len(cloud), dtype=bool)
    keep[keep_idx] = True
    return cloud.select(keep)


# --------------------------------------------------------------------------- 3DGS PLY

_PLY_RGB_SCALE = SH_C0


def write_ply(cloud: GaussianCloud, path: Path) -> None:
    """Write a 3DGS binary little-endian PLY: x y z, nx ny nz (zero), f_dc_0..2, opacity,
    scale_0..2 (log), rot_0..3 ([w, x, y, z]). Matches the INRIA 3DGS layout DA3 exports."""
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
    # normals 3:6 left zero
    cols[:, 6:9] = cloud.f_dc
    cols[:, 9] = cloud.opacities
    cols[:, 10:13] = cloud.log_scales
    cols[:, 13:17] = cloud.quats_wxyz  # rot_0 = w

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(cols.astype("<f4").tobytes())


def read_gaussians_ply(path: Path) -> GaussianCloud:
    """Read a 3DGS binary PLY written by write_ply or DA3's gs_ply exporter.

    Public reader reused by the X5 phone-asset optimizer at integration (paired with write_spz), so
    both workstreams share one PLY<->spz code path. Rotations are returned in PLY order [w, x, y, z].
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Header is ASCII, terminated by 'end_header\n'.
        header_bytes = b""
        while b"end_header\n" not in header_bytes:
            chunk = f.read(1)
            if not chunk:
                raise ValueError("PLY header not terminated")
            header_bytes += chunk
        header = header_bytes.decode("ascii")
        if "binary_little_endian" not in header:
            raise ValueError("only binary_little_endian PLY is supported")
        lines = header.splitlines()
        n = 0
        props: list[str] = []
        for ln in lines:
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property"):
                parts = ln.split()
                if parts[1] != "float":
                    raise ValueError(f"only float properties are supported, got {ln!r}")
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
    return GaussianCloud(means=means, log_scales=log_scales, quats_wxyz=quats.astype(np.float32),
                         opacities=opac, f_dc=f_dc)


# --------------------------------------------------------------------------- SPZ v4 writer

def _pack_quaternion_smallest_three(quats_xyzw: np.ndarray) -> np.ndarray:
    """Pack normalized xyzw quaternions as spz v3/4 smallest-three 32-bit little-endian records.

    Mirrors packQuaternionSmallestThree: drop the largest component, flip so it is positive, store
    its 2-bit index and each remaining component as a sign bit + 9-bit magnitude scaled by sqrt(1/2).
    """
    q = quats_xyzw.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    n = q.shape[0]
    i_largest = np.argmax(np.abs(q), axis=1)
    negate = q[np.arange(n), i_largest] < 0
    sqrt1_2 = np.sqrt(0.5)
    comp = i_largest.astype(np.uint32).copy()
    for i in range(4):
        sel = i != i_largest
        negbit = ((q[:, i] < 0) ^ negate) & sel  # only meaningful where sel
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


def pack_spz_streams(cloud: GaussianCloud) -> list[bytes]:
    """Return the five uncompressed core streams (positions, alphas, colors, scales, rotations)
    at SH degree 0, exactly as Niantic spz's packGaussians lays them out (from=UNSPECIFIED)."""
    n = len(cloud)
    # Positions: 24-bit fixed point, millimetre coords * 2^12, little-endian 3 bytes each.
    scale = float(1 << SPZ_FRACTIONAL_BITS)
    fixed = np.round(cloud.means.astype(np.float64) * scale).astype(np.int64)
    fixed &= 0xFFFFFF
    pos = np.zeros((n, 3, 3), dtype=np.uint8)
    pos[:, :, 0] = fixed & 0xFF
    pos[:, :, 1] = (fixed >> 8) & 0xFF
    pos[:, :, 2] = (fixed >> 16) & 0xFF
    positions = pos.reshape(n * 9).tobytes()

    # Alphas: sigmoid(logit) * 255.
    alpha = 1.0 / (1.0 + np.exp(-cloud.opacities.astype(np.float64)))
    alphas = _to_uint8(alpha * 255.0).tobytes()

    # Colours: f_dc -> wide RGB: c*(0.15*255) + 0.5*255, where c = 0.5 + SH_C0*f_dc.
    colour = 0.5 + SH_C0 * cloud.f_dc.astype(np.float64)
    colors = _to_uint8(colour * (SPZ_COLOR_SCALE * 255.0) + 0.5 * 255.0).reshape(n * 3).tobytes()

    # Scales: (log_scale + 10) * 16.
    scales = _to_uint8((cloud.log_scales.astype(np.float64) + 10.0) * 16.0).reshape(n * 3).tobytes()

    # Rotations: PLY [w,x,y,z] -> spz [x,y,z,w], smallest-three.
    quats_xyzw = cloud.quats_wxyz[:, [1, 2, 3, 0]]
    rotations = _pack_quaternion_smallest_three(quats_xyzw).reshape(n * 4).tobytes()

    return [positions, alphas, colors, scales, rotations]


def write_spz(cloud: GaussianCloud, path: Path) -> dict:
    """Write a .spz v4 file (SH degree 0) readable by Spark >= 2.4.0.

    Returns a small manifest: spz version, Spark version, gaussian count, byte size.
    """
    import zstandard as zstd  # optional; installed in dev (listed in the report)

    path = Path(path)
    n = len(cloud)
    streams = pack_spz_streams(cloud)
    cctx = zstd.ZstdCompressor(level=SPZ_ZSTD_LEVEL)
    compressed = [cctx.compress(s) for s in streams]

    # Header: 32 bytes, little-endian (NgspFileHeader).
    magic = 0x5053474E  # "NGSP"
    num_streams = len(streams)
    toc_byte_offset = 32  # no extensions
    header = struct.pack(
        "<IIIBBBBI12s",
        magic,
        SPZ_VERSION,
        n,
        0,                       # shDegree
        SPZ_FRACTIONAL_BITS,     # fractionalBits
        0,                       # flags
        num_streams,             # numStreams
        toc_byte_offset,
        b"\x00" * 12,            # reserved
    )
    # TOC: N x [compressedSize u64, uncompressedSize u64].
    toc = b"".join(
        struct.pack("<QQ", len(compressed[i]), len(streams[i])) for i in range(num_streams)
    )
    body = b"".join(compressed)
    out = header + toc + body
    path.write_bytes(out)
    return {
        "spz_version": SPZ_VERSION,
        "spark_min_version": SPARK_VERSION,
        "gaussians": n,
        "bytes": len(out),
        "pack_from": "UNSPECIFIED",
    }


# Backward-compatible alias; ``read_gaussians_ply`` is the public name for cross-workstream reuse.
read_ply = read_gaussians_ply
