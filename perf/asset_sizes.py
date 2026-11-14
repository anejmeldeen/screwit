"""Asset size checks (brief step 4).

For each hero scan: read its reconstruction, and if `splat_mobile_uri` / `mesh_mobile_uri`
are present, fetch them and assert gaussian count and triangle count against the budgets;
if absent, report them as MISSING (a FAIL in strict mode, a flag otherwise).

Gaussian count for a `.spz` or `.ply` splat and triangle count for a `.glb` mesh are read from
the file structure without any heavy 3D library:
- `.spz`: the SPZ v1-4 header is `magic (0x5053474e) | version(u32) | numPoints(u32) | ...`,
  little-endian. numPoints is the gaussian count. (SPEC 3.1 pins Spark >= 2.4.0 / spz v4.)
- `.ply`: read the ASCII header, take `element vertex N`; a 3DGS PLY stores one vertex per gaussian.
- `.glb`: parse the binary glTF JSON chunk and sum triangle counts across mesh primitives using
  each primitive's accessor counts (indices if present, else positions), divided by 3.

This module does no network setup of its own; the caller passes a fetch function that returns the
asset bytes for a URI (over the API's signed-asset route in a live run, or from the local fixtures
store / `data/` directory). When no fetcher or no asset is available, counts come back as None and
the budget SKIPs.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass

SPZ_MAGIC = 0x5053474E  # "NGSP" little-endian -> 'N','G','S','P'


@dataclass
class AssetFinding:
    scan_id: str
    kind: str  # "splat" | "mesh"
    uri: str | None
    present: bool
    count: int | None  # gaussians for splat, triangles for mesh
    size_bytes: int | None
    detail: str = ""


def count_spz_gaussians(data: bytes) -> int:
    """Gaussian count from an SPZ file header (versions 1-4 share this prefix)."""
    if len(data) < 12:
        raise ValueError("spz too short for a header")
    magic, _version, num_points = struct.unpack_from("<III", data, 0)
    if magic != SPZ_MAGIC:
        raise ValueError(f"not an spz file (magic {magic:#x})")
    return num_points


def count_ply_vertices(data: bytes) -> int:
    """Vertex count from a PLY header; a 3DGS PLY stores one vertex per gaussian."""
    head = data[:4]
    if head[:3] != b"ply":
        raise ValueError("not a ply file")
    # Header is ASCII, ends at 'end_header'. Read line by line.
    text = data[:65536].split(b"end_header", 1)[0].decode("ascii", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            return int(parts[2])
    raise ValueError("no 'element vertex' in ply header")


def count_glb_triangles(data: bytes) -> int:
    """Triangle count from a binary glTF (.glb): sum over mesh primitives.

    Uses the index accessor count when the primitive is indexed, else the POSITION accessor
    count. Only TRIANGLES mode (4, the glTF default) is counted; other modes are ignored.
    """
    if data[:4] != b"glTF":
        raise ValueError("not a glb file")
    _magic, _version, _length = struct.unpack_from("<III", data, 0)
    # First chunk must be JSON.
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:  # "JSON"
        raise ValueError("first glb chunk is not JSON")
    gltf = json.loads(data[20 : 20 + chunk_len].decode("utf-8"))
    accessors = gltf.get("accessors", [])
    total = 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            mode = prim.get("mode", 4)
            if mode != 4:  # not TRIANGLES
                continue
            if "indices" in prim:
                count = accessors[prim["indices"]]["count"]
            else:
                pos = prim.get("attributes", {}).get("POSITION")
                if pos is None:
                    continue
                count = accessors[pos]["count"]
            total += count // 3
    return total


def _ext_of(uri: str) -> str:
    """Lower-case file extension of a URI, ignoring any ?query or #fragment (signed URLs carry them)."""
    path = uri.split("?", 1)[0].split("#", 1)[0]
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""


def _count_for(kind: str, uri: str, data: bytes) -> int:
    ext = _ext_of(uri)
    if kind == "splat":
        if ext == ".spz":
            return count_spz_gaussians(data)
        if ext == ".ply":
            return count_ply_vertices(data)
        raise ValueError(f"unknown splat extension {ext!r} for {uri}")
    if kind == "mesh":
        if ext == ".glb":
            return count_glb_triangles(data)
        raise ValueError(f"unknown mesh extension {ext!r} for {uri}")
    raise ValueError(f"unknown asset kind {kind}")


def check_scan_assets(
    scan_id: str,
    reconstruction: dict,
    fetch: Callable[[str], bytes] | None,
) -> list[AssetFinding]:
    """Produce findings for the mobile splat and mobile mesh of one scan.

    `reconstruction` is the parsed reconstruction.json (contract 4.2). `fetch(uri) -> bytes`
    resolves an asset URI to bytes; pass None when no asset store is reachable.
    """
    findings: list[AssetFinding] = []
    for kind, field in (("splat", "splat_mobile_uri"), ("mesh", "mesh_mobile_uri")):
        uri = reconstruction.get(field)
        if not uri:
            findings.append(AssetFinding(scan_id, kind, None, present=False, count=None,
                                        size_bytes=None,
                                        detail=f"{field} absent from reconstruction (X5 asset not produced yet)"))
            continue
        if fetch is None:
            findings.append(AssetFinding(scan_id, kind, uri, present=True, count=None,
                                        size_bytes=None, detail="no asset fetcher available"))
            continue
        try:
            data = fetch(uri)
        except Exception as exc:  # noqa: BLE001 — report any fetch error, don't abort the suite
            findings.append(AssetFinding(scan_id, kind, uri, present=True, count=None,
                                        size_bytes=None, detail=f"fetch failed: {exc}"))
            continue
        try:
            count = _count_for(kind, uri, data)
            findings.append(AssetFinding(scan_id, kind, uri, present=True, count=count,
                                        size_bytes=len(data)))
        except Exception as exc:  # noqa: BLE001
            findings.append(AssetFinding(scan_id, kind, uri, present=True, count=None,
                                        size_bytes=len(data), detail=f"parse failed: {exc}"))
    return findings


def findings_to_measured(findings: list[AssetFinding]) -> dict[str, int | None]:
    """Collapse per-scan findings into the worst (largest) count per budget key.

    The budget is a ceiling, so the largest count across hero scans is the one that must pass.
    Returns None for a budget when no scan produced a count.
    """
    splat_counts = [f.count for f in findings if f.kind == "splat" and f.count is not None]
    mesh_counts = [f.count for f in findings if f.kind == "mesh" and f.count is not None]
    return {
        "assets.splat_gaussians": max(splat_counts) if splat_counts else None,
        "assets.mesh_triangles": max(mesh_counts) if mesh_counts else None,
    }
