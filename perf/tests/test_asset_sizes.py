"""Asset parsers and the asset-check logic with synthetic files (no real 3D assets needed)."""

import json
import struct

import pytest

import asset_sizes
from asset_sizes import (
    check_scan_assets,
    count_glb_triangles,
    count_ply_vertices,
    count_spz_gaussians,
    findings_to_measured,
)


def make_spz(num_points: int) -> bytes:
    # magic, version=4, numPoints, then padding for the rest of the header fields.
    return struct.pack("<III", asset_sizes.SPZ_MAGIC, 4, num_points) + b"\x00" * 20


def make_ply(num_vertices: int) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_vertices}\n"
        "property float x\n"
        "end_header\n"
    ).encode("ascii")
    return header + b"\x00" * (num_vertices * 4)


def make_glb(triangles: int) -> bytes:
    # One mesh, one indexed primitive, index accessor count = triangles * 3.
    gltf = {
        "asset": {"version": "2.0"},
        "accessors": [
            {"componentType": 5125, "count": triangles * 3, "type": "SCALAR"},  # indices
            {"componentType": 5126, "count": triangles * 3, "type": "VEC3"},    # positions
        ],
        "meshes": [{"primitives": [{"mode": 4, "indices": 0, "attributes": {"POSITION": 1}}]}],
    }
    jchunk = json.dumps(gltf).encode("utf-8")
    pad = (4 - len(jchunk) % 4) % 4
    jchunk += b" " * pad
    header = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jchunk))
    chunk = struct.pack("<II", len(jchunk), 0x4E4F534A) + jchunk  # JSON
    return header + chunk


def test_count_spz():
    assert count_spz_gaussians(make_spz(123456)) == 123456


def test_count_spz_rejects_bad_magic():
    with pytest.raises(ValueError):
        count_spz_gaussians(b"\x00" * 32)


def test_count_ply():
    assert count_ply_vertices(make_ply(5000)) == 5000


def test_count_glb_indexed():
    assert count_glb_triangles(make_glb(40000)) == 40000


def test_count_glb_non_indexed():
    gltf = {
        "asset": {"version": "2.0"},
        "accessors": [{"componentType": 5126, "count": 30, "type": "VEC3"}],
        "meshes": [{"primitives": [{"mode": 4, "attributes": {"POSITION": 0}}]}],
    }
    jchunk = json.dumps(gltf).encode("utf-8")
    jchunk += b" " * ((4 - len(jchunk) % 4) % 4)
    data = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jchunk)) + struct.pack("<II", len(jchunk), 0x4E4F534A) + jchunk
    assert count_glb_triangles(data) == 10  # 30 positions / 3


def test_check_scan_assets_absent_reports_missing():
    recon = {"splat_uri": "x/scene.spz", "mesh_uri": "x/object.glb"}  # no *_mobile_uri
    findings = check_scan_assets("scn_trap", recon, fetch=None)
    assert len(findings) == 2
    assert all(not f.present for f in findings)
    assert all(f.count is None for f in findings)
    assert "absent" in findings[0].detail


def test_check_scan_assets_present_counts_against_budget():
    store = {
        "x/scene_mobile.spz": make_spz(350_000),
        "x/object_mobile.glb": make_glb(45_000),
    }
    recon = {"splat_mobile_uri": "x/scene_mobile.spz", "mesh_mobile_uri": "x/object_mobile.glb"}
    findings = check_scan_assets("scn_trap", recon, fetch=lambda u: store[u])
    counts = findings_to_measured(findings)
    assert counts["assets.splat_gaussians"] == 350_000
    assert counts["assets.mesh_triangles"] == 45_000


def test_check_scan_assets_fetch_error_is_reported_not_raised():
    def boom(_uri):
        raise RuntimeError("signed URL expired")

    recon = {"splat_mobile_uri": "x/scene_mobile.spz"}
    findings = check_scan_assets("scn_trap", recon, fetch=boom)
    splat = [f for f in findings if f.kind == "splat"][0]
    assert splat.present and splat.count is None
    assert "fetch failed" in splat.detail


def test_findings_to_measured_takes_worst_case():
    # Largest count across scans is the ceiling that must pass.
    f1 = asset_sizes.AssetFinding("a", "splat", "u1", True, 100_000, 10)
    f2 = asset_sizes.AssetFinding("b", "splat", "u2", True, 420_000, 10)
    assert findings_to_measured([f1, f2])["assets.splat_gaussians"] == 420_000
