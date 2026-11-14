"""End-to-end CPU pipeline: fake model + synthetic board scene -> reconstruction.json validates,
the known object measures within 2%, cameras.json validates, the splat decodes, mobile variants are
wired conditionally, and progress messages carry per-stage wall times."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from screwit.contracts import Cameras, Reconstruction, Scan, ScanInput, Stage
from screwit.contracts.models import STAGE_ORDER
from screwit.recon.geometry import FakeGeometryModel
from screwit.recon.pipeline import ReconPipelineImpl

from ._spz_decoder import ensure_decoder


def _scan(scan_id="scn_e2e"):
    stages = [Stage(name=n, state="pending") for n in STAGE_ORDER]
    return Scan(scan_id=scan_id, created_at=datetime.now(UTC),
                input=ScanInput(type="video", uri=f"scans/{scan_id}/input/video.mp4",
                                duration_s=30.0, container="mp4", codec="h264", hdr=False),
                status="processing", reconstruction_revision=1, stages=stages)


def _pipeline(quirk=1.0, n_views=12):
    fake = FakeGeometryModel(n_views=n_views, gaussian_quirk=quirk)
    pipe = ReconPipelineImpl(geometry_model=fake, gaussian_cap=1_500_000)
    # Render the synthetic scene into the frames stage instead of decoding a video.
    pipe._frames_provider = fake.render_frames
    return pipe, fake


@pytest.mark.parametrize("quirk", [1.0, 1.7])
def test_end_to_end_reconstruction(storage, perception, progress, quirk):
    pipe, fake = _pipeline(quirk=quirk)
    scan = _scan()
    pipe.run(scan, storage, perception, progress)

    # reconstruction.json validates against the contract.
    raw = storage.get_bytes("scans/scn_e2e/reconstruction.json")
    recon = Reconstruction.model_validate_json(raw)
    assert recon.scan_id == "scn_e2e"
    assert recon.frame_origin == "board"
    assert recon.scale.method == "aruco"
    assert recon.splat_uri == "scans/scn_e2e/splat/scene.spz"
    assert recon.quality.gaussians and recon.quality.gaussians > 0

    # cameras.json validates.
    cams = Cameras.model_validate_json(storage.get_bytes("scans/scn_e2e/geometry/cameras.json"))
    assert len(cams.frames) == len(fake._object_gaussians_scan().means) or len(cams.frames) > 0

    # Known object measures within 2%: the synthetic object is 80 x 80 x 100 mm. Measure from the
    # object bbox the pipeline recorded.
    ext = np.array(recon.object_bbox_mm.max) - np.array(recon.object_bbox_mm.min)
    # The board+object scene; object extent dominates in Z (height 100 mm).
    assert abs(ext[2] - 100.0) <= 2.0, ext

    # Splat decodes with the independent C++ decoder (if available).
    if ensure_decoder() is not None:
        import tempfile
        from pathlib import Path

        from ._spz_decoder import decode
        tmp = Path(tempfile.mkdtemp()) / "scene.spz"
        tmp.write_bytes(storage.get_bytes("scans/scn_e2e/splat/scene.spz"))
        dec = decode(tmp, 1)
        assert dec["num_points"] == recon.quality.gaussians


def test_progress_messages_carry_wall_times(storage, perception, progress):
    pipe, _ = _pipeline()
    pipe.run(_scan(), storage, perception, progress)
    done = {e[0]: e[3] for e in progress.events if e[1] == "done"}
    # Every stage reports and its done message carries a wall-clock time like '... (0.42s)'.
    import re
    for stage in STAGE_ORDER:
        assert stage in done
        assert re.search(r"\d+\.\d+s\)?$", done[stage].rstrip()), (stage, done[stage])


def test_all_stage_transitions_reported(storage, perception, progress):
    pipe, _ = _pipeline()
    pipe.run(_scan(), storage, perception, progress)
    for stage in STAGE_ORDER:
        states = [e[1] for e in progress.events if e[0] == stage]
        assert "running" in states and "done" in states


def test_mobile_variants_from_real_optimizer(storage, perception, progress, tmp_path):
    """The X5 optimizer runs on the pipeline's own full assets: a meshopt GLB within the phone
    triangle budget and a phone .spz within the gaussian budget, both uploaded and referenced."""
    import trimesh

    from screwit.optimize import MAX_MOBILE_GAUSSIANS, MAX_MOBILE_TRIANGLES
    from screwit.optimize.mesh import decompress_glb

    from ._spz_decoder import decode

    pipe, _ = _pipeline()
    pipe.run(_scan(), storage, perception, progress)
    recon = Reconstruction.model_validate_json(storage.get_bytes("scans/scn_e2e/reconstruction.json"))
    assert recon.mesh_mobile_uri == "scans/scn_e2e/mobile/object.glb"
    assert recon.splat_mobile_uri == "scans/scn_e2e/mobile/scene.spz"

    glb = tmp_path / "m.glb"
    glb.write_bytes(storage.get_bytes(recon.mesh_mobile_uri))
    assert b"EXT_meshopt_compression" in glb.read_bytes()
    plain = tmp_path / "plain.glb"
    decompress_glb(glb, plain)
    mesh = trimesh.load(plain, force="mesh")
    assert 0 < len(mesh.faces) <= MAX_MOBILE_TRIANGLES

    spz = tmp_path / "m.spz"
    spz.write_bytes(storage.get_bytes(recon.splat_mobile_uri))
    if ensure_decoder() is None:
        pytest.skip("independent spz decoder unavailable")
    assert 0 < decode(spz)["num_points"] <= MAX_MOBILE_GAUSSIANS


def test_mobile_variant_failure_is_not_fatal(storage, perception, progress, monkeypatch):
    """If the optimizer fails (e.g. gltfpack missing) the scan still completes on full assets."""
    import screwit.optimize as opt
    from screwit.optimize.mesh import GltfpackNotFound

    def boom(*a, **k):
        raise GltfpackNotFound("gltfpack not found")

    monkeypatch.setattr(opt, "make_mobile_variants", boom)
    pipe, _ = _pipeline()
    pipe.run(_scan(), storage, perception, progress)
    recon = Reconstruction.model_validate_json(storage.get_bytes("scans/scn_e2e/reconstruction.json"))
    assert recon.mesh_mobile_uri is None
    assert recon.splat_mobile_uri is None
    assert storage.exists(recon.mesh_uri)


def test_rescale_analytic_and_stale(storage, perception, progress):
    from screwit.contracts import ScaleRequest
    from screwit.interfaces import StaleRevision

    pipe, _ = _pipeline()
    pipe.run(_scan(), storage, perception, progress)
    recon0 = Reconstruction.model_validate_json(storage.get_bytes("scans/scn_e2e/reconstruction.json"))

    # Rescale so a 50 mm span reads as 100 mm (factor 2).
    req = ScaleRequest(point_a_mm=(0, 0, 0), point_b_mm=(0, 0, 50.0), known_length_mm=100.0,
                       expected_revision=recon0.reconstruction_revision)
    recon1 = pipe.rescale("scn_e2e", req, storage, perception)
    assert recon1.reconstruction_revision == recon0.reconstruction_revision + 1
    assert recon1.scale.method == "manual"
    # bbox doubled.
    assert np.allclose(np.array(recon1.object_bbox_mm.max),
                       np.array(recon0.object_bbox_mm.max) * 2.0)
    # Perception rescale was called with the factor and new revision.
    assert perception.rescale_calls[-1] == ("scn_e2e", 2.0, recon1.reconstruction_revision)

    # Stale expected_revision is rejected.
    bad = ScaleRequest(point_a_mm=(0, 0, 0), point_b_mm=(0, 0, 50.0), known_length_mm=100.0,
                       expected_revision=0)
    with pytest.raises(StaleRevision):
        pipe.rescale("scn_e2e", bad, storage, perception)


def _splat_message(progress):
    return next(m for (stage, state, _p, m) in progress.events if stage == "splat" and state == "done")


def test_splat_uses_the_head_when_it_covers_the_object(storage, perception, progress):
    pipe, _ = _pipeline()
    pipe.run(_scan(), storage, perception, progress)
    assert "from gaussian head" in _splat_message(progress)


def test_splat_falls_back_to_depth_when_head_misses_the_object(storage, perception, progress):
    """A head whose global fit passes but whose gaussians land outside the object box (seen on a
    real CPU portrait scan) must not ship a near-empty splat."""
    from screwit.recon import geometry as GE

    pipe, fake = _pipeline()
    real_run = fake.run

    def shifted_run(frames_dir, names):
        res = real_run(frames_dir, names)
        g = res.gaussians
        far = g.means.copy()
        far[: int(0.99 * len(far))] += np.float32(5000.0)  # 99% of gaussians moved far away
        res.gaussians = type(g)(means=far, log_scales=g.log_scales, quats_wxyz=g.quats_wxyz,
                                opacities=g.opacities, f_dc=g.f_dc)
        return res

    fake.run = shifted_run
    real_align = GE.align_gaussians_to_depth

    def passing_align(result, **kw):  # pretend the global similarity fit passed
        return result.gaussians, GE.GaussianAlignment(scale=1.0, rmse_mm=0.0, inlier_fraction=0.9,
                                                      applied=True, relative_rmse=0.0)

    GE.align_gaussians_to_depth = passing_align
    try:
        pipe.run(_scan(), storage, perception, progress)
    finally:
        GE.align_gaussians_to_depth = real_align
    msg = _splat_message(progress)
    assert "from depth maps (gaussian head left" in msg
    recon = Reconstruction.model_validate_json(storage.get_bytes("scans/scn_e2e/reconstruction.json"))
    assert recon.quality.gaussians > 1000
