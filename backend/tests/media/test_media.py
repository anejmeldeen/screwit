"""Tests for the X2 test-media fixtures and their deterministic generator.

These run against the committed ``fixtures/media/`` and re-probe every file to
assert it really has the properties the manifest claims (nothing is trusted
from the manifest alone). A second, slower test regenerates the whole set into
a temporary directory and checks the generator is deterministic: the
non-encoded artifacts (cameras.json, photos) hash-match byte for byte, and the
encoded videos reproduce identical probed properties.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/make_test_media.py importable.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import make_test_media as mtm  # noqa: E402

MEDIA_DIR = REPO / "fixtures" / "media"
MANIFEST_PATH = MEDIA_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), "fixtures/media/manifest.json missing; run scripts/make_test_media.py"
    return json.loads(MANIFEST_PATH.read_text())


# --------------------------------------------------------------------------
# Manifest / fixture integrity.
# --------------------------------------------------------------------------


def test_all_files_present_and_hash_matches(manifest):
    for entry in manifest["files"]:
        p = MEDIA_DIR / entry["name"]
        assert p.exists(), f"missing media file {entry['name']}"
        assert p.stat().st_size == entry["size_bytes"], entry["name"]
        assert mtm.sha256_file(p) == entry["sha256"], f"hash drift in {entry['name']}"


def test_photos_present_and_hash_matches(manifest):
    assert len(manifest["photos"]) == 24
    for entry in manifest["photos"]:
        p = MEDIA_DIR / entry["name"]
        assert p.exists(), f"missing photo {entry['name']}"
        assert mtm.sha256_file(p) == entry["sha256"], f"hash drift in {entry['name']}"


def test_total_under_60mb(manifest):
    assert manifest["total_bytes"] < 60 * 1024 * 1024, manifest["total_mb"]


def test_expected_file_set(manifest):
    names = {e["name"] for e in manifest["files"]}
    assert names == {
        "iphone_hevc_hlg_portrait.mov",
        "android_h264.mp4",
        "desktop_vp9.webm",
        "too_short_6s.mp4",
        "truncated_android.mp4",
        "nonallowed_mpeg4.mp4",
        "fake_text.mp4",
    }


# --------------------------------------------------------------------------
# Re-probe every encoded file and assert the probed properties match the
# manifest's declared values (never trust the declared property alone).
# --------------------------------------------------------------------------


def _entry(manifest, name):
    return next(e for e in manifest["files"] if e["name"] == name)


def test_iphone_mov_properties(manifest):
    e = _entry(manifest, "iphone_hevc_hlg_portrait.mov")
    pr = mtm.probe(MEDIA_DIR / e["name"])
    assert pr["codec"] == "hevc"
    assert pr["pix_fmt"] == "yuv420p10le"
    assert pr["color_tags"] is not None
    assert "bt2020" in pr["color_tags"]
    assert "arib-std-b67" in pr["color_tags"]  # HLG transfer
    assert pr["width"] == 1920 and pr["height"] == 1080  # STORED landscape samples
    assert pr["rotation_deg"] == 90.0  # display matrix -> decoders show 1080x1920 portrait
    assert abs(pr["duration_s"] - 20.0) < 0.5
    assert pr["decode_ok"] is True


def test_android_mp4_properties(manifest):
    e = _entry(manifest, "android_h264.mp4")
    pr = mtm.probe(MEDIA_DIR / e["name"])
    assert pr["codec"] == "h264"
    assert pr["pix_fmt"] == "yuv420p"
    assert pr["width"] == 1920 and pr["height"] == 1080
    assert pr["rotation_deg"] == 0.0
    assert abs(pr["duration_s"] - 25.0) < 0.5
    assert pr["decode_ok"] is True


def test_webm_vp9_properties(manifest):
    e = _entry(manifest, "desktop_vp9.webm")
    pr = mtm.probe(MEDIA_DIR / e["name"])
    assert pr["codec"] == "vp9"
    assert pr["width"] == 1280 and pr["height"] == 720
    assert abs(pr["duration_s"] - 15.0) < 0.5
    assert pr["decode_ok"] is True


def test_too_short_clip(manifest):
    e = _entry(manifest, "too_short_6s.mp4")
    pr = mtm.probe(MEDIA_DIR / e["name"])
    assert pr["codec"] == "h264"
    assert abs(pr["duration_s"] - 6.0) < 0.5
    assert pr["duration_s"] < mtm.DURATION_MIN_S
    assert e["expected_validator"]["verdict"] == "reject"
    assert "too_short" in e["expected_validator"]["reasons"]


def test_truncated_file_fails_decode(manifest):
    e = _entry(manifest, "truncated_android.mp4")
    pr = mtm.probe(MEDIA_DIR / e["name"])
    assert pr["decode_ok"] is False, "truncated file must not fully decode"
    assert e["expected_validator"]["verdict"] == "reject"
    assert "truncated_or_undecodable" in e["expected_validator"]["reasons"]
    # It really is a prefix of the full android file.
    full = (MEDIA_DIR / "android_h264.mp4").read_bytes()
    trunc = (MEDIA_DIR / e["name"]).read_bytes()
    assert len(trunc) < len(full)
    assert full[: len(trunc)] == trunc


def test_nonallowlisted_codec(manifest):
    e = _entry(manifest, "nonallowed_mpeg4.mp4")
    pr = mtm.probe(MEDIA_DIR / e["name"])
    assert pr["codec"] == "mpeg4"
    assert pr["codec"] not in mtm.ALLOWLISTED_CODECS
    assert e["expected_validator"]["verdict"] == "reject"
    assert "codec_not_allowed" in e["expected_validator"]["reasons"]


def test_fake_text_mp4(manifest):
    e = _entry(manifest, "fake_text.mp4")
    p = MEDIA_DIR / e["name"]
    head = p.read_bytes()[:16]
    # A real MP4/MOV begins with an 'ftyp'/size box; this is ASCII text.
    assert b"ftyp" not in head
    assert e["expected_validator"]["verdict"] == "reject"
    assert "not_media_container" in e["expected_validator"]["reasons"]


# --------------------------------------------------------------------------
# Photos: EXIF orientation really written, mix of 6/8/normal and portrait.
# --------------------------------------------------------------------------


def test_photo_exif_orientations_written(manifest):
    import piexif

    counts = {1: 0, 6: 0, 8: 0}
    for e in manifest["photos"]:
        p = MEDIA_DIR / e["name"]
        ex = piexif.load(str(p))
        orient = ex["0th"].get(piexif.ImageIFD.Orientation)
        assert orient == e["exif_orientation"], f"{e['name']} EXIF mismatch"
        counts[orient] = counts.get(orient, 0) + 1
    # Some with EXIF orientation 6 and 8, per the brief.
    assert counts[6] >= 1 and counts[8] >= 1
    assert counts[1] >= 1  # and some normal


def test_photos_mixed_portrait_landscape(manifest):
    portrait = sum(1 for e in manifest["photos"] if e["orientation_is_portrait_after_exif"])
    landscape = len(manifest["photos"]) - portrait
    assert portrait >= 1 and landscape >= 1


# --------------------------------------------------------------------------
# Board ground truth: the ChArUco board is really detectable and the stored
# cameras reproject the detected corners to sub-pixel accuracy.
# --------------------------------------------------------------------------


def test_board_detection_ground_truth(manifest):
    bd = manifest["ground_truth"]["board_detection"]
    assert bd["frames_with_8plus_corners"] >= 24, "board should be detectable in many frames"
    assert bd["reproj_px_max"] is not None
    assert bd["reproj_px_max"] < 3.0, "ground-truth cameras must reproject the board tightly"


def test_expected_keyframe_ranges(manifest):
    # The three good orbit videos declare a keyframe range matching the
    # pipeline's 48-64 target (webm is 15 s so its floor is lower).
    for name in ("iphone_hevc_hlg_portrait.mov", "android_h264.mp4"):
        e = _entry(manifest, name)
        lo, hi = e["expected_keyframe_count_range"]
        assert lo <= 64 and hi >= 48


# --------------------------------------------------------------------------
# iPhone portrait clip orientation, decoded through P2's real frames.py so
# ffmpeg autorotation is exercised exactly as the pipeline does it.
# --------------------------------------------------------------------------


def _project(obj, w2c, k):
    import numpy as np

    homo = np.concatenate([obj, np.ones((len(obj), 1))], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = cam[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = (k @ (cam / z).T).T[:, :2]
    pix[z[:, 0] <= 1e-6] = float("nan")
    return pix


def _iphone_gt(manifest):
    import numpy as np

    e = _entry(manifest, "iphone_hevc_hlg_portrait.mov")
    gt = e["ground_truth"]
    k = np.array(gt["camera_K_display"])
    cams = [np.array(c) for c in gt["cameras_world_to_camera_display"]]
    cw = np.array(gt["charuco_corners_scan_mm"])
    id2idx = {cid: i for i, cid in enumerate(gt["charuco_corner_ids"])}
    return gt, k, cams, cw, id2idx


def _decode_iphone_keyframes(src, out_dir):
    """Extract keyframes via P2's real path (autorotation applied)."""
    from screwit.recon import frames as frames_mod

    return frames_mod.extract_video_frames(Path(src), Path(out_dir))


def _board_reprojection(frame_paths, k, cams, cw, id2idx):
    """For each frame, detect the board and find the best-fitting GT camera.

    Returns (per_frame_best_reproj, best_cams). The correct-orientation clip has
    a camera that reprojects the detected corners to sub-pixel error; a sideways
    or mirrored clip has none below a couple of pixels.
    """
    import numpy as np
    from PIL import Image

    from screwit.recon import scale as scale_mod

    reproj = []
    best_cams = []
    for p in frame_paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        cids, px = scale_mod.detect_board_corners(img)
        if cids is None or len(cids) < 8:
            continue
        obj = np.array([cw[id2idx[int(c)]] for c in cids])
        best = np.inf
        best_cam = None
        for w2c in cams:
            pr = _project(obj, w2c, k)
            mn = float(np.nanmean(np.linalg.norm(pr - px, axis=1)))
            if mn < best:
                best = mn
                best_cam = w2c
        reproj.append(best)
        best_cams.append(best_cam)
    return reproj, best_cams


def test_iphone_keyframes_are_portrait(manifest, tmp_path):
    from PIL import Image

    src = MEDIA_DIR / "iphone_hevc_hlg_portrait.mov"
    idx = _decode_iphone_keyframes(src, tmp_path)
    assert 48 <= idx.kept <= 64
    for fe in idx.frames[:5]:
        w, h = Image.open(tmp_path / fe.frame).size
        assert (w, h) == (1080, 1920), f"keyframe {fe.frame} is {w}x{h}, expected 1080x1920"


def test_iphone_board_reprojects_with_portrait_ground_truth(manifest, tmp_path):
    import numpy as np

    _gt, k, cams, cw, id2idx = _iphone_gt(manifest)
    idx = _decode_iphone_keyframes(MEDIA_DIR / "iphone_hevc_hlg_portrait.mov", tmp_path)
    frame_paths = [tmp_path / fe.frame for fe in idx.frames]
    reproj, _ = _board_reprojection(frame_paths, k, cams, cw, id2idx)
    under_2 = [r for r in reproj if r < 2.0]
    assert len(under_2) >= 24, (
        f"only {len(under_2)} frames reproject under 2 px "
        f"(detected board in {len(reproj)}); a sideways/mirrored clip fails here"
    )
    assert float(np.mean(reproj)) < 2.0


def test_iphone_gravity_points_up(manifest, tmp_path):
    import numpy as np
    from PIL import Image

    from screwit.recon import scale as scale_mod

    gt, k, cams, cw, id2idx = _iphone_gt(manifest)
    cube = manifest["scene"]["objects"]["cube"]
    cube_edge = manifest["scene"]["known_object_size_mm"]["cube_edge"]
    cube_c = np.array([cube["center_xy_mm"][0], cube["center_xy_mm"][1], cube_edge / 2.0])

    idx = _decode_iphone_keyframes(MEDIA_DIR / "iphone_hevc_hlg_portrait.mov", tmp_path)
    checked = 0
    for fe in idx.frames:
        img = np.asarray(Image.open(tmp_path / fe.frame).convert("RGB"))
        cids, px = scale_mod.detect_board_corners(img)
        if cids is None or len(cids) < 8:
            continue
        obj = np.array([cw[id2idx[int(c)]] for c in cids])
        best = np.inf
        best_cam = None
        for w2c in cams:
            pr = _project(obj, w2c, k)
            mn = float(np.nanmean(np.linalg.norm(pr - px, axis=1)))
            if mn < best:
                best, best_cam = mn, w2c
        if best >= 2.0:
            continue  # not a confident pose; the reprojection test already guards this
        base = _project(cube_c.reshape(1, 3), best_cam, k)[0]
        up = _project((cube_c + np.array([0, 0, 20.0])).reshape(1, 3), best_cam, k)[0]
        if np.isnan(base).any() or np.isnan(up).any():
            continue
        checked += 1
        # World +Z must project UP the image: smaller v (row) than the base point.
        assert up[1] < base[1], f"gravity points down in {fe.frame} (v {up[1]:.1f} >= {base[1]:.1f})"
    assert checked >= 24, f"only checked gravity in {checked} frames"


def test_old_sideways_clip_fails_reprojection(manifest, tmp_path):
    """Negative control: the previous bug -- an UPRIGHT LANDSCAPE render with a
    90 deg tag stamped on -- decodes sideways, so no GT camera reprojects the
    board under 2 px. Regenerate that clip into tmp and confirm it fails the
    orientation check that the fixed clip passes. Nothing in the repo changes."""
    _gt, k, cams, cw, id2idx = _iphone_gt(manifest)

    # Build the OLD clip: encode upright landscape frames, then stamp rotation 90.
    rs = mtm.render_orbit(mtm.BASE_FRAMES)
    display_frames = mtm._fill_frames(rs.frames, int(20 * 30))
    bad_tmp = tmp_path / "old_sideways_src.mov"
    mtm.encode_from_frames(
        display_frames, 30, bad_tmp,
        vcodec_args=["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30"],
        out_w=1920, out_h=1080,
        container_args=["-movflags", "+faststart", "-f", "mov"],
    )
    bad_final = tmp_path / "old_sideways.mov"
    mtm.apply_display_rotation(bad_tmp, bad_final, 90)

    out = tmp_path / "old_frames"
    idx = _decode_iphone_keyframes(bad_final, out)
    frame_paths = [out / fe.frame for fe in idx.frames]
    reproj, _ = _board_reprojection(frame_paths, k, cams, cw, id2idx)
    under_2 = [r for r in reproj if r < 2.0]
    # Portrait GT must NOT fit the sideways clip: essentially no frame under 2 px.
    assert len(under_2) < 24, (
        f"the sideways clip unexpectedly reprojected under 2 px in {len(under_2)} frames; "
        "the negative control is not exercising orientation"
    )


# --------------------------------------------------------------------------
# Determinism: regenerate into a temp dir and compare.
# --------------------------------------------------------------------------


def test_generator_is_deterministic(tmp_path):
    """Regenerate the whole media set and verify reproducibility.

    This runs the full generator (CPU render + ffmpeg encodes, ~1 min). It is
    the brief's required determinism check, so it runs as part of the suite.
    """
    out = tmp_path / "media"
    manifest2 = mtm.build(out)
    committed = json.loads(MANIFEST_PATH.read_text())

    # Non-encoded artifacts must be byte-identical (hash match): cameras.json
    # and every photo. These are pure numpy/PIL output with a fixed seed.
    cam1 = (MEDIA_DIR / "cameras.json").read_bytes()
    cam2 = (out / "cameras.json").read_bytes()
    assert cam1 == cam2, "cameras.json is not reproducible"

    photos1 = {e["name"]: e["sha256"] for e in committed["photos"]}
    photos2 = {e["name"]: e["sha256"] for e in manifest2["photos"]}
    assert photos1 == photos2, "photo hashes are not reproducible"

    # Ground-truth block (cameras, charuco corners, board poses) must match.
    assert committed["ground_truth"]["cameras_world_to_camera"] == \
        manifest2["ground_truth"]["cameras_world_to_camera"]
    assert committed["ground_truth"]["charuco_corners_scan_mm"] == \
        manifest2["ground_truth"]["charuco_corners_scan_mm"]

    # The iPhone clip's per-file portrait ground truth must also reproduce.
    e1 = _entry(committed, "iphone_hevc_hlg_portrait.mov")["ground_truth"]
    e2 = _entry(manifest2, "iphone_hevc_hlg_portrait.mov")["ground_truth"]
    assert e1["camera_K_display"] == e2["camera_K_display"]
    assert e1["cameras_world_to_camera_display"] == e2["cameras_world_to_camera_display"]
    assert e1["displayed_resolution"] == e2["displayed_resolution"] == [1080, 1920]

    # Encoded videos: codecs/containers are not guaranteed bit-identical across
    # runs, but their probed properties must reproduce exactly.
    for name in ("iphone_hevc_hlg_portrait.mov", "android_h264.mp4", "desktop_vp9.webm"):
        p1 = mtm.probe(MEDIA_DIR / name)
        p2 = mtm.probe(out / name)
        for key in ("codec", "pix_fmt", "width", "height", "rotation_deg", "decode_ok"):
            assert p1[key] == p2[key], f"{name}: {key} not reproducible ({p1[key]} != {p2[key]})"
        assert abs(p1["duration_s"] - p2["duration_s"]) < 0.1
