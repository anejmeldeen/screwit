"""The printable scale board renders an A4 300-dpi PNG and is detectable by the scale stage."""

from __future__ import annotations

import numpy as np

from screwit.recon import board as B
from screwit.recon import scale as S


def test_write_board_png_is_a4_300dpi(tmp_path):
    path = tmp_path / "board.png"
    manifest = B.write_board_png(path)
    assert path.exists()
    assert manifest["board"] == "charuco_5x7_30mm"
    assert manifest["dpi"] == 300
    # A4 at 300 dpi.
    assert manifest["page_px"][0] == round(210 / 25.4 * 300)
    assert manifest["page_px"][1] == round(297 / 25.4 * 300)


def test_printed_board_is_detectable(tmp_path):
    """Render the printable board, then detect its ChArUco corners -- the X arrow and caption text
    must not break detection."""
    from PIL import Image

    path = tmp_path / "board.png"
    B.write_board_png(path)
    img = np.asarray(Image.open(path).convert("RGB"))
    ids, corners = S.detect_board_corners(img)
    assert ids is not None
    # A 5x7 ChArUco board has 4x6 = 24 inner corners; expect most of them.
    assert len(ids) >= 16
    assert corners.shape[1] == 2


def test_board_object_points_count():
    pts = S.board_object_points()
    assert pts.shape == (24, 3)
    assert np.allclose(pts[:, 2], 0.0)  # board lies in Z=0
