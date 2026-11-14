"""Printable scale board (SPEC 2.2, 6.1 step 3).

Writes an A4, 300 dpi PNG of a ChArUco 5x7 board with 30 mm squares and a printed arrow marking the
board's +X axis, so a user can print it at 100% and lay it flat next to the object. The same board
geometry is detected in ``scale.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BOARD_NAME = "charuco_5x7_30mm"
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_MM = 30.0
MARKER_RATIO = 0.75
DPI = 300
A4_MM = (210.0, 297.0)
ARUCO_DICT = "DICT_4X4_50"


def _dictionary():
    import cv2

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))


def charuco_board():
    """The cv2 CharucoBoard used both for drawing and detection."""
    import cv2

    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_MM, SQUARE_MM * MARKER_RATIO, _dictionary()
    )


def _mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def write_board_png(path: Path) -> dict:
    """Render the board centred on an A4 page at 300 dpi, with an X-axis arrow, and return a small
    manifest (board name, square size, printed pixel size)."""
    import cv2

    path = Path(path)
    board = charuco_board()
    board_w_px = _mm_to_px(SQUARES_X * SQUARE_MM)
    board_h_px = _mm_to_px(SQUARES_Y * SQUARE_MM)
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0)

    page_w, page_h = _mm_to_px(A4_MM[0]), _mm_to_px(A4_MM[1])
    page = np.full((page_h, page_w), 255, np.uint8)
    ox = (page_w - board_w_px) // 2
    oy = (page_h - board_h_px) // 2
    page[oy:oy + board_h_px, ox:ox + board_w_px] = board_img

    page_rgb = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    # X-axis arrow: starts at the board origin corner (bottom-left in board coords), pointing +X.
    arrow_y = oy + board_h_px + _mm_to_px(12)
    x0 = ox
    x1 = ox + _mm_to_px(SQUARE_MM * 2)
    cv2.arrowedLine(page_rgb, (x0, arrow_y), (x1, arrow_y), (0, 0, 255),
                    thickness=max(2, _mm_to_px(1.0)), tipLength=0.2)
    cv2.putText(page_rgb, "X  (point away from first camera)", (x0, arrow_y - _mm_to_px(4)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(page_rgb, f"{BOARD_NAME}  -- print at 100%, check one 30 mm square with a ruler",
                (ox, oy - _mm_to_px(6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(path), page_rgb)
    return {
        "board": BOARD_NAME, "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
        "square_mm": SQUARE_MM, "dpi": DPI, "page_px": [page_w, page_h],
        "board_px": [board_w_px, board_h_px],
    }
