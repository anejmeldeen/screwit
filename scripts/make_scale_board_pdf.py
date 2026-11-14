"""Write web/public/scale-board.pdf from the single board definition (backend/screwit/recon/board.py).

The page is A4 at 300 dpi, so printing at 100% gives 30 mm squares. The script checks its own
output: every inner ChArUco corner must be found by the same detector the scale stage uses, and the
mean corner spacing must be 30 mm at 300 dpi to within 0.5%.

Usage: PYTHONPATH=backend python scripts/make_scale_board_pdf.py [out.pdf]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from screwit.recon import board
from screwit.recon.scale import detect_board_corners

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "web" / "public" / "scale-board.pdf"
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "board.png"
        manifest = board.write_board_png(png)
        img = np.asarray(Image.open(png).convert("RGB"))

        ids, corners = detect_board_corners(img)
        expected = (board.SQUARES_X - 1) * (board.SQUARES_Y - 1)
        if ids is None or len(ids) != expected:
            print(f"detector found {0 if ids is None else len(ids)} of {expected} corners", file=sys.stderr)
            return 1
        # Neighbouring corners along X are one square apart.
        by_id = dict(zip(ids.tolist(), corners, strict=True))
        cols = board.SQUARES_X - 1
        steps = [np.linalg.norm(by_id[i + 1] - by_id[i]) for i in by_id
                 if (i % cols) < cols - 1 and i + 1 in by_id]
        square_px = float(np.mean(steps))
        expected_px = board.SQUARE_MM / 25.4 * board.DPI
        err = abs(square_px - expected_px) / expected_px
        if err > 0.005:
            print(f"square is {square_px:.1f} px, expected {expected_px:.1f} px", file=sys.stderr)
            return 1

        out.parent.mkdir(parents=True, exist_ok=True)
        Image.open(png).convert("RGB").save(out, "PDF", resolution=float(board.DPI))
    w_mm = manifest["page_px"][0] / board.DPI * 25.4
    h_mm = manifest["page_px"][1] / board.DPI * 25.4
    print(f"wrote {out} ({out.stat().st_size} bytes): {manifest['board']}, page {w_mm:.0f}x{h_mm:.0f} mm, "
          f"{expected} corners detected, square {square_px:.1f} px = "
          f"{square_px / board.DPI * 25.4:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
