import { describe, it, expect } from "vitest";
import { scanPointToWeb, scanToWebMatrix } from "../lib/transform";

// scan_to_web = Rx(-90 deg), unit_scale = 0.001 (mm -> m). This is exactly the
// matrix the contracts store.
const recon = {
  scan_to_web: {
    matrix: [
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, -1, 0, 0],
      [0, 0, 0, 1],
    ] as unknown as never,
    unit_scale: 0.001,
  },
};

function closeTo(a: number[], b: number[], eps = 1e-9) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThan(eps);
  }
}

describe("scan_to_web transform", () => {
  it("maps scan (0,0,100) mm to web (0, 0.1, 0) m", () => {
    closeTo(scanPointToWeb(recon, [0, 0, 100]), [0, 0.1, 0]);
  });

  it("maps scan (0,100,0) mm to web (0, 0, -0.1) m", () => {
    closeTo(scanPointToWeb(recon, [0, 100, 0]), [0, 0, -0.1]);
  });

  it("maps scan X (100,0,0) mm to web (0.1, 0, 0) m (X unchanged)", () => {
    closeTo(scanPointToWeb(recon, [100, 0, 0]), [0.1, 0, 0]);
  });

  it("the parent matrix is rotation * scale, det +1 before scaling", () => {
    const m = scanToWebMatrix(recon);
    // The full matrix has determinant unit_scale^3 (three uniform scales).
    const det = m.determinant();
    expect(Math.abs(det - Math.pow(0.001, 3))).toBeLessThan(1e-18);
  });
});
