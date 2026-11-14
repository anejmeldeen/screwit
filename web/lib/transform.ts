// The ONE scan-frame -> web-frame transform (SPEC 3.1 "Two frames, one
// transform" and the "Web frame" bullet).
//
// Every asset (splat, part meshes, CAD STL, measure points) is a child of ONE
// parent group whose matrix is:
//   m = Matrix4().set(...16 row-major values of scan_to_web.matrix...)
//         .multiply(Matrix4().makeScale(s, s, s))   where s = unit_scale
// scan_to_web.matrix is Rx(-90 deg) (Z-up mm -> Y-up), unit_scale is 0.001
// (mm -> m). No asset carries reliable units/axes, so this is the only
// conversion; nothing else rotates or scales.

import { Matrix4 } from "three";
import type { Reconstruction } from "./contracts.gen";

type Mat4Rows = Reconstruction["scan_to_web"]["matrix"];

// Flatten a 4x4 row-major nested array into 16 numbers, row-major.
export function flattenRowMajor(rows: Mat4Rows): number[] {
  const out: number[] = [];
  for (const row of rows as unknown as number[][]) {
    for (const v of row) out.push(v as number);
  }
  if (out.length !== 16) {
    throw new Error(`scan_to_web.matrix must be 4x4, got ${out.length} values`);
  }
  return out;
}

// Build the parent-group Matrix4 from a Reconstruction's scan_to_web.
// three's Matrix4.set takes arguments in row-major order, which is exactly how
// the contract stores the matrix, so no transpose is needed before .set().
export function scanToWebMatrix(recon: {
  scan_to_web: { matrix: Mat4Rows; unit_scale: number };
}): Matrix4 {
  const r = flattenRowMajor(recon.scan_to_web.matrix);
  const s = recon.scan_to_web.unit_scale as unknown as number;
  const rot = new Matrix4().set(
    r[0], r[1], r[2], r[3],
    r[4], r[5], r[6], r[7],
    r[8], r[9], r[10], r[11],
    r[12], r[13], r[14], r[15],
  );
  return rot.multiply(new Matrix4().makeScale(s, s, s));
}

// Apply the transform to a scan-frame point, returning the web-frame point.
// Used for tests and for placing measure markers.
export function scanPointToWeb(
  recon: { scan_to_web: { matrix: Mat4Rows; unit_scale: number } },
  p: readonly [number, number, number],
): [number, number, number] {
  const m = scanToWebMatrix(recon);
  const e = m.elements; // column-major in three
  const [x, y, z] = p;
  const w = e[3] * x + e[7] * y + e[11] * z + e[15];
  const wx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
  const wy = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
  const wz = (e[2] * x + e[6] * y + e[10] * z + e[14]) / w;
  return [wx, wy, wz];
}
