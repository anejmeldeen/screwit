// Scale-trust warnings derived from reconstruction.scale (SPEC 6.1 step 3 and
// the Reconstruction contract). Pure function: given a ScaleInfo it returns the
// chips to show in the status bar. A clean aruco scan returns [] (no chip), so
// the hero fixtures render unchanged.

import type { Reconstruction } from "./contracts.gen";

export type ScaleInfo = Reconstruction["scale"];

export type ScaleWarningLevel = "warning" | "info";

export interface ScaleWarning {
  // Stable id per rule (handy for keys/tests).
  id: "learned" | "residual" | "tilt" | "corner_views" | "manual";
  level: ScaleWarningLevel;
  message: string;
}

// Thresholds from the brief / SPEC 6.1 step 3.
export const RESIDUAL_MM_LIMIT = 2;
export const BOARD_TILT_DEG_LIMIT = 10;
export const MIN_CORNER_VIEWS = 8;

// Returns every applicable warning; several can hold at once. Order is stable:
// method-level first, then fit-quality rules.
export function scaleWarnings(scale: ScaleInfo | null | undefined): ScaleWarning[] {
  if (!scale) return [];
  const out: ScaleWarning[] = [];

  if (scale.method === "learned") {
    out.push({
      id: "learned",
      level: "warning",
      message: "Scale is approximate (no board found)",
    });
  } else if (scale.method === "manual") {
    out.push({
      id: "manual",
      level: "info",
      message: "Scale set by hand",
    });
  }

  // Fit-quality rules apply to a board (aruco) scale. A null field means "not
  // reported" and does not trip the rule.
  const residual = scale.residual_mm;
  if (residual != null && residual > RESIDUAL_MM_LIMIT) {
    out.push({
      id: "residual",
      level: "warning",
      message: `Board fit is off by ${formatMm(residual)} mm: re-scan with the board flat and in view`,
    });
  }

  const tilt = scale.board_tilt_deg;
  if (tilt != null && tilt > BOARD_TILT_DEG_LIMIT) {
    out.push({
      id: "tilt",
      level: "warning",
      message: "Lay the board flat next to the object",
    });
  }

  const views = scale.corner_views;
  if (views != null && views < MIN_CORNER_VIEWS) {
    out.push({
      id: "corner_views",
      level: "warning",
      message: "Board seen in too few views",
    });
  }

  return out;
}

// Render mm with at most one decimal, dropping a trailing ".0".
function formatMm(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}
