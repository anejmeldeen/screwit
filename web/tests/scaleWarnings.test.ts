import { describe, it, expect } from "vitest";
import { scaleWarnings, type ScaleInfo } from "../lib/scaleWarnings";

function scale(partial: Partial<ScaleInfo>): ScaleInfo {
  return { method: "aruco", ...partial } as ScaleInfo;
}

describe("scaleWarnings", () => {
  it("clean aruco scan -> no chip (hero fixtures unchanged)", () => {
    const w = scaleWarnings(
      scale({
        method: "aruco",
        board: "charuco_5x7_30mm",
        corner_views: 41,
        residual_mm: 0.6,
        board_tilt_deg: 2.1,
      }),
    );
    expect(w).toEqual([]);
  });

  it("learned method -> approximate warning", () => {
    const w = scaleWarnings(scale({ method: "learned" }));
    expect(w).toHaveLength(1);
    expect(w[0]).toMatchObject({
      id: "learned",
      level: "warning",
      message: "Scale is approximate (no board found)",
    });
  });

  it("manual method -> info chip", () => {
    const w = scaleWarnings(scale({ method: "manual" }));
    expect(w).toHaveLength(1);
    expect(w[0]).toMatchObject({
      id: "manual",
      level: "info",
      message: "Scale set by hand",
    });
  });

  it("residual_mm > 2 -> board-fit warning with the value", () => {
    const w = scaleWarnings(scale({ residual_mm: 45.6, corner_views: 41, board_tilt_deg: 1 }));
    const r = w.find((x) => x.id === "residual")!;
    expect(r.level).toBe("warning");
    expect(r.message).toBe(
      "Board fit is off by 45.6 mm: re-scan with the board flat and in view",
    );
  });

  it("residual integer renders with no decimal", () => {
    const w = scaleWarnings(scale({ residual_mm: 5 }));
    expect(w.find((x) => x.id === "residual")!.message).toContain("off by 5 mm");
  });

  it("residual exactly at the limit (2) does NOT warn", () => {
    expect(scaleWarnings(scale({ residual_mm: 2 }))).toEqual([]);
  });

  it("board_tilt_deg > 10 -> lay flat warning", () => {
    const w = scaleWarnings(scale({ board_tilt_deg: 11 }));
    expect(w).toEqual([
      { id: "tilt", level: "warning", message: "Lay the board flat next to the object" },
    ]);
  });

  it("board_tilt_deg exactly 10 does NOT warn", () => {
    expect(scaleWarnings(scale({ board_tilt_deg: 10 }))).toEqual([]);
  });

  it("few board views do NOT warn (the view count is not a rule)", () => {
    expect(scaleWarnings(scale({ corner_views: 2 }))).toEqual([]);
    expect(scaleWarnings(scale({ corner_views: 8 }))).toEqual([]);
  });

  it("several rules can show at once (the real 45.6 mm / 11 deg case)", () => {
    const w = scaleWarnings(
      scale({ method: "aruco", residual_mm: 45.6, board_tilt_deg: 11, corner_views: 6 }),
    );
    expect(w.map((x) => x.id)).toEqual(["residual", "tilt"]);
  });

  it("learned plus a bad fit shows the method chip first", () => {
    const w = scaleWarnings(scale({ method: "learned", residual_mm: 3 }));
    expect(w.map((x) => x.id)).toEqual(["learned", "residual"]);
  });

  it("null / missing fields do not trip any rule", () => {
    expect(
      scaleWarnings(
        scale({ residual_mm: null, board_tilt_deg: null, corner_views: null }),
      ),
    ).toEqual([]);
    expect(scaleWarnings(null)).toEqual([]);
    expect(scaleWarnings(undefined)).toEqual([]);
  });
});
