import { describe, expect, it } from "vitest";
import { partColor } from "../lib/palette";
import palette from "../lib/partPalette.json";

describe("shared homepage component palette", () => {
  it("matches each original component to its preview palette slot", () => {
    palette.forEach((color, i) => expect(partColor(0, `p${i + 1}`)).toBe(color));
  });
  it("keeps a component's color when isolated or reordered after combine", () => {
    for (const id of ["p2", "p4", "merged_abc123", "merged_p2_p1"]) {
      expect(partColor(0, id)).toBe(partColor(5, id));
    }
  });
  it("preserves the original homepage colors rather than the neon editor palette", () => {
    expect(palette).toEqual([
      "#c85a46", "#5aa0d2", "#d2be5a", "#78c882", "#b478c8", "#969696",
    ]);
  });
});
