import { describe, expect, it } from "vitest";
import { partColor } from "../lib/palette";
import palette from "../lib/partPalette.json";

describe("shared muted component palette", () => {
  it("matches each original component to its preview palette slot", () => {
    palette.forEach((color, i) => expect(partColor(0, `p${i + 1}`)).toBe(color));
  });
  it("keeps a component's color when isolated or reordered after combine", () => {
    for (const id of ["p2", "p4", "merged_abc123", "merged_p2_p1"]) {
      expect(partColor(0, id)).toBe(partColor(5, id));
    }
  });
  it("uses muted colors instead of bright or neon defaults", () => {
    for (const color of palette) {
      const rgb = color.slice(1).match(/../g)!.map(c => parseInt(c, 16));
      expect(Math.max(...rgb)).toBeLessThan(190);
      expect(Math.max(...rgb) - Math.min(...rgb)).toBeLessThan(80);
    }
  });
});
