import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Part } from "../lib/contracts.gen";
import {
  preferredSource,
  preferredValueMm,
  resolveDimension,
  resolveAllDimensions,
  editableMeasurements,
} from "../lib/parts";

const fixtures = resolve(__dirname, "..", "..", "fixtures", "scans");
const trapParts = JSON.parse(
  readFileSync(resolve(fixtures, "scn_trap", "parts.json"), "utf8"),
) as Part[];
const knobParts = JSON.parse(
  readFileSync(resolve(fixtures, "scn_knob", "parts.json"), "utf8"),
) as Part[];

function byId(parts: Part[], id: string): Part {
  const p = parts.find((x) => x.part_id === id);
  if (!p) throw new Error(`part ${id} not found`);
  return p;
}

describe("DerivedDimension resolution", () => {
  it("resolves outer_diameter = 2 * caliper radius for the tailpiece p1", () => {
    const p1 = byId(trapParts, "p1");
    // p1_f1.radius: scan 18.6, snapped 19.05, caliper 19.05, preferred caliper.
    const dims = resolveAllDimensions(p1);
    const od = dims.find((d) => d.name === "outer_diameter")!;
    expect(od.featureId).toBe("p1_f1");
    expect(od.field).toBe("radius");
    expect(od.factor).toBe(2);
    expect(od.preferred).toBe("caliper");
    // 2 * 19.05 = 38.1
    expect(od.preferredMm).toBeCloseTo(38.1, 6);
    expect(od.scanMm).toBeCloseTo(37.2, 6);
    expect(od.snappedMm).toBeCloseTo(38.1, 6);
    expect(od.caliperMm).toBeCloseTo(38.1, 6);
  });

  it("resolves a derived dimension with no factor (length) as factor 1", () => {
    const p1 = byId(trapParts, "p1");
    const len = resolveAllDimensions(p1).find((d) => d.name === "length")!;
    expect(len.factor).toBe(1);
    expect(len.preferredMm).toBeCloseTo(138.0, 6);
  });

  it("resolves an OwnDimension (slip nut height) from its own value", () => {
    const p3 = byId(trapParts, "p3");
    const h = resolveAllDimensions(p3).find((d) => d.name === "height")!;
    expect(h.featureId).toBeNull();
    expect(h.preferred).toBe("scan");
    expect(h.preferredMm).toBeCloseTo(27.6, 6);
  });

  it("snapped preference when snapped set but no caliper (J-bend p2)", () => {
    const p2 = byId(trapParts, "p2");
    const od = resolveAllDimensions(p2).find((d) => d.name === "outer_diameter")!;
    expect(od.preferred).toBe("snapped");
    expect(od.preferredMm).toBeCloseTo(38.1, 6); // 2 * 19.05
  });

  it("d_shaft diameter resolves from caliper radius (knob p2)", () => {
    const p2 = byId(knobParts, "p2");
    const sd = resolveAllDimensions(p2).find((d) => d.name === "shaft_diameter")!;
    expect(sd.preferred).toBe("caliper");
    expect(sd.preferredMm).toBeCloseTo(6.0, 6); // 2 * 3.0
    const fd = resolveAllDimensions(p2).find((d) => d.name === "flat_depth")!;
    expect(fd.field).toBe("flat_depth");
    expect(fd.preferredMm).toBeCloseTo(1.5, 6);
  });

  it("returns null for a dimension whose feature is missing", () => {
    const p1 = byId(trapParts, "p1");
    const broken = { name: "x", derived_from: "nope_f9.radius", factor: 2 } as never;
    expect(resolveDimension(p1, broken)).toBeNull();
  });
});

describe("preferred source rule", () => {
  it("caliper beats snapped beats scan", () => {
    expect(
      preferredSource({ scan_mm: 1, snapped_mm: 2, caliper_mm: 3, preferred: "scan" }),
    ).toBe("caliper");
    expect(
      preferredSource({ scan_mm: 1, snapped_mm: 2, caliper_mm: null, preferred: "scan" }),
    ).toBe("snapped");
    expect(
      preferredSource({ scan_mm: 1, snapped_mm: null, caliper_mm: null, preferred: "snapped" }),
    ).toBe("scan");
  });

  it("preferredValueMm follows the preferred source", () => {
    expect(
      preferredValueMm({ scan_mm: 1, snapped_mm: 2, caliper_mm: 3, preferred: "scan" }),
    ).toBe(3);
  });
});

describe("editable feature measurements", () => {
  it("lists radius and length for a cylinder feature", () => {
    const p1 = byId(trapParts, "p1");
    const fields = editableMeasurements(p1).map((e) => `${e.featureId}.${e.field}`);
    expect(fields).toContain("p1_f1.radius");
    expect(fields).toContain("p1_f1.length");
  });

  it("includes flat_depth for a d_shaft feature", () => {
    const p2 = byId(knobParts, "p2");
    const fields = editableMeasurements(p2).map((e) => `${e.featureId}.${e.field}`);
    expect(fields).toContain("p2_f1.flat_depth");
    expect(fields).toContain("p2_f1.radius");
  });
});
