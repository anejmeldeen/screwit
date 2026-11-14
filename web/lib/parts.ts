// Part / Measurement helpers (SPEC 4.3).
//
// A dimension is either an OwnDimension (holds its own Measurement) or a
// DerivedDimension (a read-only VIEW of a feature Measurement, named by
// `derived_from` = "<feature_id>.<field>" with an optional `factor`, e.g. 2.0
// for a diameter from a radius). PATCH always targets the underlying feature.
// `preferred` on a Measurement is caliper if caliper_mm is set, else snapped if
// snapped_mm is set, else scan; downstream reads the preferred value and
// records which it was.

import type { Part, PartNS } from "./contracts.gen";

type Measurement = PartNS.Measurement;

export type MeasurementField =
  | "radius"
  | "length"
  | "flat_depth"
  | "extent_u"
  | "extent_v"
  | "depth";

export type AnyFeature = Part["features"][number];

const MEASUREMENT_FIELDS: MeasurementField[] = [
  "radius",
  "length",
  "flat_depth",
  "extent_u",
  "extent_v",
  "depth",
];

// Is this feature a measurable (non-thread_suspected) feature?
function isMeasurableFeature(
  f: AnyFeature,
): f is Extract<AnyFeature, { feature_id: string }> {
  return f.type !== "thread_suspected";
}

export function findFeature(part: Part, featureId: string): AnyFeature | undefined {
  return part.features.find((f) => f.feature_id === featureId);
}

// Read a Measurement off a feature by field name, if present.
export function featureMeasurement(
  feature: AnyFeature,
  field: string,
): Measurement | undefined {
  if (!isMeasurableFeature(feature)) return undefined;
  const rec = feature as unknown as Record<string, unknown>;
  const m = rec[field];
  if (m && typeof m === "object" && "preferred" in (m as object)) {
    return m as Measurement;
  }
  return undefined;
}

// The source currently preferred for a Measurement. Recompute rather than trust
// a stale `preferred` so the UI and downstream agree with the SPEC rule.
export function preferredSource(m: Measurement): "scan" | "snapped" | "caliper" {
  if (m.caliper_mm != null) return "caliper";
  if (m.snapped_mm != null) return "snapped";
  return "scan";
}

// The preferred numeric value of a Measurement in millimetres (or null if the
// scan value itself is null).
export function preferredValueMm(m: Measurement): number | null {
  switch (preferredSource(m)) {
    case "caliper":
      return m.caliper_mm ?? null;
    case "snapped":
      return m.snapped_mm ?? null;
    default:
      return m.scan_mm ?? null;
  }
}

export interface ResolvedDimension {
  name: string;
  // The underlying Measurement (own or the referenced feature's).
  measurement: Measurement;
  // feature_id + field when derived; null for an own dimension.
  featureId: string | null;
  field: MeasurementField | null;
  factor: number;
  // Convenience: preferred source + the three candidate values scaled by factor.
  preferred: "scan" | "snapped" | "caliper";
  scanMm: number | null;
  snappedMm: number | null;
  caliperMm: number | null;
  preferredMm: number | null;
}

function scaleOrNull(v: number | null | undefined, factor: number): number | null {
  return v == null ? null : v * factor;
}

// Resolve a Part dimension into its concrete Measurement and scaled values.
// For a DerivedDimension, follow `derived_from` to the feature Measurement and
// multiply every candidate value (and the preferred value) by `factor`.
export function resolveDimension(
  part: Part,
  dim: Part["dimensions"][number],
): ResolvedDimension | null {
  // OwnDimension: has its own `value` Measurement.
  if ("value" in dim && dim.value) {
    const m = dim.value;
    return {
      name: dim.name,
      measurement: m,
      featureId: null,
      field: null,
      factor: 1,
      preferred: preferredSource(m),
      scanMm: m.scan_mm ?? null,
      snappedMm: m.snapped_mm ?? null,
      caliperMm: m.caliper_mm ?? null,
      preferredMm: preferredValueMm(m),
    };
  }
  // DerivedDimension: "<feature_id>.<field>".
  if ("derived_from" in dim && dim.derived_from) {
    const [featureId, field] = dim.derived_from.split(".");
    const factor = (dim.factor as number | undefined) ?? 1;
    const feature = findFeature(part, featureId);
    if (!feature) return null;
    const m = featureMeasurement(feature, field);
    if (!m) return null;
    const fld = MEASUREMENT_FIELDS.includes(field as MeasurementField)
      ? (field as MeasurementField)
      : null;
    return {
      name: dim.name,
      measurement: m,
      featureId,
      field: fld,
      factor,
      preferred: preferredSource(m),
      scanMm: scaleOrNull(m.scan_mm, factor),
      snappedMm: scaleOrNull(m.snapped_mm, factor),
      caliperMm: scaleOrNull(m.caliper_mm, factor),
      preferredMm: scaleOrNull(preferredValueMm(m), factor),
    };
  }
  return null;
}

export function resolveAllDimensions(part: Part): ResolvedDimension[] {
  return part.dimensions
    .map((d) => resolveDimension(part, d))
    .filter((d): d is ResolvedDimension => d !== null);
}

// Enumerate every editable feature Measurement on a part (for caliper entry),
// as (feature_id, field, measurement) tuples.
export function editableMeasurements(
  part: Part,
): { featureId: string; field: MeasurementField; measurement: Measurement; featureType: string }[] {
  const out: {
    featureId: string;
    field: MeasurementField;
    measurement: Measurement;
    featureType: string;
  }[] = [];
  for (const f of part.features) {
    if (!isMeasurableFeature(f)) continue;
    for (const field of MEASUREMENT_FIELDS) {
      const m = featureMeasurement(f, field);
      if (m) {
        out.push({
          featureId: f.feature_id,
          field,
          measurement: m,
          featureType: (f.type as string) ?? "unknown",
        });
      }
    }
  }
  return out;
}
