"use client";

import type { Reconstruction, Scan } from "@/lib/contracts.gen";
import { scaleWarnings } from "@/lib/scaleWarnings";

// Status bar driven ONLY by Scan.stages (not the top-level status), SPEC 2.3 /
// 4.1. Stage order is fixed. `skipped` stages are greyed; `failed` gets a
// friendly message.
const STAGE_ORDER: { name: string; label: string }[] = [
  { name: "frames", label: "Frames" },
  { name: "geometry", label: "Geometry" },
  { name: "scale", label: "Scale" },
  { name: "object_mask", label: "Object mask" },
  { name: "splat", label: "Splat" },
  { name: "mesh", label: "Mesh" },
  { name: "parts", label: "Parts" },
];

const FRIENDLY: Record<string, string> = {
  frames: "Not enough sharp frames; try a slower orbit.",
  geometry: "Could not solve camera poses; re-record with more overlap.",
  scale: "Could not read the scale board; set the scale manually in the viewer.",
  object_mask: "Could not isolate the object from the background.",
  splat: "Splat export failed; the mesh view still works.",
  mesh: "Mesh reconstruction failed.",
  parts: "Part detection failed.",
};

function dotClass(state?: string) {
  switch (state) {
    case "done":
      return "bg-emerald-400";
    case "running":
      return "bg-accent animate-pulse";
    case "failed":
      return "bg-rose-500";
    case "skipped":
      return "bg-wash";
    default:
      return "bg-wash";
  }
}

export function StatusBar({
  scan,
  scale = null,
}: {
  scan: Scan | null;
  scale?: Reconstruction["scale"] | null;
}) {
  const warnings = scaleWarnings(scale);

  if (!scan) {
    return (
      <div
        data-testid="status-bar"
        className="flex gap-3 overflow-x-auto py-2"
      >
        {STAGE_ORDER.map((s) => (
          <div
            key={s.name}
            data-testid={`stage-${s.name}`}
            data-state="pending"
            className="skeleton h-6 w-24 shrink-0"
          />
        ))}
      </div>
    );
  }

  const byName = new Map<string, (typeof scan.stages)[number]>(
    scan.stages.map((s) => [s.name as string, s]),
  );
  const failed = scan.stages.find((s) => s.state === "failed");

  return (
    <div data-testid="status-bar">
      <ol className="flex items-center gap-x-2 overflow-x-auto whitespace-nowrap pb-1">
        {STAGE_ORDER.map((s, i) => {
          const stage = byName.get(s.name);
          const state = stage?.state;
          const skipped = state === "skipped";
          return (
            <li key={s.name} className="flex items-center gap-2">
              <span
                data-testid={`stage-${s.name}`}
                data-state={state ?? "pending"}
                className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                  skipped
                    ? "bg-wash text-muted"
                    : "bg-wash text-ink"
                }`}
                title={stage?.message ?? s.label}
              >
                <span className={`h-2 w-2 rounded-full ${dotClass(state)}`} />
                {s.label}
                {state === "running" && stage?.progress != null ? (
                  <span className="text-muted">
                    {Math.round(stage.progress * 100)}%
                  </span>
                ) : null}
              </span>
              {i < STAGE_ORDER.length - 1 ? (
                <span className="text-muted">›</span>
              ) : null}
            </li>
          );
        })}
      </ol>
      {warnings.length ? (
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          {warnings.map((w) => (
            <span
              key={w.id}
              data-testid="scale-warning"
              data-scale-warning={w.id}
              className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ${
                w.level === "info"
                  ? "bg-wash text-ink"
                  : "bg-amber-100 text-amber-800"
              }`}
              title={w.message}
            >
              <span aria-hidden>{w.level === "info" ? "ⓘ" : "⚠"}</span>
              {w.message}
            </span>
          ))}
        </div>
      ) : null}
      {failed ? (
        <p className="mt-2 text-sm text-rose-800">
          {FRIENDLY[failed.name] ?? "A stage failed."}{" "}
          {failed.message ? (
            <span className="text-rose-800/80">({failed.message})</span>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}
