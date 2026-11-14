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
      return "bg-slate-600";
    default:
      return "bg-slate-700";
  }
}

// Weight each stage by its share of wall time (geometry dominates; see cpu-verify timings) so a
// single bar moves monotonically 0->100 across the whole scan. A running stage contributes its
// own progress fraction; skipped stages are dropped from the denominator so the bar still reaches
// 100 for mesh input.
const STAGE_WEIGHT: Record<string, number> = {
  frames: 0.15,
  geometry: 0.45,
  scale: 0.03,
  object_mask: 0.07,
  splat: 0.1,
  mesh: 0.15,
  parts: 0.05,
};

function overallPercent(scan: Scan): number {
  let total = 0;
  let acc = 0;
  for (const s of scan.stages) {
    if (s.state === "skipped") continue;
    const w = STAGE_WEIGHT[s.name] ?? 0;
    total += w;
    if (s.state === "done" || s.state === "failed") {
      acc += w;
    } else if (s.state === "running") {
      acc += w * (s.progress ?? 0);
    }
  }
  if (total <= 0) return 0;
  return Math.round((acc / total) * 100);
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
      <div data-testid="status-bar">
        <div className="mb-2" data-testid="overall-progress" data-percent={0}>
          <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
            <span>Loading…</span>
            <span className="tabular-nums">0%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
              style={{ width: "0%" }}
            />
          </div>
        </div>
        <div className="flex gap-3 overflow-x-auto py-2">
          {STAGE_ORDER.map((s) => (
            <div
              key={s.name}
              data-testid={`stage-${s.name}`}
              data-state="pending"
              className="skeleton h-6 w-24 shrink-0"
            />
          ))}
        </div>
      </div>
    );
  }

  const byName = new Map<string, (typeof scan.stages)[number]>(
    scan.stages.map((s) => [s.name as string, s]),
  );
  const failed = scan.stages.find((s) => s.state === "failed");
  const running = scan.stages.find((s) => s.state === "running");
  const pct = overallPercent(scan);
  const showBar = scan.status !== "ready" && !failed;

  return (
    <div data-testid="status-bar">
      {showBar ? (
        <div className="mb-2" data-testid="overall-progress" data-percent={pct}>
          <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
            <span>
              {running ? (byName.get(running.name)?.message ?? `${running.name}…`) : "Starting…"}
            </span>
            <span className="tabular-nums">{pct}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      ) : null}
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
                    ? "bg-slate-800 text-slate-500"
                    : "bg-slate-800 text-slate-200"
                }`}
                title={stage?.message ?? s.label}
              >
                <span className={`h-2 w-2 rounded-full ${dotClass(state)}`} />
                {s.label}
                {state === "running" && stage?.progress != null ? (
                  <span className="text-slate-400">
                    {Math.round(stage.progress * 100)}%
                  </span>
                ) : null}
              </span>
              {i < STAGE_ORDER.length - 1 ? (
                <span className="text-slate-600">›</span>
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
                  ? "bg-slate-700 text-slate-200"
                  : "bg-amber-800/60 text-amber-200"
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
        <p className="mt-2 text-sm text-rose-300">
          {FRIENDLY[failed.name] ?? "A stage failed."}{" "}
          {failed.message ? (
            <span className="text-rose-400/80">({failed.message})</span>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}
