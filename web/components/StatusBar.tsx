"use client";

import { useEffect, useRef, useState } from "react";
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

// Weight each stage by its share of wall time so a single bar moves 0->100 across the whole scan.
// Measured on an M3 Pro (headphones clip, 8 views): frames ~13 s, geometry ~30 s on the Apple GPU
// (about 4 min on the CPU, under 1 s from the video cache), scale + depth cleanup ~4 s,
// mask/splat/mesh ~2 s together, parts ~38 s (one Claude call to group the parts, one for the
// scene graph). The client-side creep below covers the variation between runs.
// Skipped stages are dropped from the denominator so the bar still reaches 100 for mesh input.
const STAGE_WEIGHT: Record<string, number> = {
  frames: 0.15,
  geometry: 0.3,
  scale: 0.05,
  object_mask: 0.03,
  splat: 0.02,
  mesh: 0.05,
  parts: 0.4,
};

// (percent now, percent at the end of the running stage). A running stage contributes its own
// progress fraction. Before the first stage starts (upload check) and between stages, the ceiling
// is the end of the next pending stage, so the bar can creep instead of sitting still.
function progressSpan(scan: Scan): { pct: number; ceiling: number } {
  let total = 0;
  let acc = 0;
  let ceiling = 0;
  let running = false;
  let nextPending = 0;
  for (const s of scan.stages) {
    if (s.state === "skipped") continue;
    const w = STAGE_WEIGHT[s.name] ?? 0;
    total += w;
    if (s.state === "done" || s.state === "failed") {
      acc += w;
      ceiling += w;
    } else if (s.state === "running") {
      acc += w * (s.progress ?? 0);
      ceiling += w;
      running = true;
    } else if (!nextPending) {
      nextPending = w;
    }
  }
  if (!running) ceiling += nextPending;
  if (total <= 0) return { pct: 0, ceiling: 0 };
  return { pct: (acc / total) * 100, ceiling: (ceiling / total) * 100 };
}

function isActive(scan: Scan | null): boolean {
  return !!scan && ["uploaded", "validating", "processing"].includes(scan.status);
}

// The value the bar shows: eases toward the server's percent instead of jumping (a cached stage
// can finish in a fraction of a second), and while a stage runs between server updates it creeps
// slowly ahead, never more than 4 points past the server and never past the end of that stage, so
// the bar neither sits frozen nor has to come back. Never moves backwards unless the server's own
// percent drops (a restarted scan).
function useGradualPercent(target: number, ceiling: number, active: boolean): number {
  const [shown, setShown] = useState(target);
  const live = useRef({ shown: target, target, ceiling, active, lastTarget: target });
  const cur = live.current;
  if (target < cur.lastTarget - 5) cur.shown = target; // the server went backwards: a restart
  cur.lastTarget = target;
  cur.target = target;
  cur.ceiling = ceiling;
  cur.active = active;
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.25, (now - last) / 1000);
      last = now;
      const r = live.current;
      let s = r.shown;
      if (s < r.target) {
        // Close about half the gap per second (finish faster once everything is done), never
        // slower than 3 points per second.
        const rate = r.target >= 99.5 ? 2.0 : 1.2;
        s = Math.min(r.target, s + Math.max(3 * dt, (r.target - s) * (1 - Math.exp(-rate * dt))));
      } else if (r.active) {
        const cap = Math.min(r.ceiling - 0.5, r.target + 4);
        if (s < cap) s += (cap - s) * (1 - Math.exp(-dt / 20)); // asymptotic, time constant 20 s
      }
      if (Math.abs(s - r.shown) > 1e-3) {
        r.shown = s;
        setShown(s);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return Math.max(0, Math.min(100, shown));
}

const STAGE_LABEL: Record<string, string> = Object.fromEntries(
  STAGE_ORDER.map((s) => [s.name, s.label]),
);

// Live "Preparing your object" progress: the weighted overall bar plus the current stage's label,
// message, and percent. Driven by the SSE-updated Scan (its per-stage progress ticks). Shown while
// a scan is processing; renders a starting state before the first stage reports.
export function ScanProgress({ scan }: { scan: Scan | null }) {
  const span = scan ? progressSpan(scan) : { pct: 0, ceiling: 0 };
  const pct = Math.round(span.pct);
  const running = scan?.stages.find((s) => s.state === "running");
  const shown = useGradualPercent(span.pct, span.ceiling, isActive(scan));
  const headline = !scan
    ? "Preparing your object…"
    : running
      ? `Preparing: ${STAGE_LABEL[running.name] ?? running.name}`
      : scan.status === "ready"
        ? "Ready"
        : "Preparing your object…";
  const detail = running?.message ?? null;
  return (
    <div data-testid="scan-progress" data-percent={pct}>
      <div className="mb-1 flex items-center justify-between text-xs text-muted">
        <span>
          {headline}
          {detail ? <span className="text-slate-400"> · {detail}</span> : null}
        </span>
        <span className="tabular-nums">{Math.floor(shown)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
        <div className="h-full rounded-full bg-accent" style={{ width: `${shown}%` }} />
      </div>
    </div>
  );
}

export function StatusBar({
  scan,
  scale = null,
}: {
  scan: Scan | null;
  scale?: Reconstruction["scale"] | null;
}) {
  const warnings = scaleWarnings(scale);
  const span = scan ? progressSpan(scan) : { pct: 0, ceiling: 0 };
  const shown = useGradualPercent(span.pct, span.ceiling, isActive(scan));

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
  const pct = Math.round(span.pct);
  const showBar = scan.status !== "ready" && !failed;

  return (
    <div data-testid="status-bar">
      {showBar ? (
        <div className="mb-2" data-testid="overall-progress" data-percent={pct}>
          <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
            <span>
              {running ? (byName.get(running.name)?.message ?? `${running.name}…`) : "Starting…"}
            </span>
            <span className="tabular-nums">{Math.floor(shown)}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <div className="h-full rounded-full bg-accent" style={{ width: `${shown}%` }} />
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
