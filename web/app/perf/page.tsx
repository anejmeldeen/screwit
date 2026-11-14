"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { splatUriFor, detectQuality, AssetQuality } from "@/lib/device";
import { PERF_MARKS } from "@/lib/perf";
import { ViewerPanel } from "@/components/viewer/ViewerPanel";
import type { Part, Reconstruction } from "@/lib/contracts.gen";

// On-device performance readout. Loads a hero scan (?scan=scn_trap) and shows
// live fps (p50/p95 over the last 5 s) plus the viewer:first-mesh,
// viewer:first-splat and chat:first-token marks, so a person can read numbers
// off a real phone and the X6 Playwright/Lighthouse tests can assert the marks.
function PerfInner() {
  const sp = useSearchParams();
  const scanId = sp.get("scan") || "scn_trap";

  const [recon, setRecon] = useState<Reconstruction | null>(null);
  const [parts, setParts] = useState<Part[]>([]);
  const [quality, setQuality] = useState<AssetQuality>("desktop");
  const [err, setErr] = useState<string | null>(null);

  const samples = useRef<{ t: number; fps: number }[]>([]);
  const [stats, setStats] = useState<{ p50: number; p95: number; last: number }>({
    p50: 0,
    p95: 0,
    last: 0,
  });
  const [marks, setMarks] = useState<Record<string, number | null>>({
    [PERF_MARKS.firstMesh]: null,
    [PERF_MARKS.firstSplat]: null,
    [PERF_MARKS.chatFirstToken]: null,
  });

  useEffect(() => {
    setQuality(detectQuality());
    Promise.all([api.getReconstruction(scanId), api.getParts(scanId)])
      .then(([r, p]) => {
        setRecon(r);
        setParts(p);
      })
      .catch((e) => setErr(String(e?.message ?? e)));
  }, [scanId]);

  // Poll performance marks so the on-screen readout updates when each first draws.
  useEffect(() => {
    const id = setInterval(() => {
      if (typeof performance === "undefined" || !performance.getEntriesByName)
        return;
      setMarks((cur) => {
        const next = { ...cur };
        for (const name of Object.keys(cur)) {
          if (next[name] == null) {
            const e = performance.getEntriesByName(name, "mark")[0];
            if (e) next[name] = Math.round(e.startTime);
          }
        }
        return next;
      });
    }, 200);
    return () => clearInterval(id);
  }, []);

  const onFps = useCallback((fps: number) => {
    const now = performance.now();
    const arr = samples.current;
    arr.push({ t: now, fps });
    // Keep a 5 s window.
    const cutoff = now - 5000;
    while (arr.length && arr[0].t < cutoff) arr.shift();
    const vals = arr.map((s) => s.fps).sort((a, b) => a - b);
    if (vals.length) {
      const p = (q: number) => vals[Math.min(vals.length - 1, Math.floor(q * vals.length))];
      setStats({ p50: p(0.5), p95: p(0.95), last: fps });
    }
  }, []);

  const splatUri = recon ? splatUriFor(recon, quality) : null;

  return (
    <div className="flex h-[calc(100vh-57px)] flex-col">
      <div className="grid grid-cols-2 gap-2 border-b border-slate-800 bg-panel px-4 py-2 text-xs sm:grid-cols-4">
        <Stat label="fps (now)" value={stats.last.toFixed(0)} />
        <Stat label="fps p50 / 5 s" value={stats.p50.toFixed(0)} />
        <Stat label="fps p95 / 5 s" value={stats.p95.toFixed(0)} />
        <Stat label="quality" value={quality} />
        <Stat
          label="first-mesh"
          value={fmtMark(marks[PERF_MARKS.firstMesh])}
          id="mark-first-mesh"
        />
        <Stat
          label="first-splat"
          value={fmtMark(marks[PERF_MARKS.firstSplat])}
          id="mark-first-splat"
        />
        <Stat
          label="chat:first-token"
          value={fmtMark(marks[PERF_MARKS.chatFirstToken])}
          id="mark-chat-first-token"
        />
        <Stat label="scan" value={scanId} />
      </div>

      <div className="min-h-0 flex-1">
        {err ? (
          <p className="p-4 text-rose-300">Could not load {scanId}: {err}</p>
        ) : recon ? (
          <ViewerPanel
            scanId={scanId}
            recon={recon}
            parts={parts}
            selectedId={null}
            hoveredId={null}
            focusPartId={null}
            onHover={() => {}}
            onSelect={() => {}}
            cadOverlay={null}
            onScaled={() => {}}
            onFpsSample={onFps}
          />
        ) : (
          <p className="p-4 text-slate-400">Loading {scanId}…</p>
        )}
        {/* Expose the resolved mobile/full choice for the harness. */}
        <span className="sr-only" data-splat-uri={splatUri ?? ""} />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  id,
}: {
  label: string;
  value: string;
  id?: string;
}) {
  return (
    <div id={id} className="rounded bg-slate-800 px-2 py-1">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className="font-mono text-sm text-slate-100" data-perf-value={value}>
        {value}
      </div>
    </div>
  );
}

function fmtMark(v: number | null): string {
  return v == null ? "—" : `${v} ms`;
}

export default function PerfPage() {
  return (
    <Suspense fallback={<p className="p-4 text-slate-400">Loading…</p>}>
      <PerfInner />
    </Suspense>
  );
}
