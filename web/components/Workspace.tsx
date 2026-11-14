"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useScan, useReconstruction, useParts, isStageDone } from "@/lib/hooks";
import { StatusBar } from "@/components/StatusBar";
import { ViewerPanel } from "@/components/viewer/ViewerPanel";
import { PartsPanel } from "@/components/PartsPanel";
import { ChatPanel } from "@/components/ChatPanel";
import { ErrorNote } from "@/components/ui";
import { markOnce, PERF_MARKS } from "@/lib/perf";
import type { CadResult } from "@/lib/contracts.gen";
import type { ViewerCadOverlay } from "@/components/viewer/Viewer";

export function Workspace({
  scanId,
  initialPartId = null,
}: {
  scanId: string;
  initialPartId?: string | null;
}) {
  const { scan, error: scanError } = useScan(scanId);
  const { recon } = useReconstruction(scanId, scan);
  const { parts, refresh: refreshParts } = useParts(scanId, scan);

  const [selectedId, setSelectedId] = useState<string | null>(initialPartId);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [focusPartId, setFocusPartId] = useState<string | null>(initialPartId);
  const [cadOverlay, setCadOverlay] = useState<ViewerCadOverlay | null>(null);

  const [hiddenPartIds, setHiddenPartIds] = useState<string[]>([]);
  const [isolatedId, setIsolatedId] = useState<string | null>(null);
  const isolationDialog = useRef<HTMLDialogElement>(null);
  const isolatedParts = useMemo(() => (parts ?? []).filter(p => p.part_id === isolatedId), [parts, isolatedId]);
  useEffect(() => {
    const dialog = isolationDialog.current;
    if (isolatedId && dialog && !dialog.open) dialog.showModal();
    if (!isolatedId && dialog?.open) dialog.close();
  }, [isolatedId]);

  const partsReady = isStageDone(scan, "parts");

  const headerSub = useMemo(() => {
    if (!scan) return "";
    if (scan.status === "failed") return "Processing failed";
    if (scan.status === "ready") return "Ready";
    return "Processing…";
  }, [scan]);

  // Mark the shell as painted once the panels, status bar and empty viewer
  // canvas are on screen (before any 3D asset loads). Two rAFs = after the
  // first committed paint.
  useEffect(() => {
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => markOnce(PERF_MARKS.shellReady));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, []);

  const onPartChip = useCallback((id: string) => {
    setSelectedId(id);
    setFocusPartId(id);
    // Re-trigger focus even if the same part is clicked again.
    setTimeout(() => setFocusPartId((cur) => (cur === id ? id : cur)), 0);
  }, []);

  const onHighlight = useCallback((ids: string[]) => {
    if (ids.length) {
      setSelectedId(ids[0]);
      setFocusPartId(ids[0]);
    }
  }, []);

  // A cad_result card payload -> viewer overlay. placement comes from the CAD
  // request; the card carries files + checks. We read placement from the card
  // if present, else default to the origin.
  const onShowCad = useCallback((payload: Record<string, unknown>) => {
    const cad = payload as unknown as CadResult;
    const placement =
      (payload.placement as { translate_mm: number[]; rotation_quat: number[] }) ??
      { translate_mm: [0, 0, 0], rotation_quat: [0, 0, 0, 1] };
    setCadOverlay({ cad, placement });
  }, []);

  return (
    <div className="flex h-[calc(100vh-57px)] flex-col">
      {/* Status bar — fixed height so it never reflows the grid below (CLS). */}
      <div className="min-h-[84px] shrink-0 overflow-hidden border-b border-slate-800 bg-panel px-4 py-2">
        <div className="mb-1 flex items-center justify-between">
          <h1 className="text-sm font-semibold">
            Scan {scanId}{" "}
            <span className="font-normal text-slate-400">· {headerSub}</span>
          </h1>
        </div>
        {scanError ? (
          <div className="space-y-1">
            <ErrorNote message={`Could not load scan: ${scanError}`} />
          </div>
        ) : (
          <StatusBar scan={scan} scale={recon?.scale ?? null} />
        )}
      </div>

      {/* Workspace. On mobile the columns stack; each section reserves a fixed
          height up front so later content never pushes earlier content (CLS).
          On lg+ it is a single full-height 3-column row. */}
      <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[60vh_40vh_60vh] lg:grid-rows-1 lg:grid-cols-[1fr_320px_360px]">
        <section className="relative min-h-0 border-b border-slate-800 lg:border-b-0 lg:border-r">
          <ViewerPanel
            scanId={scanId}
            recon={recon}
            parts={parts ?? []}
            hiddenPartIds={hiddenPartIds}
            selectedId={selectedId}
            hoveredId={hoveredId}
            focusPartId={focusPartId}
            onHover={setHoveredId}
            onSelect={(id) => {
              setSelectedId(id);
              setFocusPartId(id);
            }}
            cadOverlay={cadOverlay}
            onScaled={() => {
              refreshParts();
            }}
          />
        </section>

        <section className="min-h-0 overflow-y-auto border-b border-slate-800 p-3 lg:border-b-0 lg:border-r">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-300">Parts</h2>
            {hiddenPartIds.length > 0 ? <button className="text-xs text-accent" onClick={() => setHiddenPartIds([])}>Show all ({hiddenPartIds.length} hidden)</button> : null}
          </div>
          <PartsPanel
            hiddenPartIds={hiddenPartIds}
            onToggleHidden={id => setHiddenPartIds(ids => ids.includes(id) ? ids.filter(p => p !== id) : [...ids, id])}
            onIsolate={setIsolatedId}
            scanId={scanId}
            parts={parts}
            partsReady={partsReady}
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id);
              setFocusPartId(id);
            }}
            onChanged={refreshParts}
          />
        </section>

        <section className="min-h-0">
          <ChatPanel
            scanId={scanId}
            onHighlight={onHighlight}
            onPartChip={onPartChip}
            onShowCad={onShowCad}
          />
        </section>
      </div>
      <dialog ref={isolationDialog} aria-labelledby="isolation-title"
        onCancel={() => setIsolatedId(null)} onClose={() => setIsolatedId(null)}
        className="fixed inset-0 h-[85vh] w-[90vw] max-w-6xl rounded-xl border border-slate-600 bg-ink p-0 text-slate-100 backdrop:bg-black/70">
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
            <h2 id="isolation-title">Isolated component · {isolatedParts[0]?.label}</h2>
            <button autoFocus onClick={() => setIsolatedId(null)} className="rounded border border-slate-600 px-3 py-1">Close</button>
          </div>
          <div className="min-h-0 flex-1">
            {isolatedId && isolatedParts.length > 0 ? <ViewerPanel key={isolatedId}
              scanId={scanId} recon={recon} parts={isolatedParts}
              selectedId={isolatedId} hoveredId={null} focusPartId={null}
              onHover={() => {}} onSelect={() => {}} cadOverlay={null}
              onScaled={refreshParts} isolated /> : null}
          </div>
        </div>
      </dialog>
    </div>
  );
}
