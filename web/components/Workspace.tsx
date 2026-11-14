"use client";

import { useCallback, useEffect, useState } from "react";
import { useScan, useReconstruction, useParts, isStageDone } from "@/lib/hooks";
import { scaleWarnings } from "@/lib/scaleWarnings";
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

  useEffect(() => {
    if (parts && selectedId && !parts.some((p) => p.part_id === selectedId)) {
      setSelectedId(null); setFocusPartId(null); setHoveredId(null);
    }
  }, [parts, selectedId]);

  const partsReady = isStageDone(scan, "parts");

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
    <div className="workbench flex flex-col">
      <h1 className="sr-only">Workbench</h1>
      {scanError ? <div className="workbench-notice"><ErrorNote message={`Could not load this object: ${scanError}`} /></div> : null}
      {scan?.status === "failed" ? <div className="workbench-notice" role="alert">This scan could not finish. Try another recording.</div> : null}
      {!scanError && (!scan || (scan.status !== "ready" && scan.status !== "failed")) ? <div className="workbench-notice" role="status">Preparing your object…</div> : null}
      {scaleWarnings(recon?.scale ?? null).map((warning) => <div key={warning.id} className="workbench-notice" role="note">{warning.message}</div>)}
      <div className="workbench-grid">
        <section className="workbench-viewer" aria-label="3D viewer">
          <ViewerPanel
            scanId={scanId}
            recon={recon}
            parts={parts ?? []}
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

        <section className="workbench-parts" aria-label="Object parts">
          <h2 className="parts-heading">Parts</h2>
          <PartsPanel
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

        <section className="workbench-chat" aria-label="Assistant">
          <ChatPanel
            scanId={scanId}
            onHighlight={onHighlight}
            onPartChip={onPartChip}
            onShowCad={onShowCad}
          />
        </section>
      </div>
    </div>
  );
}
