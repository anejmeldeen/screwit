"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useScan, useReconstruction, useParts, isStageDone, usePersistentNumber } from "@/lib/hooks";
import { scaleWarnings } from "@/lib/scaleWarnings";
import { ViewerPanel } from "@/components/viewer/ViewerPanel";
import { PartsPanel } from "@/components/PartsPanel";
import { ResizeHandle } from "@/components/ResizeHandle";
import { PartDownloads } from "@/components/PartDownloads";
import { ChatPanel } from "@/components/ChatPanel";
import { ScanProgress } from "@/components/StatusBar";
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

  const gridRef = useRef<HTMLDivElement>(null);
  const [gridWidth, setGridWidth] = useState(0);
  const [gridHeight, setGridHeight] = useState(0);
  const [partsWidth, setPartsWidth] = usePersistentNumber("ui.partsWidth", 310);
  const [chatWidth, setChatWidth] = usePersistentNumber("ui.chatWidth", 340);
  const [partsHeight, setPartsHeight] = usePersistentNumber("ui.partsHeight", 280);
  const [chatHeight, setChatHeight] = usePersistentNumber("ui.chatHeight", 480);
  // Measure the grid synchronously BEFORE the first paint (useLayoutEffect), so the responsive
  // layout (wide/stacked, panel clamps) is computed from the real width on frame one. Measuring in
  // a plain effect left gridWidth at 0 for the first paint -> the layout rendered stacked/narrow and
  // then visibly SNAPPED to the real layout once the ResizeObserver fired. Seed from the current
  // client rect, then keep it live with the observer.
  useLayoutEffect(() => {
    const grid = gridRef.current;
    if (!grid) return;
    const rect = grid.getBoundingClientRect();
    setGridWidth(rect.width);
    setGridHeight(rect.height);
    const observer = new ResizeObserver(([entry]) => {
      setGridWidth(entry.contentRect.width); setGridHeight(entry.contentRect.height);
    });
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);
  const wide = gridWidth >= 1280;
  const stacked = gridWidth < 760;
  // Keep only a small usable viewer area; panels can use the remaining space.
  const actualPartsWidth = Math.min(partsWidth, Math.max(240, gridWidth - 260 - 244));
  const maxChatWidth = Math.max(260, gridWidth - (wide ? actualPartsWidth + 244 : 232));
  const actualChatWidth = Math.min(chatWidth, maxChatWidth);
  const maxPartsWidth = Math.max(240, gridWidth - actualChatWidth - 244);
  const actualPartsHeight = stacked ? partsHeight : Math.min(partsHeight, Math.max(160, gridHeight - 232));

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

  useEffect(() => {
    if (parts && selectedId && !parts.some((p) => p.part_id === selectedId)) {
      setSelectedId(null); setFocusPartId(null); setHoveredId(null);
    }
  }, [parts, selectedId]);

  const partsReady = isStageDone(scan, "parts");

  // When a scan we watched processing turns ready, keep the progress bar up for a moment so it
  // visibly fills to 100% and says Ready, instead of vanishing mid-way. The bar stays mounted the
  // whole time (unmounting it reset its animation and made it jump). A scan that was already ready
  // when opened shows no bar at all.
  const status = scan?.status;
  const [watching, setWatching] = useState(false);
  useEffect(() => {
    if (!status) return;
    if (status !== "ready" && status !== "failed") {
      setWatching(true);
      return;
    }
    if (status === "ready") {
      const t = setTimeout(() => setWatching(false), 2200);
      return () => clearTimeout(t);
    }
    setWatching(false); // failed: the failure notice replaces the bar
  }, [status]);

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
      {!scanError && (watching || !scan || (scan.status !== "ready" && scan.status !== "failed")) ? <div className="workbench-notice" role="status"><ScanProgress scan={scan} /></div> : null}
      {scaleWarnings(recon?.scale ?? null).map((warning) => <div key={warning.id} className="workbench-notice" role="note">{warning.message}</div>)}
      <div ref={gridRef} className="workbench-grid" style={{
        "--parts-width": `${actualPartsWidth}px`, "--chat-width": `${actualChatWidth}px`,
        "--parts-height": `${actualPartsHeight}px`, "--chat-height": `${chatHeight}px`,
      } as CSSProperties}>
        <section className="workbench-viewer" aria-label="3D viewer">
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

        <ResizeHandle label="Resize parts panel" className="parts-resizer" orientation={wide ? "vertical" : "horizontal"}
          value={wide ? actualPartsWidth : actualPartsHeight} min={wide ? 240 : 160}
          max={wide ? maxPartsWidth : stacked ? Math.max(1600, gridHeight * 2) : Math.max(160, gridHeight - 232)} reverse={!stacked}
          onChange={wide ? setPartsWidth : setPartsHeight} />
        <section className="workbench-parts" aria-label="Object parts">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="parts-heading">Parts</h2>
            {hiddenPartIds.length > 0 ? <button className="text-xs text-accent" onClick={() => setHiddenPartIds([])}>Show all ({hiddenPartIds.length} hidden)</button> : null}
          </div>
          {partsReady && parts?.length ? <PartDownloads scanId={scanId} /> : null}
          {parts?.find(p => p.part_id === selectedId) ? <PartDownloads scanId={scanId} part={parts.find(p => p.part_id === selectedId)} /> : null}
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

        <ResizeHandle label="Resize assistant panel" className="chat-resizer" orientation={stacked ? "horizontal" : "vertical"}
          value={stacked ? chatHeight : actualChatWidth} min={stacked ? 280 : 260}
          max={stacked ? Math.max(1600, gridHeight * 2) : maxChatWidth} reverse={!stacked}
          onChange={stacked ? setChatHeight : setChatWidth} />
        <section className="workbench-chat" aria-label="Assistant">
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
        className="isolation-dialog">
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <h2 id="isolation-title">Isolated component · {isolatedParts[0]?.label}</h2>
            <button autoFocus onClick={() => setIsolatedId(null)} className="rounded border border-line px-3 py-1">Close</button>
          </div>
          {isolatedParts[0] ? <PartDownloads key={isolatedParts[0].part_id} scanId={scanId} part={isolatedParts[0]} /> : null}
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
