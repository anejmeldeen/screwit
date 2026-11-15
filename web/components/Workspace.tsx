"use client";

import Link from "next/link";
import { partColor } from "@/lib/palette";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useScan, useReconstruction, useParts, isStageDone, usePersistentNumber } from "@/lib/hooks";
import { scaleWarnings } from "@/lib/scaleWarnings";
import { ViewerPanel } from "@/components/viewer/ViewerPanel";
import { PartsPanel } from "@/components/PartsPanel";
import { ResizeHandle } from "@/components/ResizeHandle";
import { PartDownloads } from "@/components/PartDownloads";
import { ManualContext } from "@/components/cards/ManualContext";
import { ChatPanel } from "@/components/ChatPanel";
import { ScanProgress } from "@/components/StatusBar";
import { api } from "@/lib/api";
import { markBuildOpened, readCachedBuilds, writeCachedBuilds } from "@/lib/buildsCache";
import { useAccount } from "@/components/AccountProvider";
import { ScrewyIcon } from "@/components/ScrewyIcon";
import { ErrorNote } from "@/components/ui";
import { markOnce, PERF_MARKS } from "@/lib/perf";
import type { CadResult } from "@/lib/contracts.gen";
import type { ViewerCadOverlay } from "@/components/viewer/Viewer";

export function Workspace({
  scanId,
  initialPartId = null,
  creationForm, title,
}: {
  creationForm?: ReactNode;
  title?: string;
  scanId: string;
  initialPartId?: string | null;
}) {
  const { scan, error: scanError } = useScan(scanId);
  const { recon } = useReconstruction(scanId, scan);
  const { parts, setParts, refresh: refreshParts } = useParts(scanId, scan);

  const [savedTitle, setSavedTitle] = useState("");
  const username = useAccount()?.username ?? "";
  useEffect(() => {
    if (!scanId) return;
    let alive = true;
    // The dashboard's cached list gives the title at once and learns that this workspace was just
    // opened; the fresh list then replaces both.
    setSavedTitle(readCachedBuilds(username)?.find(build => build.scan_id === scanId)?.title ?? "");
    markBuildOpened(username, scanId);
    api.openBuild(scanId).catch(() => {});
    api.builds().then(builds => {
      writeCachedBuilds(username, builds);
      markBuildOpened(username, scanId);
      if (alive) setSavedTitle(builds.find(build => build.scan_id === scanId)?.title ?? "");
    }).catch(() => {});
    return () => { alive = false; };
  }, [scanId, username]);

  const gridRef = useRef<HTMLDivElement>(null);
  const dockRef = useRef<HTMLElement>(null);
  const [dockHeight, setDockHeight] = useState(0);
  const contentsRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState(0);
  const [heightAdjustment, setHeightAdjustment] = usePersistentNumber("ui.hierarchyHeightAdjustment", 0);
  const availableHeight = Math.max(1, dockHeight - 12);
  const minimumHeight = Math.min(42, availableHeight);
  const maximumHeight = Math.max(minimumHeight, availableHeight - 100);
  const naturalHeight = Math.min(contentHeight, maximumHeight);
  const hierarchyHeight = Math.max(minimumHeight, Math.min(maximumHeight, naturalHeight + heightAdjustment));
  const [gridWidth, setGridWidth] = useState(0);
  const [leftWidth, setLeftWidth] = usePersistentNumber("ui.objectDockWidth", 0);
  const [rightWidth, setRightWidth] = usePersistentNumber("ui.assistantWidth", 0);
  const [expanded, setExpanded] = useState(true);
  useLayoutEffect(() => {
    const grid = gridRef.current;
    const dock = dockRef.current;
    const contents = contentsRef.current;
    if (!grid || !dock || !contents) return;
    const measure = () => { setGridWidth(grid.clientWidth); setDockHeight(dock.clientHeight); setContentHeight(contents.offsetHeight); };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(contents);
    observer.observe(dock);
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);
  const maxLeft = Math.max(150, Math.min(420, gridWidth * .27));
  const maxRight = Math.max(220, Math.min(560, gridWidth * .35));
  const left = Math.min(leftWidth || Math.max(150, gridWidth * .165), maxLeft);
  const right = Math.min(rightWidth || Math.max(220, gridWidth * .215), maxRight);
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

  const name = title || savedTitle || "Workspace";
  const selected = parts?.find(part => part.part_id === selectedId);
  const [combineFrom, setCombineFrom] = useState<string | null>(null);
  const [combineError, setCombineError] = useState("");
  const combining = useRef(false);
  async function select(id: string | null) {
    if (combining.current) return;
    if (combineFrom && id) {
      if (id === combineFrom) return;
      combining.current = true;
      setCombineError("");
      try {
        const merged = await api.mergeParts(scanId, {part_ids: [combineFrom, id]});
        setParts(current => [...(current ?? []).filter(p => p.part_id !== combineFrom && p.part_id !== id), merged]);
        setHiddenPartIds(current => current.filter(p => p !== combineFrom && p !== id));
        setSelectedId(merged.part_id);
        setCombineFrom(null);
        refreshParts();
      } catch (err) {
        setCombineError(err instanceof Error ? err.message : "Could not combine these parts.");
      } finally { combining.current = false; }
      return;
    }
    setCombineFrom(null);
    setCombineError("");
    setSelectedId(id);
    setFocusPartId(null);
  }

  return (
    <div className="workbench flex flex-col">
      <header className="scene-title"><Link href="/">Workspaces</Link><span aria-hidden="true">/</span><h1>{name}</h1></header>
      {scanError ? <div className="workbench-notice"><ErrorNote message={`Could not load this object: ${scanError}`} /></div> : null}
      {scan?.status === "failed" ? <div className="workbench-notice" role="alert">This scan could not finish. Try another recording.</div> : null}
      {!creationForm && !scanError && (watching || !scan || (scan.status !== "ready" && scan.status !== "failed")) ? <div className="workbench-notice" role="status"><ScanProgress scan={scan} /></div> : null}
      {combineFrom && <div className="workbench-notice" role="status">Choose another part in the scene or hierarchy.{" "}
        <button onClick={() => { if (!combining.current) { setCombineFrom(null); setCombineError(""); } }}>Cancel</button>
        {combineError && <span role="alert"> {combineError}</span>}
      </div>}
      <div ref={gridRef} className="scene-layout" style={{ "--left": `${left}px`, "--right": `${right}px` } as CSSProperties}>
        <aside ref={dockRef} className="scene-object-dock" style={{ gridTemplateRows: `${hierarchyHeight}px 12px minmax(0,1fr)` }}>
          <section className="scene-hierarchy" aria-label="Hierarchy">
            <div ref={contentsRef} className="hierarchy-contents">
            <h2 className="dock-heading">Hierarchy</h2>
            <div className="scene-root">
              <button onClick={() => setExpanded(!expanded)} aria-label={expanded ? "Collapse assembly" : "Expand assembly"} aria-expanded={expanded}>{expanded ? "▾" : "▸"}</button>
              <button onClick={() => select(null)} aria-pressed={!selected}>{name}</button>
            </div>
            {expanded && <ul className="scene-tree">{(parts || []).map((part, index) =>
              <li key={part.part_id}><button aria-pressed={selectedId === part.part_id}
                onClick={() => select(part.part_id)} onMouseEnter={() => setHoveredId(part.part_id)} onMouseLeave={() => setHoveredId(null)}>
                <i style={{backgroundColor:partColor(index, part.color_source_id ?? part.part_id)}} />{part.label}{hiddenPartIds.includes(part.part_id) && <small>Hidden</small>}
              </button></li>)}</ul>}
            {!parts?.length && <p className="dock-note">{creationForm ? "No parts yet" : partsReady ? "No parts available" : "Waiting for parts…"}</p>}
            </div>
          </section>
          <ResizeHandle label="Resize hierarchy height" className="object-split" orientation="horizontal"
            min={minimumHeight} max={maximumHeight} value={hierarchyHeight}
            onChange={height => setHeightAdjustment(height - naturalHeight)} />
          <section key={selected?.part_id ?? "assembly"} className="scene-inspector" aria-label="Inspector">
            <h2 className="dock-heading">Inspector</h2>
            {!selected && <><h2>{name}</h2><p className="dock-note">{creationForm ? "Add photos or a video to build your assembly." : `${parts?.length ?? 0} components`}</p>
              {scaleWarnings(recon?.scale ?? null).map(w => <p className="dock-note" key={w.id}>{w.message}</p>)}</>}
            {!selected && hiddenPartIds.length > 0 && <button onClick={() => setHiddenPartIds([])}>Show all parts</button>}
            {!creationForm && <PartsPanel hiddenPartIds={hiddenPartIds}
              onToggleHidden={id => setHiddenPartIds(ids => ids.includes(id) ? ids.filter(p => p !== id) : [...ids, id])}
              onIsolate={setIsolatedId} scanId={scanId} parts={parts} partsReady={partsReady}
              selectedId={selectedId} onCombine={() => { setCombineFrom(selectedId); setCombineError(""); }} onChanged={refreshParts} />}
            {partsReady && !!parts?.length && <details name="inspector-section" className="scene-export"><summary>Export {selected ? "component" : "assembly"}</summary>
              <PartDownloads scanId={scanId} part={selected} /></details>}
          </section>
        </aside>
        <ResizeHandle label="Resize hierarchy and inspector" className="scene-split" orientation="vertical" value={left} min={150} max={maxLeft} onChange={setLeftWidth} />
        <section className="workbench-viewer" aria-label={creationForm ? "Workspace setup" : "3D viewer"}>
          {creationForm ? <div className="workspace-setup">{creationForm}</div> : <ViewerPanel combining={!!combineFrom}
            scanId={scanId}
            recon={recon}
            parts={parts ?? []}
            hiddenPartIds={hiddenPartIds}
            selectedId={selectedId}
            hoveredId={hoveredId}
            focusPartId={focusPartId}
            onHover={setHoveredId}
            onSelect={select}
            cadOverlay={cadOverlay}
            onScaled={() => {
              refreshParts();
            }}
          />}
        </section>

        <ResizeHandle label="Resize assistant" className="scene-split" orientation="vertical" value={right} min={220} max={maxRight} reverse onChange={setRightWidth} />
        <aside className="scene-dock">
          <section className="scene-assistant" aria-label="Assistant">
          {creationForm ? <div className="chat-panel">
            <header className="chat-heading"><span className="chat-mark screwy-avatar"><ScrewyIcon /></span><div><h2>Screwy</h2><p>Let’s build.</p></div></header>
            <div className="chat-messages"><div className="chat-empty"><h3>Start with your object.</h3><p>Name your workspace and add a video or photos. Once your scan starts, we can explore it together.</p></div></div>
          </div> : <ManualContext.Provider value={{ name, parts: parts ?? [] }}><ChatPanel
            scanId={scanId}
            onHighlight={onHighlight}
            onPartChip={onPartChip}
            onShowCad={onShowCad}
            onPartsChanged={refreshParts}
          /></ManualContext.Provider>}
          </section>
        </aside>
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
