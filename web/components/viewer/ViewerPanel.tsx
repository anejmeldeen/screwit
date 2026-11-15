"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type {
  CadResult,
  Part,
  Reconstruction,
} from "@/lib/contracts.gen";
import { api, ApiError } from "@/lib/api";
import { assetUrl } from "@/lib/storage";
import {
  AssetQuality,
  detectQuality,
  isPhone,
  splatUriFor,
} from "@/lib/device";
import { afterPaint, PERF_MARKS } from "@/lib/perf";
import { edgeStats, featureDistance, type FeaturePick, type MeasureMode } from "@/lib/editorGeometry";
import type { PickedPoint } from "./MeasureLayer";
import type { ViewerCadOverlay } from "./Viewer";

// The heavy three.js viewer is a separate chunk (ssr:false) so the shell and
// landing carry no 3D code. We kick the import as soon as this module loads
// (prewarmViewer) so the three.js chunk downloads in PARALLEL with the shell
// and the scan API calls, instead of only after <Viewer> first renders.
const importViewer = () => import("./Viewer");
const Viewer = dynamic(importViewer, {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center text-muted">
      Loading 3D viewer…
    </div>
  ),
});

// Fire the viewer + whole-mesh chunk downloads immediately on the client. Safe
// to call repeatedly; the dynamic import is memoised by the bundler.
function prewarmViewer() {
  if (typeof window === "undefined") return;
  void importViewer();
  void import("./WholeMesh");
}

const NO_HIDDEN_PARTS: string[] = [];
type Reading = { label: string; length: number; delta: number[]; travel?: number[] };
const CARD_LONG_EDGE_MM = 85.6; // credit-card long edge, default known length

export interface ViewerPanelProps {
  isolated?: boolean;
  combining?: boolean;
  hiddenPartIds?: string[];
  scanId: string;
  recon: Reconstruction | null;
  parts: Part[];
  selectedId: string | null;
  hoveredId: string | null;
  focusPartId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
  cadOverlay: ViewerCadOverlay | null;
  onScaled: () => void; // reconstruction changed; parent should refetch
  onFpsSample?: (fps: number) => void;
}

export function ViewerPanel({
  scanId,
  isolated = false, combining = false,
  hiddenPartIds = NO_HIDDEN_PARTS,
  recon,
  parts,
  selectedId,
  hoveredId,
  focusPartId,
  onHover,
  onSelect,
  cadOverlay,
  onScaled,
  onFpsSample,
}: ViewerPanelProps) {
  const splatReady = !isolated && !!recon?.splat_uri;
  const [mode, setMode] = useState<"splat" | "mesh">("mesh");
  const [quality, setQuality] = useState<AssetQuality>("desktop");
  const [phone, setPhone] = useState(false);
  const [explode, setExplode] = useState(0);
  const [measureActive, setMeasureActive] = useState(false);
  const [measureMode, setMeasureMode] = useState<MeasureMode>("point");
  const [featurePicks, setFeaturePicks] = useState<FeaturePick[]>([]);
  const [scaleActive, setScaleActive] = useState(false);
  const [points, setPoints] = useState<PickedPoint[]>([]);
  const [measureResult, setMeasureResult] = useState<Reading | null>(null);
  const [scaleLength, setScaleLength] = useState(String(CARD_LONG_EDGE_MM));
  const [cadVisible, setCadVisible] = useState(true);
  const [busyMsg, setBusyMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // The mesh has been drawn at least once (viewer:first-mesh fired). The splat
  // is never fetched before this, so the mesh always wins the "visible" budget.
  useEffect(() => {
    if (combining) { setMeasureActive(false); setScaleActive(false); setMode("mesh"); }
  }, [combining]);
  const [meshDrawn, setMeshDrawn] = useState(false);
  // The user explicitly asked for the splat view (phone only ever fetches the
  // splat after this). On desktop we also allow it after idle (see effect).
  const [splatRequested, setSplatRequested] = useState(false);

  useEffect(() => {
    setPhone(isPhone());
    setQuality(detectQuality());
    prewarmViewer();
  }, []);

  // Watch for the first-mesh mark so we can release the splat afterwards.
  useEffect(() => {
    if (meshDrawn) return;
    const id = setInterval(() => {
      if (
        typeof performance !== "undefined" &&
        performance.getEntriesByName(PERF_MARKS.firstMesh, "mark").length
      ) {
        setMeshDrawn(true);
        clearInterval(id);
      }
    }, 150);
    return () => clearInterval(id);
  }, [meshDrawn]);

  // Desktop convenience: once the mesh is drawn and the browser is idle, allow
  // the splat so a desktop user can toggle to it with no wait. Phones NEVER
  // auto-allow — they only fetch the splat when the user selects Splat.
  useEffect(() => {
    if (phone || splatRequested || !meshDrawn || !splatReady) return;
    afterPaint(() => setSplatRequested(true));
  }, [phone, splatRequested, meshDrawn, splatReady]);

  // Default to the MESH view: it is the budget-critical "mesh visible under 5 s"
  // path and keeps Spark (~950 KB) and the .spz out of the initial load. The
  // splat is fetched only after the mesh has drawn AND the user chose Splat
  // (desktop also after idle); a phone never downloads the splat unasked.
  const splatAllowed = splatReady && meshDrawn && splatRequested;

  // On phones, render the light whole-object mobile mesh for the mesh view when
  // one exists (keeps first-mesh under budget on slow 4G); desktop uses the
  // per-part meshes (picking, colours, explode).
  const mobileMeshUri = null; // One editable parts view on every device.
  const needsScale = recon?.scale?.method === "learned";
  useEffect(() => {
    setPoints([]); setFeaturePicks([]); setMeasureResult(null);
  }, [parts, hiddenPartIds, recon?.reconstruction_revision]);

  const splatUri = useMemo(
    () => (recon ? splatUriFor(recon, quality) : null),
    [recon, quality],
  );

  // Warm the mesh in the HTTP cache the moment its URL is known, in parallel
  // with the viewer JS download, so GLTFLoader reuses the bytes instead of
  // starting the fetch only after three.js has parsed. Phone: the light
  // whole-object mesh; desktop: the first part GLB (the rest stream after).
  const prefetchMeshUri = mobileMeshUri ?? parts[0]?.mesh_uri ?? null;
  useEffect(() => {
    if (!prefetchMeshUri) return;
    const url = assetUrl(prefetchMeshUri);
    const ctrl = new AbortController();
    // Low-priority cache warm; GLTFLoader's later GET hits the HTTP cache.
    fetch(url, { signal: ctrl.signal, cache: "force-cache" }).catch(() => {});
    return () => ctrl.abort();
  }, [prefetchMeshUri]);

  function onPick(p: PickedPoint) {
    const next = points.length === 2 ? [p] : [...points, p];
    setPoints(next);
    setMeasureResult(null);
    if (next.length === 2) {
      if (measureActive) runMeasure(next);
      // scale is submitted by the explicit button so the user can type length
    }
  }

  function runMeasure(pts: PickedPoint[]) {
    const delta = pts[1].scan.map((v, i) => v - pts[0].scan[i]);
    setMeasureResult({ label: measureMode === "point" ? "Distance" : "Shortest gap", length: Math.hypot(...delta), delta });
  }

  function onFeaturePick(pick: FeaturePick) {
    setErr(null);
    if (measureMode === "edge-length") {
      setFeaturePicks([pick]);
      const s = edgeStats(pick.feature);
      setPoints([]);
      setMeasureResult({ label: "Edge length", length: s.length, travel: [s.x, s.y, s.z], delta: [s.delta.x, s.delta.y, s.delta.z] });
      return;
    }
    if (featurePicks.length === 1 && featurePicks[0].partId === pick.partId && featurePicks[0].feature.id === pick.feature.id) return;
    const next = featurePicks.length === 1 ? [...featurePicks, pick] : [pick];
    setFeaturePicks(next); setPoints([]); setMeasureResult(null);
    if (next.length === 2) {
      const [a, b] = featureDistance(next[0].feature, next[1].feature);
      const pts = [a, b].map(v => ({ scan: v.toArray() as [number, number, number] }));
      setPoints(pts); runMeasure(pts);
    }
  }

  async function submitScale() {
    if (points.length !== 2 || !recon) return;
    setBusyMsg("Rescaling…");
    setErr(null);
    try {
      await api.scale(scanId, {
        point_a_mm: points[0].scan as unknown as never,
        point_b_mm: points[1].scan as unknown as never,
        known_length_mm: Number(scaleLength),
        expected_revision: recon.reconstruction_revision,
      } as never);
      setPoints([]);
      setScaleActive(false);
      onScaled();
    } catch (e) {
      setErr(describe(e));
    } finally {
      setBusyMsg(null);
    }
  }

  function resetPoints() {
    setPoints([]);
    setFeaturePicks([]);
    setMeasureResult(null);
  }

  const requiredPicks = measureActive && measureMode === "edge-length" ? 1 : 2;
  const picked = scaleActive || measureMode === "point" ? points.length : featurePicks.length;
  const unit = scaleActive || measureMode === "point" ? "point" : measureMode === "face-distance" ? "face" : "edge";
  const validLength = Number.isFinite(Number(scaleLength)) && Number(scaleLength) > 0;
  const clearanceCheck = cadOverlay
    ? cadOverlay.cad.checks.find((c) => /clearance_mm$/.test(c.name))
    : undefined;

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="scene-tools"><strong>Scene</strong>
        <div className="scene-display-controls" role="group" aria-label="Display controls">
        {/* Splat/mesh toggle. The mesh button is always present; the splat
            button appears only when a splat exists. Selecting Splat marks it as
            user-requested, which is what releases the splat fetch. */}
        {splatReady ? (
          <div className="flex overflow-hidden rounded border border-line">
            <button
              data-testid="view-toggle-mesh"
              aria-pressed={mode === "mesh"}
              className={`px-3 py-1 ${mode === "mesh" ? "bg-accent text-paper" : "bg-panel"}`}
              onClick={() => setMode("mesh")}
            >
              Parts view
            </button>
            <button
              data-testid="view-toggle-splat"
              aria-pressed={mode === "splat"}
              disabled={hiddenPartIds.length > 0}
              title={hiddenPartIds.length > 0 ? "Show all components to use splat view" : "Splat view"}
              className={`px-3 py-1 ${mode === "splat" ? "bg-accent text-paper" : "bg-panel"}`}
              onClick={() => {
                setSplatRequested(true);
                setMeasureActive(false); setScaleActive(false); setExplode(0); resetPoints();
                setMode("splat");
              }}
            >
              Original scan
            </button>
          </div>
        ) : null}
        {!isolated ? (
        <label className="flex items-center gap-2">
          <span className="text-muted">Separate parts</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={explode}
            onChange={(e) => {
              setMode("mesh");
              setExplode(parseFloat(e.target.value));
            }}
            disabled={parts.length < 2 || measureActive || scaleActive}
            aria-label="Explode components"
            aria-valuetext={`${Math.round(explode * 100)}% separated`}
            title={parts.length < 2 ? "Explode needs at least two parts" : "Separate parts"}
          />
        </label>
        ) : null}
        </div>

        <button
          className={`rounded border px-2 py-1 ${
            measureActive ? "border-accent text-accent" : "border-line"
          }`}
          aria-pressed={measureActive}
          onClick={() => {
            setMeasureActive((v) => !v);
            setMode("mesh"); setExplode(0);
            setScaleActive(false);
            resetPoints();
          }}
        >
          Measure
        </button>

        <button
          className={`rounded border px-2 py-1 ${
            scaleActive ? "border-accent text-accent" : "border-line"
          } ${needsScale ? "ring-1 ring-amber-500" : ""}`}
          aria-pressed={scaleActive}
          onClick={() => {
            setScaleActive((v) => !v);
            setMeasureMode("point");
            setMode("mesh"); setExplode(0);
            setMeasureActive(false);
            resetPoints();
          }}
          disabled={isolated}
          title={isolated ? "Set scale in the assembly window" : "Set scale from a known length"}
        >
          Set scale
        </button>

        {cadOverlay ? (
          <button
            className={`rounded border px-2 py-1 ${
              cadVisible ? "border-orange-500 text-orange-800" : "border-line"
            }`}
            onClick={() => setCadVisible((v) => !v)}
          >
            Show replacement
          </button>
        ) : null}
      </div>

      {/* Interaction hints / results */}
      {(measureActive || scaleActive || measureResult || err || busyMsg) && (
        <div className="scene-instrument">
          {measureActive && <div className="instrument-modes" role="group" aria-label="Measurement type">
            {([["point", "Point to point"], ["edge-length", "Edge length"], ["edge-distance", "Edge to edge"], ["face-distance", "Face to face"]] as const).map(([mode, label]) =>
              <button key={mode} aria-pressed={measureMode === mode} onClick={() => { setMeasureMode(mode); resetPoints(); }}>{label}</button>)}
          </div>}
          {(measureActive || scaleActive) && <div className="instrument-pick">
            <span className="pick-dots" aria-hidden="true">{Array.from({length: requiredPicks}, (_, i) => <i key={i} data-filled={i < picked} />)}</span>
            <span role="status">{scaleActive ? (picked === 2 ? "Reference endpoints selected" : "Pick the ends of a known length") :
              picked === requiredPicks ? "Selection complete" : unit === "point" ? "Pick two points on the surface" :
              requiredPicks === 1 ? "Pick the highlighted edge to trace its length" : `Pick two ${unit}s to measure their shortest gap`}</span>
            <small>{picked}/{requiredPicks}</small>
            {picked > 0 && <button disabled={!!busyMsg} onClick={resetPoints}>Clear</button>}
          </div>}
          {measureResult && <div className="instrument-reading" aria-live="polite">
            <div><span>{measureResult.label}</span><strong>{measureResult.length.toFixed(2)} <small>mm</small></strong></div>
            <dl>{[...(measureResult.travel ? [["Axis travel", measureResult.travel] as const] : []), ["Displacement", measureResult.delta] as const].map(([label, values]) =>
              <div key={label}><dt>{label} (mm)</dt><dd>{values.map((v, i) => <span key={i}>{"XYZ"[i]} <b>{v.toFixed(2)}</b></span>)}</dd></div>)}</dl>
          </div>}
          {scaleActive && <form className="instrument-scale" onSubmit={event => { event.preventDefault(); if (picked === 2 && validLength && !busyMsg) void submitScale(); }}>
            <label>Reference length<span><input aria-label="Reference length in millimetres" type="number" step="any" min="0" required value={scaleLength} onChange={e => setScaleLength(e.target.value)} /><small>mm</small></span></label>
            <button disabled={picked !== 2 || !validLength || !!busyMsg}>Apply scale</button>
          </form>}
          {clearanceCheck ? (
            <span className="text-orange-800">
              Clearance {Number(clearanceCheck.value).toFixed(2)} mm
              {clearanceCheck.input_source ? (
                <span className="text-muted"> ({clearanceCheck.input_source})</span>
              ) : null}
            </span>
          ) : null}
          {busyMsg ? <span className="text-muted">{busyMsg}</span> : null}
          {err ? <span className="text-rose-800">{err}</span> : null}
        </div>
      )}

      {/* The 3D canvas */}
      <div className="relative min-h-0 flex-1">
        {recon ? (
          <Viewer
            recon={recon}
            hiddenPartIds={hiddenPartIds}
            measureMode={measureMode} featurePicks={featurePicks} onFeaturePick={onFeaturePick}
            parts={parts}
            splatUri={splatUri}
            mobileMeshUri={mobileMeshUri}
            mode={mode}
            splatAllowed={splatAllowed}
            explode={explode}
            measureActive={measureActive || scaleActive}
            measurePoints={points}
            onMeasurePick={onPick}
            selectedId={selectedId}
            hoveredId={hoveredId}
            onHover={onHover}
            onSelect={onSelect}
            focusPartId={focusPartId}
            cadOverlay={cadOverlay}
            cadOverlayVisible={cadVisible}
            onFpsSample={onFpsSample}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-muted">
            {recon ? "Preparing 3D viewer…" : "Waiting for reconstruction…"}
          </div>
        )}
      </div>
    </div>
  );
}

function describe(e: unknown): string {
  if (e instanceof ApiError) return `${e.message} (${e.status})`;
  return String((e as Error)?.message ?? e);
}

export default ViewerPanel;
