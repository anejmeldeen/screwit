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
  hasMobileAssets,
  isPhone,
  splatUriFor,
} from "@/lib/device";
import { afterPaint, PERF_MARKS } from "@/lib/perf";
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
    <div className="flex h-full w-full items-center justify-center text-slate-500">
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

const CARD_LONG_EDGE_MM = 85.6; // credit-card long edge, default known length

export interface ViewerPanelProps {
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
  const splatReady = !!recon?.splat_uri;
  const [mode, setMode] = useState<"splat" | "mesh">("mesh");
  const [quality, setQuality] = useState<AssetQuality>("desktop");
  const [phone, setPhone] = useState(false);
  const [explode, setExplode] = useState(0);
  const [measureActive, setMeasureActive] = useState(false);
  const [scaleActive, setScaleActive] = useState(false);
  const [points, setPoints] = useState<PickedPoint[]>([]);
  const [measureResult, setMeasureResult] = useState<string | null>(null);
  const [scaleLength, setScaleLength] = useState<number>(CARD_LONG_EDGE_MM);
  const [cadVisible, setCadVisible] = useState(true);
  const [busyMsg, setBusyMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // The mesh has been drawn at least once (viewer:first-mesh fired). The splat
  // is never fetched before this, so the mesh always wins the "visible" budget.
  const [meshDrawn, setMeshDrawn] = useState(false);
  // The user explicitly asked for the splat view (phone only ever fetches the
  // splat after this). On desktop we also allow it after idle (see effect).
  const [splatRequested, setSplatRequested] = useState(false);

  useEffect(() => {
    setQuality(detectQuality());
    setPhone(isPhone());
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
  const mobileMeshUri = useMemo(
    () => (phone && recon?.mesh_mobile_uri ? recon.mesh_mobile_uri : null),
    [phone, recon],
  );
  const needsScale = recon?.scale?.method === "learned";

  const splatUri = useMemo(
    () => (recon ? splatUriFor(recon, quality) : null),
    [recon, quality],
  );
  const showMobileToggle = hasMobileAssets(recon);

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

  async function onPick(p: PickedPoint) {
    const next = [...points, p].slice(-2);
    setPoints(next);
    if (next.length === 2) {
      if (measureActive) await runMeasure(next);
      // scale is submitted by the explicit button so the user can type length
    }
  }

  async function runMeasure(pts: PickedPoint[]) {
    setErr(null);
    try {
      const res = await api.measure(scanId, {
        a_mm: pts[0].scan as unknown as never,
        b_mm: pts[1].scan as unknown as never,
      } as never);
      setMeasureResult(`${(res as { distance_mm: number }).distance_mm.toFixed(1)} mm`);
    } catch (e) {
      setErr(describe(e));
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
        known_length_mm: scaleLength,
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
    setMeasureResult(null);
  }

  const clearanceCheck = cadOverlay
    ? cadOverlay.cad.checks.find((c) => /clearance_mm$/.test(c.name))
    : undefined;

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 p-2 text-sm">
        {/* Splat/mesh toggle. The mesh button is always present; the splat
            button appears only when a splat exists. Selecting Splat marks it as
            user-requested, which is what releases the splat fetch. */}
        {splatReady ? (
          <div className="flex overflow-hidden rounded border border-slate-700">
            <button
              data-testid="view-toggle-mesh"
              className={`px-3 py-1 ${mode === "mesh" ? "bg-accent text-ink" : "bg-panel"}`}
              onClick={() => setMode("mesh")}
            >
              Mesh
            </button>
            <button
              data-testid="view-toggle-splat"
              className={`px-3 py-1 ${mode === "splat" ? "bg-accent text-ink" : "bg-panel"}`}
              onClick={() => {
                setSplatRequested(true);
                setMode("splat");
              }}
            >
              Splat
            </button>
          </div>
        ) : (
          <button
            data-testid="view-toggle-mesh"
            className="rounded border border-slate-700 bg-accent px-3 py-1 text-ink"
            disabled
          >
            Mesh view
          </button>
        )}

        {showMobileToggle ? (
          <div className="flex overflow-hidden rounded border border-slate-700">
            <button
              className={`px-2 py-1 ${quality === "desktop" ? "bg-slate-700" : "bg-panel"}`}
              onClick={() => setQuality("desktop")}
              title="Full-resolution assets"
            >
              HQ
            </button>
            <button
              className={`px-2 py-1 ${quality === "mobile" ? "bg-slate-700" : "bg-panel"}`}
              onClick={() => setQuality("mobile")}
              title="Lighter phone assets"
            >
              Phone
            </button>
          </div>
        ) : null}

        <label className="flex items-center gap-2">
          <span className="text-slate-400">Explode</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={explode}
            onChange={(e) => setExplode(parseFloat(e.target.value))}
            onMouseDown={() => mode === "splat" && setMode("mesh")}
          />
        </label>

        <button
          className={`rounded border px-2 py-1 ${
            measureActive ? "border-accent text-accent" : "border-slate-700"
          }`}
          onClick={() => {
            setMeasureActive((v) => !v);
            setScaleActive(false);
            resetPoints();
          }}
        >
          Measure
        </button>

        <button
          className={`rounded border px-2 py-1 ${
            scaleActive ? "border-accent text-accent" : "border-slate-700"
          } ${needsScale ? "ring-1 ring-amber-500" : ""}`}
          onClick={() => {
            setScaleActive((v) => !v);
            setMeasureActive(false);
            resetPoints();
          }}
          title="Set scale from a known length"
        >
          Set scale
        </button>

        {cadOverlay ? (
          <button
            className={`rounded border px-2 py-1 ${
              cadVisible ? "border-orange-500 text-orange-300" : "border-slate-700"
            }`}
            onClick={() => setCadVisible((v) => !v)}
          >
            CAD overlay
          </button>
        ) : null}
      </div>

      {/* Interaction hints / results */}
      {(measureActive || scaleActive || measureResult || err || busyMsg) && (
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-800 bg-slate-900/60 px-3 py-2 text-sm">
          {measureActive ? (
            <span className="text-slate-300">
              Click two points. {points.length}/2 picked.
            </span>
          ) : null}
          {measureResult ? (
            <span className="font-medium text-emerald-300">
              Distance: {measureResult}
            </span>
          ) : null}
          {scaleActive ? (
            <span className="flex items-center gap-2 text-slate-300">
              Click the two ends of a known length ({points.length}/2).
              <input
                type="number"
                step="0.01"
                value={scaleLength}
                onChange={(e) =>
                  setScaleLength(parseFloat(e.target.value) || CARD_LONG_EDGE_MM)
                }
                className="w-24 rounded border border-slate-700 bg-panel px-2 py-0.5"
              />
              mm
              <button
                disabled={points.length !== 2 || !!busyMsg}
                onClick={submitScale}
                className="rounded bg-accent px-2 py-0.5 text-ink disabled:opacity-40"
              >
                Apply scale
              </button>
            </span>
          ) : null}
          {clearanceCheck ? (
            <span className="text-orange-300">
              Clearance {Number(clearanceCheck.value).toFixed(2)} mm
              {clearanceCheck.input_source ? (
                <span className="text-slate-400"> ({clearanceCheck.input_source})</span>
              ) : null}
            </span>
          ) : null}
          {busyMsg ? <span className="text-slate-400">{busyMsg}</span> : null}
          {err ? <span className="text-rose-300">{err}</span> : null}
        </div>
      )}

      {/* The 3D canvas */}
      <div className="relative min-h-0 flex-1">
        {recon ? (
          <Viewer
            recon={recon}
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
          <div className="flex h-full items-center justify-center text-slate-500">
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
