"use client";

import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Matrix4, Vector3, PerspectiveCamera, GridHelper } from "three";
import { OrbitControls as OrbitControlsImpl } from "three/examples/jsm/controls/OrbitControls.js";
import type { CadResult, Part, Reconstruction } from "@/lib/contracts.gen";
import { scanToWebMatrix } from "@/lib/transform";
import { PartMeshes } from "./PartMeshes";
import { MeasureLayer, PickedPoint } from "./MeasureLayer";

// Spark (~950 KB) loads ONLY when the splat view is shown; the STL loader loads
// ONLY when a CAD overlay is requested. Both are React.lazy so their chunks are
// fetched on first render inside the Canvas, not when the viewer mounts.
const SplatLayer = lazy(() =>
  import("./SplatLayer").then((m) => ({ default: m.SplatLayer })),
);
const CadOverlay = lazy(() =>
  import("./CadOverlay").then((m) => ({ default: m.CadOverlay })),
);
const WholeMesh = lazy(() =>
  import("./WholeMesh").then((m) => ({ default: m.WholeMesh })),
);

export interface ViewerCadOverlay {
  cad: CadResult;
  placement: { translate_mm: number[]; rotation_quat: number[] };
}

export interface ViewerProps {
  recon: Reconstruction;
  parts: Part[];
  splatUri: string | null; // already resolved for quality
  // Whole-object mobile mesh (reconstruction.mesh_mobile_uri). When set, the
  // mesh view renders this single light GLB instead of the heavy per-part GLBs
  // (the phone "mesh visible under 5 s" path). Null => per-part meshes.
  mobileMeshUri: string | null;
  mode: "splat" | "mesh";
  // Gate: only mount the splat (and fetch Spark + the .spz) when true. The
  // parent sets this once the mesh has drawn AND the user selected Splat, so a
  // phone never downloads the splat unless asked.
  splatAllowed: boolean;
  explode: number;
  measureActive: boolean;
  measurePoints: PickedPoint[];
  onMeasurePick: (p: PickedPoint) => void;
  selectedId: string | null;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
  focusPartId: string | null;
  cadOverlay: ViewerCadOverlay | null;
  cadOverlayVisible: boolean;
  onFpsSample?: (fps: number) => void;
}

// Collect per-part world centroids so a chip click can focus the camera.
function useCentroidRegistry() {
  const map = useRef(new Map<string, Vector3>());
  const register = useCallback((id: string, c: Vector3) => {
    map.current.set(id, c.clone());
  }, []);
  return { map, register };
}

// Orbit controls built directly on three/examples (not drei), so the mesh path
// does not pull drei's Grid/shaderMaterial/three-stdlib bundle. Registered as
// the scene's default controls (set({controls})) so FocusController can read
// and update the same instance via useThree(s => s.controls).
function OrbitControlsLite() {
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const invalidate = useThree((s) => s.invalidate);
  const set = useThree((s) => s.set);
  const get = useThree((s) => s.get);

  useEffect(() => {
    const controls = new OrbitControlsImpl(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    const onChange = () => invalidate();
    controls.addEventListener("change", onChange);
    const prev = get().controls;
    set({ controls: controls as unknown as typeof prev });
    return () => {
      controls.removeEventListener("change", onChange);
      controls.dispose();
      set({ controls: prev });
    };
  }, [camera, gl, invalidate, set, get]);

  // Damping needs an update each frame while active.
  useFrame(() => {
    const c = get().controls as unknown as { update?: () => void } | null;
    c?.update?.();
  });
  return null;
}

// Smoothly move the OrbitControls target + camera toward a part centroid.
function FocusController({
  target,
}: {
  target: Vector3 | null;
}) {
  const controls = useThree(
    (s) => s.controls as unknown as { target: Vector3; update: () => void } | null,
  );
  const camera = useThree((s) => s.camera);
  const goal = useRef<Vector3 | null>(null);
  if (target && goal.current !== target) goal.current = target;

  useFrame(() => {
    if (!goal.current || !controls) return;
    const t = controls.target as Vector3;
    const next = t.clone().lerp(goal.current, 0.15);
    t.copy(next);
    // Pull the camera toward a comfortable offset from the target.
    const offset = new Vector3(0.25, 0.2, 0.25);
    const desired = goal.current.clone().add(offset);
    camera.position.lerp(desired, 0.12);
    controls.update();
    if (t.distanceTo(goal.current) < 0.001) goal.current = null;
  });
  return null;
}

function FpsSampler({ onSample }: { onSample?: (fps: number) => void }) {
  const acc = useRef({ frames: 0, t: 0 });
  useFrame((_, dt) => {
    if (!onSample) return;
    acc.current.frames += 1;
    acc.current.t += dt;
    if (acc.current.t >= 0.25) {
      onSample(acc.current.frames / acc.current.t);
      acc.current.frames = 0;
      acc.current.t = 0;
    }
  });
  return null;
}

export function Viewer(props: ViewerProps) {
  const {
    recon,
    parts,
    splatUri,
    mobileMeshUri,
    mode,
    splatAllowed,
    explode,
    measureActive,
    measurePoints,
    onMeasurePick,
    selectedId,
    hoveredId,
    onHover,
    onSelect,
    focusPartId,
    cadOverlay,
    cadOverlayVisible,
    onFpsSample,
  } = props;

  const parentMatrix = useMemo(() => scanToWebMatrix(recon), [recon]);
  const gridHelper = useMemo(() => {
    const g = new GridHelper(2, 40, 0x334155, 0x1f2937);
    return g;
  }, []);
  const { map: centroids, register } = useCentroidRegistry();
  const [focusTarget, setFocusTarget] = useState<Vector3 | null>(null);

  // When the focus part changes, resolve its registered world centroid.
  const lastFocus = useRef<string | null>(null);
  if (focusPartId && focusPartId !== lastFocus.current) {
    lastFocus.current = focusPartId;
    const c = centroids.current.get(focusPartId);
    if (c) setFocusTarget(c.clone());
  }
  if (!focusPartId) lastFocus.current = null;

  const placement = cadOverlay?.placement ?? null;

  return (
    <Canvas
      data-testid="viewer-canvas"
      // Render on demand when idle; orbit/measure interactions invalidate.
      frameloop="demand"
      dpr={[1, 2]}
      gl={{ antialias: false }}
      camera={{ position: [0.3, 0.25, 0.3], near: 0.001, far: 100, fov: 50 }}
      onCreated={({ camera }) => {
        (camera as PerspectiveCamera).lookAt(0, 0.05, 0);
      }}
    >
      <color attach="background" args={["#f3e5ac"]} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[1, 2, 1]} intensity={1.1} />
      <directionalLight position={[-1, 1, -1]} intensity={0.4} />

      {/* Grid at the scan-frame origin (web frame: XZ plane, Y up). Plain
          three GridHelper (no drei shader Grid) to keep the mesh-path bundle
          small; 2 m extent, 40 divisions = 0.05 m cells. */}
      <primitive object={gridHelper} />

      {/* ONE parent group applying scan_to_web; every asset is a child. */}
      <group matrix={parentMatrix} matrixAutoUpdate={false}>
        <Suspense fallback={null}>
          {/* The splat mounts only once allowed: on a phone that means the user
              selected Splat (after the mesh drew); on desktop also after idle.
              Mounting is what fetches Spark + the .spz, so a phone never
              downloads the splat unasked. It is hidden until splat mode. */}
          {splatAllowed && splatUri ? (
            <SplatLayer url={splatUri} visible={mode === "splat"} />
          ) : null}
        </Suspense>

        <Suspense fallback={null}>
          {mobileMeshUri ? (
            // Phone path: one light whole-object mesh.
            <WholeMesh url={mobileMeshUri} visible={mode === "mesh"} />
          ) : (
            // Desktop path: per-part meshes with picking, colours and explode.
            <PartMeshes
              parts={parts}
              visible={mode === "mesh"}
              selectedId={selectedId}
              hoveredId={hoveredId}
              explode={explode}
              onHover={onHover}
              onSelect={onSelect}
              registerCentroid={register}
            />
          )}
        </Suspense>

        {cadOverlay && placement ? (
          <Suspense fallback={null}>
            <CadOverlay
              cad={cadOverlay.cad}
              placement={placement}
              visible={cadOverlayVisible}
            />
          </Suspense>
        ) : null}

        <MeasureLayer
          active={measureActive}
          parentMatrix={parentMatrix}
          points={measurePoints}
          onPick={onMeasurePick}
        />
      </group>

      <OrbitControlsLite />
      <FocusController target={focusTarget} />
      <FpsSampler onSample={onFpsSample} />
    </Canvas>
  );
}

export default Viewer;
