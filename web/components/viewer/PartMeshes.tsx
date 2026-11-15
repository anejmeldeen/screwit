"use client";

import { useEffect, useMemo, useRef } from "react";
import { useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Box3, Color, Group, Mesh, MeshStandardMaterial, Vector3 } from "three";
import type { Part } from "@/lib/contracts.gen";
import { assetUrl } from "@/lib/storage";
import { configureGltfLoader } from "./gltf";
import { markOnce, PERF_MARKS } from "@/lib/perf";

import { explodeOffsets, type FeaturePick, type MeasureMode } from "@/lib/editorGeometry";
import { FeatureSurface } from "./FeatureSurface";
import type { PickedPoint } from "./MeasureLayer";
import { partColor } from "@/lib/palette";
export { partColor };

// Load one part GLB with the meshopt-enabled loader. On a 404 (or any load
// error) we render nothing for that part and the caller shows a message, so the
// viewer never crashes.
function usePartGltf(url: string) {
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    configureGltfLoader(loader as GLTFLoader);
  });
  return { gltf };
}

function PartMesh({
  part,
  index,
  selected,
  hovered,
  explode,
  onHover,
  onSelect,
  registerCentroid,
  offset, measureMode, featurePicks, onFeaturePick, onMeasurePick,
}: {
  part: Part;
  index: number;
  selected: boolean;
  hovered: boolean;
  explode: number;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
  registerCentroid: (id: string, worldCentroid: Vector3) => void;
  offset: Vector3;
  measureMode: MeasureMode | null;
  featurePicks: FeaturePick[];
  onFeaturePick: (p: FeaturePick) => void;
  onMeasurePick: (p: PickedPoint) => void;
}) {
  const { gltf } = usePartGltf(assetUrl(part.mesh_uri));
  const groupRef = useRef<Group>(null);

  const scene = useMemo(() => {
    const clone = gltf.scene.clone(true);
    clone.traverse((object) => {
      if (!(object instanceof Mesh)) return;
      // Own the display material: glTF defaults to metalness=1, and baked
      // vertex colors otherwise multiply (darken) the part palette.
      object.material = new MeshStandardMaterial({
        color: partColor(index, part.color_source_id ?? part.part_id),
        metalness: 0, roughness: 1,
      });
    });
    return clone;
  }, [gltf, index, part.color_source_id, part.part_id]);
  useEffect(() => () => {
    scene.traverse((object) => {
      if (object instanceof Mesh) (object.material as MeshStandardMaterial).dispose();
    });
  }, [scene]);
  const baseColor = useMemo(() => new Color(partColor(index, part.color_source_id ?? part.part_id)), [index, part.color_source_id, part.part_id]);

  // The first part mesh to draw marks the viewer's first-mesh timestamp.
  useEffect(() => {
    markOnce(PERF_MARKS.firstMesh);
  }, [scene]);

  useEffect(() => {
    scene.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) {
        for (const mat of (Array.isArray(m.material) ? m.material : [m.material]) as { color?: Color; emissive?: Color }[]) {
          if (mat.color) {
            mat.color.copy(baseColor);
          }
          if (mat.emissive) {
            mat.emissive.copy(
              selected
                ? new Color("#302a20")
                : hovered
                  ? new Color("#18160f")
                  : new Color("#000000"),
            );
          }
        }
      }
    });
  }, [scene, baseColor, selected, hovered]);

  // Report the part centroid in WORLD space (for focus animation) whenever the
  // parent transform settles.
  useEffect(() => {
    const g = groupRef.current;
    if (!g) return;
    g.updateWorldMatrix(true, true);
    const box = new Box3().setFromObject(g);
    const center = new Vector3();
    box.getCenter(center);
    registerCentroid(part.part_id, center);
  }, [scene, explode, part.part_id, registerCentroid]);

  const displacement = offset.clone().multiplyScalar(explode);

  return (
    <group
      ref={groupRef}
      position={displacement}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(part.part_id);
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        onHover(null);
      }}
      onClick={(e) => {
        if (measureMode || e.delta > 4) return;
        e.stopPropagation();
        onSelect(part.part_id);
      }}
    >
      <FeatureSurface scene={scene} partId={part.part_id} mode={measureMode}
        picks={featurePicks} onFeature={onFeaturePick} onPoint={onMeasurePick} />
      {selected && !measureMode ? (
        // Simple selection outline: a slightly larger wireframe box.
        <SelectionOutline part={part} />
      ) : null}
    </group>
  );
}

function SelectionOutline({ part }: { part: Part }) {
  const center = part.obb.center as [number, number, number];
  const extent = part.obb.extent as [number, number, number];
  const rotation = part.obb.rotation_quat as [number, number, number, number];
  return (
    <mesh raycast={() => {}} position={center}
      quaternion={rotation}>
      <boxGeometry args={[extent[0] * 1.04, extent[1] * 1.04, extent[2] * 1.04]} />
      <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.5} />
    </mesh>
  );
}

export function PartMeshes({
  parts,
  visible,
  selectedId,
  hoveredId,
  explode,
  onHover,
  onSelect,
  registerCentroid,
  hiddenPartIds = [], measureMode = null, featurePicks = [], onFeaturePick = () => {}, onMeasurePick = () => {},
}: {
  hiddenPartIds?: string[];
  measureMode?: MeasureMode | null;
  featurePicks?: FeaturePick[];
  onFeaturePick?: (p: FeaturePick) => void;
  onMeasurePick?: (p: PickedPoint) => void;
  parts: Part[];
  visible: boolean;
  selectedId: string | null;
  hoveredId: string | null;
  explode: number;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
  registerCentroid: (id: string, worldCentroid: Vector3) => void;
}) {
  const offsets = useMemo(() => explodeOffsets(parts), [parts]);
  if (!visible) return null;
  return (
    <group>
      {parts.map((p, i) => hiddenPartIds.includes(p.part_id) ? null : (
        <PartMesh
          key={p.part_id}
          part={p}
          offset={offsets.get(p.part_id)!}
          measureMode={measureMode} featurePicks={featurePicks}
          onFeaturePick={onFeaturePick} onMeasurePick={onMeasurePick}
          index={i}
          selected={selectedId === p.part_id}
          hovered={hoveredId === p.part_id}
          explode={explode}
          onHover={onHover}
          onSelect={onSelect}
          registerCentroid={registerCentroid}
        />
      ))}
    </group>
  );
}
