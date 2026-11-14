"use client";

import { useEffect, useMemo, useRef } from "react";
import { useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Box3, Color, Group, Mesh, Vector3 } from "three";
import type { Part } from "@/lib/contracts.gen";
import { assetUrl } from "@/lib/storage";
import { configureGltfLoader } from "./gltf";
import { markOnce, PERF_MARKS } from "@/lib/perf";

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
}: {
  part: Part;
  index: number;
  selected: boolean;
  hovered: boolean;
  explode: number;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
  registerCentroid: (id: string, worldCentroid: Vector3) => void;
}) {
  const { gltf } = usePartGltf(assetUrl(part.mesh_uri));
  const groupRef = useRef<Group>(null);

  const scene = useMemo(() => gltf.scene.clone(true), [gltf]);
  const baseColor = useMemo(() => new Color(partColor(index)), [index]);

  // The first part mesh to draw marks the viewer's first-mesh timestamp.
  useEffect(() => {
    markOnce(PERF_MARKS.firstMesh);
  }, [scene]);

  // Explode direction: unit vector from scan origin to the part centroid, in
  // scan-frame mm. Offset grows with the slider.
  const dir = useMemo(() => {
    const c = part.centroid_mm as unknown as number[];
    const v = new Vector3(c[0], c[1], c[2]);
    const len = v.length();
    return len > 1e-6 ? v.clone().normalize() : new Vector3(0, 0, 0);
  }, [part.centroid_mm]);

  useEffect(() => {
    scene.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) {
        const mat = (m.material as { color?: Color; emissive?: Color }) ?? {};
        if (mat.color) {
          mat.color.copy(baseColor);
        }
        if (mat.emissive) {
          mat.emissive.copy(
            selected
              ? new Color("#ffffff")
              : hovered
                ? new Color("#334155")
                : new Color("#000000"),
          );
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

  const offset = dir.clone().multiplyScalar(explode * 80); // scan-frame mm

  return (
    <group
      ref={groupRef}
      position={[offset.x, offset.y, offset.z]}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(part.part_id);
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        onHover(null);
      }}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(part.part_id);
      }}
    >
      <primitive object={scene} />
      {selected ? (
        // Simple selection outline: a slightly larger wireframe box.
        <SelectionOutline target={scene} />
      ) : null}
    </group>
  );
}

function SelectionOutline({ target }: { target: Group }) {
  const box = useMemo(() => {
    const b = new Box3().setFromObject(target);
    const size = new Vector3();
    const center = new Vector3();
    b.getSize(size);
    b.getCenter(center);
    return { size, center };
  }, [target]);
  return (
    <mesh position={[box.center.x, box.center.y, box.center.z]}>
      <boxGeometry args={[box.size.x * 1.04, box.size.y * 1.04, box.size.z * 1.04]} />
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
}: {
  parts: Part[];
  visible: boolean;
  selectedId: string | null;
  hoveredId: string | null;
  explode: number;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
  registerCentroid: (id: string, worldCentroid: Vector3) => void;
}) {
  if (!visible) return null;
  return (
    <group>
      {parts.map((p, i) => (
        <PartMesh
          key={p.part_id}
          part={p}
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
