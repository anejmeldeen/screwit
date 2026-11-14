"use client";

import { useEffect, useMemo, useState } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import { BufferGeometry, DoubleSide, Float32BufferAttribute, Matrix4, Mesh, Object3D } from "three";
import { FeaturePick, MeasureMode, MeshFeature, meshTopology, nearestEdge } from "@/lib/editorGeometry";
import type { PickedPoint } from "./MeasureLayer";

export function FeatureSurface({ scene, partId, mode, picks, onFeature, onPoint }: {
  scene: Object3D; partId: string; mode: MeasureMode | null;
  picks: FeaturePick[]; onFeature: (pick: FeaturePick) => void; onPoint: (point: PickedPoint) => void;
}) {
  const [hover, setHover] = useState<MeshFeature | null>(null);
  const meshes = useMemo(() => {
    const result = new Map<string, { matrix: Matrix4; topology: ReturnType<typeof meshTopology> }>();
    if (!mode) return result;
    scene.traverse(o => {
      if (!(o as Mesh).isMesh) return;
      const mesh = o as Mesh;
      // Relative to the part's wrapper, never to its exploded/world position.
      const chain: Object3D[] = []; let node: Object3D | null = mesh;
      while (node) { chain.unshift(node); if (node === scene) break; node = node.parent; }
      const matrix = new Matrix4(); chain.forEach(n => { n.updateMatrix(); matrix.multiply(n.matrix); });
      const geometry = mode === "point" ? null : mesh.geometry.clone().applyMatrix4(matrix);
      const topology = geometry ? meshTopology(geometry) : { edges: [], faces: [], triangleFaces: [] };
      geometry?.dispose();
      for (const f of [...topology.edges, ...topology.faces]) f.id = `${mesh.uuid}:${f.id}`;
      result.set(mesh.uuid, { matrix, topology });
    });
    return result;
  }, [scene, mode]);
  function hit(e: ThreeEvent<MouseEvent | PointerEvent>) {
    const data = meshes.get(e.object.uuid); if (!data) return null;
    const point = e.object.worldToLocal(e.point.clone()).applyMatrix4(data.matrix);
    const feature = mode === "face-distance"
      ? data.topology.triangleFaces[e.faceIndex ?? -1] ?? null
      : nearestEdge(data.topology.edges, point);
    return { point, feature };
  }
  const selected = picks.filter(p => p.partId === partId).map(p => p.feature);
  const allEdges = useMemo(() => [...meshes.values()].flatMap(m => m.topology.edges), [meshes]);
  return <group>
    <primitive object={scene}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        if (!mode || mode === "point") return;
        e.stopPropagation(); setHover(hit(e)?.feature ?? null);
      }}
      onPointerOut={() => setHover(null)}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        if (!mode || e.delta > 4) return;
        e.stopPropagation(); const h = hit(e); if (!h) return;
        if (mode === "point") onPoint({ scan: h.point.toArray() as [number, number, number] });
        else if (h.feature) onFeature({ partId, feature: h.feature });
      }} />
    {mode?.startsWith("edge") ? <FeatureLines features={allEdges} color="#94a3b8" /> : null}
    {mode ? selected.map((f, i) => <FeatureHighlight key={f.id} feature={f} color={i === 0 ? "#34d399" : "#f472b6"} />) : null}
    {mode && hover && !selected.includes(hover) ? <FeatureHighlight feature={hover} color="#fbbf24" /> : null}
  </group>;
}

function FeatureLines({ features, color }: { features: MeshFeature[]; color: string }) {
  const geometry = useMemo(() => {
    const coords: number[] = [];
    for (const f of features) for (let i = 1; i < f.points.length; i++) coords.push(...f.points[i - 1].toArray(), ...f.points[i].toArray());
    return new BufferGeometry().setAttribute("position", new Float32BufferAttribute(coords, 3));
  }, [features]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return <lineSegments geometry={geometry} raycast={() => {}} renderOrder={5}>
    <lineBasicMaterial color={color} depthTest={false} transparent opacity={0.85} />
  </lineSegments>;
}
function FeatureHighlight({ feature, color }: { feature: MeshFeature; color: string }) {
  const geometry = useMemo(() => {
    const coords = feature.triangles.flatMap(t => [...t.a.toArray(), ...t.b.toArray(), ...t.c.toArray()]);
    return new BufferGeometry().setAttribute("position", new Float32BufferAttribute(coords, 3));
  }, [feature]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return feature.kind === "edge" ? <FeatureLines features={[feature]} color={color} /> :
    <mesh geometry={geometry} raycast={() => {}} renderOrder={5}>
      <meshBasicMaterial color={color} side={DoubleSide} depthTest={false} transparent opacity={0.45} />
    </mesh>;
}
