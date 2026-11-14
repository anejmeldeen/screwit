"use client";

import { Vector3, Quaternion } from "three";

// Markers and measurement segments live in scan-frame millimetres. Surface
// raycasts are handled by FeatureSurface, directly on the rendered meshes.
export interface PickedPoint {
  scan: [number, number, number];
}

export function MeasureLayer({ points }: { points: PickedPoint[] }) {
  return (
    <group>
      {points.map((p, i) => (
        <mesh raycast={() => {}} key={i} position={new Vector3(p.scan[0], p.scan[1], p.scan[2])}>
          {/* radius in scan mm; small sphere */}
          <sphereGeometry args={[3, 16, 16]} />
          <meshBasicMaterial color={i === 0 ? "#34d399" : "#f472b6"} depthTest={false} />
        </mesh>
      ))}
      {points.length === 2 ? (
        <MeasureSegment a={points[0].scan} b={points[1].scan} />
      ) : null}
    </group>
  );
}

function MeasureSegment({
  a,
  b,
}: {
  a: [number, number, number];
  b: [number, number, number];
}) {
  const start = new Vector3(...a);
  const end = new Vector3(...b);
  const mid = start.clone().add(end).multiplyScalar(0.5);
  const dir = end.clone().sub(start);
  const len = dir.length();
  const rotation = new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), dir.clone().normalize());
  if (len < 1e-9) return null;
  return (
    <mesh position={mid} quaternion={rotation} raycast={() => {}} renderOrder={10}>
      <boxGeometry args={[1.5, 1.5, len]} />
      <meshBasicMaterial color="#fbbf24" depthTest={false} />
    </mesh>
  );
}
