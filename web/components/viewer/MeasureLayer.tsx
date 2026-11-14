"use client";

import { useRef } from "react";
import { ThreeEvent } from "@react-three/fiber";
import { Vector3, Matrix4 } from "three";

// Measure / manual-scale interaction. Click two points on the scene; the parent
// group's inverse matrix converts each world hit back to the SCAN frame (mm) so
// /measure and /scale receive scan-frame points as the API expects.
//
// This layer draws a large invisible pick target is NOT used; instead the
// parent passes the parent-group world matrix so we can invert hits. Markers
// are rendered in SCAN coordinates as children of the parent group.
export interface PickedPoint {
  scan: [number, number, number];
}

export function MeasureLayer({
  active,
  parentMatrix,
  points,
  onPick,
}: {
  active: boolean;
  parentMatrix: Matrix4;
  points: PickedPoint[];
  onPick: (p: PickedPoint) => void;
}) {
  const inv = useRef(new Matrix4());
  inv.current.copy(parentMatrix).invert();

  // A click anywhere on a child mesh bubbles here (parent group onClick) when
  // active; convert the world hit point to scan coordinates.
  function handleClick(e: ThreeEvent<MouseEvent>) {
    if (!active) return;
    e.stopPropagation();
    const world = e.point.clone();
    const scan = world.applyMatrix4(inv.current);
    onPick({ scan: [scan.x, scan.y, scan.z] });
  }

  return (
    <group onClick={handleClick}>
      {points.map((p, i) => (
        <mesh key={i} position={new Vector3(p.scan[0], p.scan[1], p.scan[2])}>
          {/* radius in scan mm; small sphere */}
          <sphereGeometry args={[3, 16, 16]} />
          <meshBasicMaterial color={i === 0 ? "#34d399" : "#f472b6"} />
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
  // Thin box as a line that is visible through the splat.
  const quat = new Vector3(0, 1, 0);
  void quat;
  return (
    <group position={mid}>
      <mesh
        // Orient a unit box along the segment.
        onUpdate={(self) => {
          self.lookAt(end);
        }}
      >
        <boxGeometry args={[1.5, 1.5, len]} />
        <meshBasicMaterial color="#fbbf24" />
      </mesh>
    </group>
  );
}
