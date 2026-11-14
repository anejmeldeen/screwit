"use client";

import { useMemo } from "react";
import { useLoader } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Quaternion } from "three";
import type { CadResult } from "@/lib/contracts.gen";
import { assetUrl } from "@/lib/storage";

// Overlay a generated CAD STL at its placement (translate_mm + rotation_quat
// [x,y,z,w]) in a distinct colour under the parent group. SPEC 2.3 / P1 item 4.
export function CadOverlay({
  cad,
  placement,
  visible,
}: {
  cad: CadResult;
  placement: { translate_mm: number[]; rotation_quat: number[] };
  visible: boolean;
}) {
  const url = cad.files?.stl ? assetUrl(cad.files.stl) : null;
  if (!url) return null;
  return (
    <CadStl url={url} placement={placement} visible={visible} />
  );
}

function CadStl({
  url,
  placement,
  visible,
}: {
  url: string;
  placement: { translate_mm: number[]; rotation_quat: number[] };
  visible: boolean;
}) {
  const geom = useLoader(STLLoader, url);
  const t = placement.translate_mm;
  const q = placement.rotation_quat; // [x, y, z, w]
  const quat = useMemo(
    () => new Quaternion(q[0], q[1], q[2], q[3]),
    [q],
  );
  return (
    <mesh
      geometry={geom}
      visible={visible}
      position={[t[0], t[1], t[2]]}
      quaternion={quat}
    >
      <meshStandardMaterial
        color="#f97316"
        transparent
        opacity={0.7}
        emissive="#7c2d12"
        emissiveIntensity={0.3}
      />
    </mesh>
  );
}
