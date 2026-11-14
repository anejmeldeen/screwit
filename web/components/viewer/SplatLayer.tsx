"use client";

import { useEffect, useState } from "react";
import { useThree } from "@react-three/fiber";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { markOnce, PERF_MARKS } from "@/lib/perf";

// Spark integration (SPEC 3.1 / P1 item 4): exactly ONE SparkRenderer, created
// with the r3f `gl` (WebGLRenderer) and added to the scene (without it nothing
// renders), disposed on unmount. A SplatMesh loads the .spz under the parent
// group. The .spz is Spark version 4, which Spark 2.4.0 reads.
export function SplatLayer({
  url,
  visible,
}: {
  url: string;
  visible: boolean;
}) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);
  const [splat, setSplat] = useState<SplatMesh | null>(null);

  // One SparkRenderer for the whole scene.
  useEffect(() => {
    const spark = new SparkRenderer({ renderer: gl, onDirty: () => invalidate() });
    scene.add(spark);
    return () => {
      scene.remove(spark);
      // SparkRenderer is a THREE.Mesh; free its GPU resources.
      spark.dispose?.();
    };
  }, [gl, scene, invalidate]);

  // One SplatMesh per splat URL, re-created when the URL changes.
  useEffect(() => {
    let disposed = false;
    const mesh = new SplatMesh({
      url,
      onLoad: () => {
        markOnce(PERF_MARKS.firstSplat);
        invalidate();
      },
    });
    if (!disposed) setSplat(mesh);
    return () => {
      disposed = true;
      setSplat(null);
      mesh.dispose?.();
    };
  }, [url, invalidate]);

  if (!splat) return null;
  // The SplatMesh is an Object3D; render it as a primitive child of the parent
  // group so the single scan_to_web transform applies to it.
  return <primitive object={splat} visible={visible} />;
}
