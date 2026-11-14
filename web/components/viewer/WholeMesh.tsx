"use client";

import { useEffect, useMemo } from "react";
import { Mesh, MeshStandardMaterial } from "three";
import { useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { configureGltfLoader } from "./gltf";
import { markOnce, PERF_MARKS } from "@/lib/perf";

// A single whole-object GLB (reconstruction.mesh_mobile_uri), used on phones
// instead of the heavy per-part GLBs. It is tiny (meshopt-compressed, a few
// thousand triangles) so the "mesh visible under 5 s" budget is met on slow 4G.
// No picking/explode here — that is the desktop per-part path (PartMeshes).
export function WholeMesh({ url, visible }: { url: string; visible: boolean }) {
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    configureGltfLoader(loader as GLTFLoader);
  });
  const scene = useMemo(() => {
    const clone = gltf.scene.clone(true);
    clone.traverse((object) => {
      if (!(object instanceof Mesh)) return;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      const display = materials.map((source) => {
        const material = source.clone();
        if (material instanceof MeshStandardMaterial) {
          material.metalness = 0;
          material.roughness = 1;
        }
        return material;
      });
      object.material = Array.isArray(object.material) ? display : display[0];
    });
    return clone;
  }, [gltf]);
  useEffect(() => () => {
    scene.traverse((object) => {
      if (!(object instanceof Mesh)) return;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => material.dispose());
    });
  }, [scene]);

  // First whole-object mesh drawn marks the viewer's first-mesh timestamp.
  useEffect(() => {
    markOnce(PERF_MARKS.firstMesh);
  }, [scene]);

  return <primitive object={scene} visible={visible} />;
}
