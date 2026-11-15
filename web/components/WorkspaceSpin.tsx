"use client";

import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { Box3, Group, Mesh, MeshStandardMaterial, Sphere, Vector3 } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { api } from "@/lib/api";
import { partColor } from "@/lib/palette";
import { assetUrl } from "@/lib/storage";
import { configureGltfLoader } from "./viewer/gltf";

type Asset = { url: string; color?: string };

function Orbit({ assets, ready }: { assets: Asset[]; ready: () => void }) {
  const models = useLoader(GLTFLoader, assets.map(asset => asset.url), configureGltfLoader);
  const camera = useThree(state => state.camera);
  const angle = useRef(-Math.PI / 4);
  const assembly = useMemo(() => {
    const group = new Group();
    const materials: MeshStandardMaterial[] = [];
    models.forEach((model, index) => {
      const scene = model.scene.clone(true);
      if (assets[index].color) {
        const material = new MeshStandardMaterial({ color: assets[index].color, roughness: 1 });
        materials.push(material);
        scene.traverse(node => { if (node instanceof Mesh) node.material = material; });
      }
      group.add(scene);
    });
    const sphere = new Box3().setFromObject(group).getBoundingSphere(new Sphere());
    group.position.copy(sphere.center).multiplyScalar(-1);
    group.scale.setScalar(1 / (sphere.radius || 1));
    group.position.multiplyScalar(group.scale.x);
    return { group, materials };
  }, [models, assets]);
  useEffect(() => {
    ready();
    return () => assembly.materials.forEach(material => material.dispose());
  }, [assembly, ready]);
  useFrame((_, delta) => {
    angle.current += Math.min(delta, 0.05) * 0.85;
    camera.up.set(0, 0, 1);
    camera.position.set(3 * Math.cos(angle.current), 3 * Math.sin(angle.current), 1.8);
    camera.lookAt(0, 0, 0);
  });
  return <primitive object={assembly.group} />;
}

class PreviewBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? null : this.props.children; }
}

export default function WorkspaceSpin({ id }: { id: string }) {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let active = true;
    async function load() {
      const parts = await api.getParts(id);
      const items = parts.length ? parts.map((part, index) => ({
        url: assetUrl(part.mesh_uri), color: partColor(index, part.part_id),
      })) : [{ url: assetUrl((await api.getReconstruction(id)).mesh_uri) }];
      if (active) setAssets(items);
    }
    load().catch(() => {});
    return () => { active = false; };
  }, [id]);
  const show = useMemo(() => () => setReady(true), []);
  if (!assets.length) return null;
  return <PreviewBoundary><div className="workspace-spin" aria-hidden="true" style={{ opacity: ready ? 1 : 0 }}>
    <Canvas dpr={[1, 1.5]} camera={{ fov: 40, position: new Vector3(2, -2, 1.8) }}>
      <color attach="background" args={["#181918"]} />
      <ambientLight intensity={1.6} /><directionalLight position={[3, -2, 5]} intensity={2} />
      <Suspense fallback={null}><Orbit assets={assets} ready={show} /></Suspense>
    </Canvas>
  </div></PreviewBoundary>;
}
