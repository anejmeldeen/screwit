import { AmbientLight, Box3, Color, DirectionalLight, Group, Mesh, MeshStandardMaterial,
  OrthographicCamera, Scene, Vector3, WebGLRenderer, type Material, type Texture } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { configureGltfLoader } from "@/components/viewer/gltf";
import { assetUrl } from "@/lib/storage";
import type { Part } from "@/lib/contracts.gen";

// One off-screen canvas produces printable stills. The editor keeps its camera and selection.
export async function renderManualViews(parts: Part[], selections: string[][], signal: AbortSignal) {
  const images = new Map<string, string>();
  if (!parts.length || signal.aborted) return images;
  const loader = new GLTFLoader();
  configureGltfLoader(loader);
  const renderer = new WebGLRenderer({ antialias: true });
  const scene = new Scene();
  const assembly = new Group();
  scene.add(assembly);
  scene.background = new Color("#f4f3ed");
  const active = new MeshStandardMaterial({ color: "#d69a19", roughness: 1 });
  const normal = new MeshStandardMaterial({ color: "#b6b7ac", roughness: 1 });
  const ghost = new MeshStandardMaterial({ color: "#9b9d97", transparent: true, opacity: .18, depthWrite: false });
  const originals = new Set<Material>();
  try {
    renderer.setSize(640, 420, false);
    // Wait for every load to settle before freeing any GPU resources.
    const loaded = await Promise.allSettled(parts.map(async part => {
      const gltf = await loader.loadAsync(assetUrl(part.mesh_uri));
      gltf.scene.name = part.part_id;
      assembly.add(gltf.scene);
      gltf.scene.traverse(o => {
        if (o instanceof Mesh) {
          for (const m of Array.isArray(o.material) ? o.material : [o.material]) originals.add(m);
          o.material = normal;
        }
      });
    }));
    if (signal.aborted || loaded.some(r => r.status === "rejected")) return images;
    const box = new Box3().setFromObject(assembly);
    const radius = box.getSize(new Vector3()).length() * .54;
    if (!Number.isFinite(radius) || radius <= 0) return images;
    const center = box.getCenter(new Vector3());
    const aspect = 640 / 420;
    const camera = new OrthographicCamera(-radius*aspect, radius*aspect, radius, -radius, radius*.01, radius*10);
    camera.up.set(0, 0, 1);
    camera.position.copy(center).addScaledVector(new Vector3(1,-1.7,.85).normalize(), radius*3);
    camera.lookAt(center);
    scene.add(new AmbientLight(0xffffff, 1.6));
    const light = new DirectionalLight(0xffffff, 2.3);
    light.position.copy(camera.position);
    light.target.position.copy(center);
    scene.add(light, light.target);
    for (const ids of selections) {
      const key = JSON.stringify(ids);
      if (images.has(key)) continue;
      for (const part of assembly.children) {
        const material = !ids.length ? normal : ids.includes(part.name) ? active : ghost;
        part.traverse(o => { if (o instanceof Mesh) o.material = material; });
      }
      renderer.render(scene, camera);
      images.set(key, renderer.domElement.toDataURL("image/png"));
    }
    return images;
  } finally {
    assembly.traverse(o => { if (o instanceof Mesh) o.geometry.dispose(); });
    for (const material of originals) {
      for (const value of Object.values(material)) if ((value as Texture | null)?.isTexture) value.dispose();
      material.dispose();
    }
    for (const m of [active, normal, ghost]) m.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  }
}
