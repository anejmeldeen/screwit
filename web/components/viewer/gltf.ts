// Shared GLTFLoader configuration. Phone meshes (mesh_mobile_uri) are
// meshopt-compressed GLB, so three's MeshoptDecoder must be registered on the
// loader (AGENTS.md "Performance budgets"). Registering it is harmless for
// uncompressed desktop meshes too.
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";

export function configureGltfLoader(loader: GLTFLoader): void {
  loader.setMeshoptDecoder(MeshoptDecoder);
}
