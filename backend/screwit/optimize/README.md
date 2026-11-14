# X5 phone asset optimizer

Produces the phone-sized variants of a scan's full-size assets, as a library P2's pipeline calls
(SPEC 6.1 steps 4 and 6, AGENTS.md "Performance budgets"). Everything stays in **scan-frame
millimetres** with no axis or unit change (SPEC 3.1).

```python
from screwit.optimize import make_mobile_variants
from screwit.recon import gaussians            # P2's gaussian IO (integration)

variants = make_mobile_variants(
    full_mesh_path="scans/<id>/mesh/object.glb",      # full GLB/PLY, <=200k tris
    full_splat_path="scans/<id>/splat/scene.ply",     # full 3DGS PLY, or None
    out_dir="scans/<id>/mobile",
    gaussian_io=gaussians,                            # needed only when a splat is given
)
# -> out_dir/object.glb (phone mesh) and out_dir/scene.spz (phone splat)
# variants.mesh_stats / variants.splat_stats carry counts and byte sizes.
```

These feed the optional `mesh_mobile_uri` and `splat_mobile_uri` on the Reconstruction contract
(SPEC 4.2); the client picks the full or mobile asset by device capability and offers a toggle.

## Mesh: decimate then meshopt-compress

- **Decimator: Open3D `simplify_quadric_decimation`** (Open3D pinned at **0.19.0**, already a project
  dependency). Chosen over pyfqmr because it is already installed (no new dependency), it **preserves
  per-vertex colour** through decimation, and it is fast enough: a 327k-triangle coloured sphere
  decimates to 50k in ~2.2 s with colours kept and the bounding box within 0.005 mm. Target is at
  most **50k triangles** (`MAX_MOBILE_TRIANGLES`). A mesh already under the cap is passed through
  untouched. The optimizer re-measures the bounding box before and after and refuses to write a
  mesh whose box shifted more than 0.5 mm.
- **Compressor: meshopt via `gltfpack`**, the official meshoptimizer binary. We install the pinned
  npm **`gltfpack` 1.3.0** package (`optimize/toolchain/package.json` + `package-lock.json`), which ships
  meshoptimizer's own cross-platform WASM build of gltfpack (`library.wasm`), so there is no
  per-platform native binary to fetch. The version and a sha512 subresource-integrity hash are
  pinned in `optimize/toolchain/package-lock.json`. Install once with `npm install` in
  `screwit/optimize/toolchain`; override the binary location with `SCREWIT_GLTFPACK` if needed.
- gltfpack is run with **`-cc -noq`**: `-cc` writes the `EXT_meshopt_compression` extension, and
  `-noq` turns **off** `KHR_mesh_quantization`. Quantization is off on purpose: it would snap vertex
  positions to a 14-bit grid and bake a dequantize transform into the node, which (a) perturbs the
  scan-frame millimetre coordinates and (b) is silently ignored by loaders that don't honour the
  transform, both of which break the "coordinates unchanged" rule. `-noq` keeps meshopt's lossless
  vertex/index encoding only, so the decompressed geometry is bit-for-bit the decimated mesh. The
  trade-off is a slightly larger file than quantized meshopt; the geometry is small (<=50k tris) so
  this is a few tens of KB, well inside budget.

### three.js loader requirement for P1

The phone GLB uses `EXT_meshopt_compression`, so **`GLTFLoader` cannot read it without
`MeshoptDecoder`**. P1 must wire the decoder:

```js
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

const loader = new GLTFLoader();
loader.setMeshoptDecoder(MeshoptDecoder);   // required for the mobile mesh; harmless for the full mesh
```

`meshopt_decoder.module.js` ships with three.js (`three/examples/jsm/libs/`). Without
`setMeshoptDecoder`, loading the mobile GLB throws
`THREE.GLTFLoader: No DRACOLoader instance provided` / `KHR/EXT ... not supported`. The desktop
(full-size) GLB is uncompressed and loads with or without the decoder, so calling
`setMeshoptDecoder` unconditionally is safe.

## Splat: select by importance, write .spz

- Keeps at most **400k gaussians** (`MAX_MOBILE_GAUSSIANS`) of the input, chosen by
  **importance = `sigmoid(opacity) * volume`** (volume from the three log scales) with **spatial
  stratification**: the means are binned into a grid (~2 gaussians per cell) and the highest-scoring
  member of each cell is taken round-robin, so coverage stays even instead of collapsing onto the
  densest patch. Positions, rotations, scales, opacities and colours of the kept gaussians are
  written through unchanged (no re-fitting, no rescale).
- Output is `.spz` **v4** (`from = UNSPECIFIED`), readable by Spark **>= 2.4.0** (SPEC 3.1 table).
- **No cross-worktree import.** P2 owns the real SPZ writer and 3DGS PLY reader
  (`screwit.recon.gaussians`). This module talks to them through the `GaussianIO` Protocol and ships
  a minimal local PLY reader / SPZ writer for its own tests. `MobileGaussians` mirrors P2's
  `GaussianCloud` field-for-field (means, log_scales, `quats_wxyz` = `[w, x, y, z]` PLY order,
  opacities = pre-sigmoid logits, `f_dc` = SH degree-0), so **integration wires P2's
  `read_ply`/`write_spz` in** as the `GaussianIO` with no data reshaping.

### Quaternion orders

- 3DGS PLY stores rotations `[w, x, y, z]` (`rot_0` is w); kept internally as `quats_wxyz`.
- The spz `GaussianCloud` and the section-4 contracts use `[x, y, z, w]`; the reorder
  (`wxyz -> xyzw`, columns `[1, 2, 3, 0]`) happens only at the spz packing boundary.
