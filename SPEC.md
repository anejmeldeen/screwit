# Screw It: Product and Technical Specification

Screw It is a website where a user films or photographs a physical object, gets an interactive 3D model of it, and asks for help in plain language. The AI explains how the object is built and works, diagnoses problems ("my sink is leaking"), proposes a fix with step-by-step instructions and a parts list, and when a part can be printed, generates a 3D-printable CAD model that fits the original.

---

## 1. Goals and scope

### 1.1 Hackathon goals
1. A user can upload a phone video (or 20-60 photos) of an object from the website.
2. The site shows an interactive 3D view of the object (a mesh, plus a Gaussian splat when the splat model's licence is cleared; see 1.5).
3. The object is broken into labelled parts with real-world dimensions in millimetres.
4. The user chats with an agent that can see the 3D model and parts, and that answers "how does this work?" and "how do I fix X?".
5. The agent returns a repair plan: diagnosis, numbered steps, tools, and parts to buy with links.
6. For a printable modification, the agent generates a parametric CAD part, checks that it is printable and fits, shows it in the 3D view next to the original, and offers STL/STEP download.

### 1.2 Out of scope
- User accounts, payments, multi-user collaboration.
- Real finite-element analysis (use heuristic checks instead).
- General-purpose support for any object. We tune for the demo objects in section 9.
- Native mobile apps. Mobile browser capture is enough.

### 1.3 Demo objects ("hero" scenarios)
- **Scenario A (repair and buy):** a sink P-trap or faucet that leaks. Result: diagnosis (worn slip-nut washer or loose nut), steps, and a parts list with the correct size (for example a 1-1/2" slip-joint washer for a kitchen trap; a bathroom lavatory often has a 1-1/4" tailpiece into a 1-1/2" trap and then needs a 1-1/2" x 1-1/4" reducing washer, so the catalogue carries both and the fixtures use the real hero trap's caliper values).
- **Scenario B (modify and print):** a broken or missing part, such as a cabinet knob, a shelf bracket, or a hose adapter. Result: a printable replacement sized from the scan where the scan can resolve the dimension, and from a caliper value the user enters where it cannot (at the geometry model's resolution a 6 mm D-shaft spans only 6-12 pixels and its 1.5 mm flat one to three, and the depth noise is larger than the fit tolerance, so the agent asks for those numbers rather than guessing; see 6.3).

Practise both on the exact physical objects you will bring to judging.

### 1.4 Judged MVP versus stretch
The judged demo runs on the **hero scans processed before judging**, with every automatically produced result checked by a human first: part labels and dimensions compared against calipers (and corrected through the parts API where the scan was off), the scene graph read through, the parts catalogue curated, the CAD part from the template library, and the replacement part already printed. A live scan during the demo is a bonus that shows the pipeline running, not the thing the claims rest on. **Every "done when" criterion in section 7 is judged against these cached hero artifacts**; live-path speed and never-seen-object behaviour are stretch.

| Judged MVP (must work) | Stretch (only after the MVP is proven end to end) |
|---|---|
| Cached hero scans in the viewer: mesh, and the splat if the splat model's licence is cleared (1.5); the stage-progress stream itself (it is how the team runs and checks every scan before judging) | Live scan of an object the team has never processed; showing a live scan's progress during the judged demo; the live path finishing inside a time budget |
| Parts, dimensions (scan value, snapped value, caliper value where entered) and scene graph for the hero objects | Part split editing, arbitrary objects, GLB import from phone scanning apps |
| Chat: explain, diagnose, repair plan, parts to buy from the curated catalogue; the agent asks the user to confirm any dimension the scan cannot resolve before printing | Live product search |
| Template CAD with primitive-based fit check, printability checks, STL/STEP download, preprinted part | LLM-written CAD (codegen), interference volume from mesh booleans, slicer-accurate time estimates |
| Strength estimate shown as advisory text | Anything in section 11 |

### 1.5 No training, and the licence gate
Nothing in this system is trained or fine-tuned by the team, and there is **no per-scene optimization**. Reconstruction uses a pre-trained feed-forward geometry model that produces camera poses, depth maps and 3D Gaussians in one pass (section 6.1). Classic Gaussian-splat training (COLMAP plus 10-30 minutes of optimization per scene) is deliberately excluded.

**Licence gate (day zero):** the Depth Anything 3 checkpoints with a Gaussian head are the Giant and Nested families (`DA3-GIANT-1.1`, `DA3NESTED-GIANT-LARGE-1.1`, and their deprecated unsuffixed originals), all released under CC BY-NC 4.0; this spec selects `DA3NESTED-GIANT-LARGE-1.1`. Confirm with the hackathon organisers whether that licence is acceptable for this event before building on it. If it is not, the geometry path is MapAnything's Apache-2.0 checkpoint (`facebook/map-anything-apache`), the display layer is the vertex-coloured mesh, and the splat is dropped from the MVP table (AnySplat's weights also need a licence check before use). Record the approved checkpoint in the README.

---

## 2. User experience (website)

### 2.1 Pages
| Route | Purpose |
|---|---|
| `/` | Landing page: one-line pitch, "Start a scan" button, demo gallery of pre-processed scans. |
| `/scan/new` | Capture: record video in the browser (phone camera) or drag-and-drop a video/photos. Shows capture tips and the scale-board instructions. |
| `/scan/[id]` | Main workspace: 3D viewer (left/centre), parts panel, chat panel (right), processing status bar. |
| `/scan/[id]/part/[partId]` | Optional deep link that focuses the viewer on one part. |

### 2.2 Capture flow
1. User prints the **scale board** at 100% scale (a ChArUco board, A4 PDF linked on the page, at least 16 corners, with a printed arrow marking its X axis) and checks one printed square with a ruler before use; a board shown on a screen has an unknown physical size and is not accepted. **The board lies flat on the same horizontal surface the object stands on**, next to the object, with the arrow pointing away from the camera's first position; the pipeline checks this (6.1 step 3). Fallback when there is no board: a credit card (85.60 x 53.98 mm) in frame, which the user later clicks on in the viewer through the manual scale step (6.1 step 3).
2. User records a slow orbit video, or uploads photos. Accepted duration is 10-90 s; the recommended range is 20-60 s and the page says so. On phones the page offers `<input type="file" accept="video/*" capture="environment">`; on most phones this opens the native camera, but `capture` is only a hint, so the ordinary file picker is always offered too. The browser and device decide the container and codec: iPhones usually deliver MOV with HEVC (10-bit HDR by default on iPhone 12 and later), Android usually MP4 with H.264, and a desktop `MediaRecorder` gives WebM or MP4. The server probes and normalises every file (6.1 step 1); nothing is assumed from the browser. Tell iPhone users to turn HDR video off in Camera settings for cleaner frames.
3. Client pre-checks: duration (10-90 s), file size (limit 500 MB), and warnings for blur or darkness when they can be detected cheaply. These are a courtesy; the server re-validates everything.
4. Upload goes straight to object storage with a pre-signed POST (3.3). The page then moves to `/scan/[id]`.

### 2.3 Workspace (`/scan/[id]`)
- **Status bar** showing each pipeline stage in order: Frames, Geometry (poses and depth), Scale, Object mask, Splat, Mesh, Parts. Progress is pushed live over Server-Sent Events (SSE). Geometry itself takes seconds; the whole reconstruction half (through Mesh) a minute or two; Parts a few minutes more.
- **3D viewer**
  - Toggle between Splat view (photoreal) and Mesh view (clean, clickable).
  - Hover highlights a part. Click selects it and shows its label, dimensions, and material guess.
  - "Exploded view" slider that pushes parts apart along their centroid directions.
  - Measurement tool: click two points, get a distance in mm.
  - Overlay of generated CAD parts (in a distinct colour) at their mating position, with a fit/clearance indicator.
- **Parts panel**: list of detected parts with label, confidence, and dimensions showing the scan value, the snapped standard size, and a caliper value if one was entered. Users can rename a part, merge parts, and enter a caliper-measured dimension; split is stretch.
- **Scale step (when no board was found)**: the viewer asks the user to click the two ends of a known length (the credit card's long edge, 85.60 mm) and calls `/scale`.
- **Chat panel**
  - Streaming answers.
  - The agent can reference parts as chips (`[[part:p3]]`) that highlight in the viewer when clicked.
  - Rich cards inside the chat: Repair Plan card, Parts-to-Buy card, CAD Result card (thumbnail, checks, download buttons, "show in viewer").
  - User can attach extra photos mid-chat ("here's a close-up of the leak").
- **Early answers before parts are ready**: as soon as frames are extracted, the agent can already chat using keyframes (vision model on images). It upgrades its answers once the mesh and parts exist. Perception is the slow stage (a few minutes), so this keeps the demo responsive.

---

## 3. System architecture

```
Browser (Next.js)
  |  REST + SSE (all over HTTPS, bearer token)
  v
API server (FastAPI, Python)  ---- object storage (S3 or local disk)
  |  enqueue jobs                    scans/{scan_id}/...
  v
Job queue (Redis + RQ)
  |
  +--> ONE GPU worker, concurrency 1, runs the stages in order:
  |      frames -> geometry (feed-forward model) -> scale -> object mask (SAM 2) -> splat + TSDF mesh -> parts
  |      (reconstruction code is P2's, the object-mask and parts stages call P3's functions; between
  |       geometry and the object mask the worker writes the gaussians and depth to disk, drops every
  |       GPU tensor and the model, and asserts VRAM is back under a threshold before loading SAM 2)
  +--> CPU worker: CAD (template library, fit and printability checks, STL/STEP export)
  |
  +--> Agent service (LLM with tools; called by the chat endpoint)
          tools -> scene graph, parts, measurement, CAD jobs, parts search, renders
```

### 3.1 Tech choices (defaults; swap if a teammate knows something better)
| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind | Fast to build, easy deploy. |
| 3D viewer | three.js via react-three-fiber; splats rendered with **Spark ≥ 2.4.0** (pinned): one `SparkRenderer` attached to the scene (without it nothing renders) plus a `SplatMesh` per splat (a normal `Object3D`; loads `.ply`, `.spz`, `.splat`, `.ksplat`, `.sog`). 2.4.0 is the first release that reads `.spz` **version 4**, which is what the spz packer writes by default; older Spark releases read only versions 1-3, so the Spark version and the spz `PackOptions.version` are pinned together in the golden fixture | Splats and meshes in one scene; Spark publishes an official react-three-fiber example. GaussianSplats3D is the fallback; its viewer owns the scene, which fits react-three-fiber less well. |
| API | FastAPI + Pydantic | Python, same language as the ML and CAD code; Pydantic models are the data contracts. |
| Jobs | Redis + RQ, GPU worker started with `--worker-class rq.SimpleWorker` (no fork, so CUDA works; note it sends no heartbeat during a job, so set job timeouts generously) | One GPU queue, one CPU queue. Forking workers break CUDA ("Cannot re-initialize CUDA in forked subprocess"). |
| Storage | S3 bucket (private; pre-signed POST for uploads with size and type limits, signed GET for assets; CORS rule for the web origin) or local `./data` in dev | Large files (videos, splats, meshes). The browser talks to S3 directly, so the bucket CORS rule must allow the methods and headers used. |
| Database | Postgres in Docker, bound to `127.0.0.1` only (no `-p 0.0.0.0` publishing, which bypasses the host firewall), via SQLModel | Scan, part and chat records. Several worker processes writing SQLite would hit lock errors. Redis is bound the same way. |
| Reconstruction | ffmpeg; **Depth Anything 3** (`DA3NESTED-GIANT-LARGE-1.1`: poses, depth, metric scale, and a Gaussian head that exports a splat; licence gate in 1.5) with **MapAnything** (`facebook/map-anything-apache`) as the Apache-licensed geometry alternative (no splat); Open3D `ScalableTSDFVolume` for the mesh; trimesh, pymeshfix | Feed-forward, no COLMAP and no per-scene optimization. One model call gives poses, depth and gaussians. |
| Perception | SAM 2 (video propagation and automatic masks), Grounded SAM 2 (Grounding DINO text prompts to boxes to SAM 2 masks), Claude for labels, Open3D + pyransac3d for geometry fitting | Text-promptable, consistent part masks across frames. |
| Agent | Claude with tool use, streamed | Reasoning, tool calls, vision. |
| CAD | CadQuery or build123d, trimesh (`proximity.thickness`) for checks | Code-based CAD with a fixed template library. |
| GPU host | One cloud GPU instance with 24 GB VRAM (AWS g5 A10G or g6 L4) and a view cap set by measurement (6.1 step 2); a 48 GB L40S (g6e) buys headroom if the budget allows | Feed-forward reconstruction plus SAM 2, one job at a time. |

Check library versions on day one. 3D tooling changes quickly.

**Two frames, one transform.** All computation (reconstruction, perception, measurement, CAD) happens in the **scan frame**: right-handed, **millimetres**, Z up. **Rotation always comes from the support plane:** the dominant plane under the object (Open3D `segment_plane` on the geometry model's point maps) defines Z, because a board can be placed at any angle and its normal is not gravity. With a scale board, the board supplies the uniform scale, the origin (the board's origin corner) and the in-plane X direction (its printed arrow); the scale stage checks that the board normal and the support plane normal agree within 10 degrees and otherwise flags the scan ("lay the board flat next to the object") and falls back to the no-board origin rule. Without a board, the scale stage sets rotation and scale only; the origin, `object_bottom_center` (the centre of the object's bounding box projected onto the support plane), is fixed by the object-mask stage, which runs before any downstream artifact exists, and that translation is applied to cameras, depth and gaussians before the splat and mesh stages write anything. Nothing downstream is ever published in a frame that later changes. The feed-forward model emits poses, depth and gaussians in one arbitrary frame; the scale step computes a single similarity transform (rotation, translation, uniform scale) from that frame to the scan frame and applies it once to the cameras, the depth maps or point maps, and the gaussians.

- **Camera contract.** `geometry/cameras.json` stores, per keyframe, `world_to_camera` as a 4x4 row-major matrix in the scan frame with OpenCV axes (X right, Y down, Z forward), plus `K` for the exact image grid the model used (6.1 step 1). Depth Anything 3 returns world-to-camera already; MapAnything returns camera-to-world, and its adapter inverts. Every consumer (triangulation, TSDF, mask lifting) builds `P = K [R | t]` from this one field.
- **Transforming gaussians.** A 3DGS `.ply` stores means, **log** scales, unit quaternions in **`[w, x, y, z]`** order (`rot_0` is w), inverse-sigmoid opacities and SH coefficients. The spz library's `GaussianCloud.rotations` and the contracts in section 4 are **`[x, y, z, w]`**, so reorder when reading the PLY and again when filling the spz cloud; a wrong order scrambles every splat's orientation while leaving positions correct, so the golden fixture includes one elongated gaussian with a known rotation. Applying the similarity `(R, t, s)`: `mean' = s R mean + t`, `quat' = quat(R) * quat`, `log_scale' = log_scale + ln(s)`, opacity unchanged. Colour is kept at SH degree 0, which is rotation-invariant, so no SH rotation is needed (DA3 exports DC-only). With the DA3 **nested** model there is one extra step first: its metric alignment multiplies the depth and the extrinsic translations by `prediction.scale_factor` but leaves the gaussians unscaled (its own video exporter divides the translations back out to compensate), so multiply the gaussian means by `scale_factor` and add `ln(scale_factor)` to the log scales before applying the scan-frame similarity. Skipping this leaves the splat offset from the mesh by exactly that factor.
- **Web frame.** three.js is Y up, and the viewer's units are **metres**. The viewer puts the splat, the part meshes, the CAD STL and the measurement points under one parent group and applies exactly one transform: `scan_to_web.matrix` (the rotation taking Z up to Y up, `Rx(-90°)`, determinant +1, stored row-major) and `scan_to_web.unit_scale = 0.001` (millimetres to metres). In react-three-fiber: `<group matrix={m} matrixAutoUpdate={false}>` where `m = new Matrix4().set(...16 row-major values...).multiply(new Matrix4().makeScale(0.001, 0.001, 0.001))`. Files carry no reliable unit or axis metadata (glTF assumes Y up and metres, STL has none), so every asset is written in the scan frame and this parent transform is the only conversion.
- **SPZ packing, no conversion.** The spz library's `PackOptions.from` names the coordinate system of the input data and converts it into the file's RUB (Y up) convention; declaring the scan frame's real convention (`RFU`) would therefore bake `Rx(-90°)` into the file, and the parent group would then rotate the splat a second time. Pack with `from = UNSPECIFIED` so the numeric scan-frame coordinates are stored unchanged and Spark renders them as they are; the parent group remains the only rotation. (`UnpackOptions` is a C++-library concept; Spark's `SplatMesh` has no such parameter.) The packer's default `PackOptions.version` is 4, which Spark reads from 2.4.0 onward (3.1 table); the exact pack call, the spz version and the Spark version are part of the golden fixture. Verify on day one with an asymmetric test object that nothing is mirrored and that a 100 mm object measures 100 mm in the viewer; keep that as a golden fixture test.

### 3.2 Storage layout
```
scans/{scan_id}/
  input/            original video, photos, or an imported .glb
  frames/           full-resolution keyframes (JPEG) for ArUco and the agent; model/ (the model-grid copies, e.g. 504x280); frames.json (index, sharpness, grid mapping)
  geometry/         cameras.json (world_to_camera 4x4 + K per keyframe, scan frame), depth/{frame}.npz (float32 mm), conf/{frame}.npz, points.ply
  masks/            object/{frame}.png (main-object mask, model grid)
  splat/            scene.ply, scene.spz (compressed for the web)
  mesh/             object.glb (decimated for the web), object_full.ply
  parts/            p1.glb, p2.glb, ... ; masks/{frame}_{part}.png
  scene_graph.json
  cad/{cad_id}/     model.py, model.step, model.stl, preview.png, checks.json
  renders/          part thumbnails used by the agent and UI
```

### 3.3 Security baseline
- Storage bucket is private. The browser only gets short-lived pre-signed URLs.
- LLM and parts-search API keys live in environment variables on the server, never in the frontend bundle.
- GPU workers are not reachable from the internet. Only the API server is public, and only during judging.
- The API must be served over HTTPS. The web app is on HTTPS, and browsers block HTTP API calls from an HTTPS page (mixed content); the camera API also needs HTTPS. Put Caddy (automatic Let's Encrypt) or a Cloudflare Tunnel in front of the API on the GPU host.
- **Access gate and limits.** CORS only governs browsers; anyone who finds the endpoint can call it directly, mint upload URLs, burn LLM credits and fill the single GPU queue. Every API route requires a demo token issued to the team and judges, rotated after the event. The demo link carries it in the URL **fragment** (`/#k=...`): fragments are never sent to any server, so they do not reach the web host's, proxy's or API's access logs. On first load the app reads the fragment, stores the token in `sessionStorage`, removes it from the address bar with `history.replaceState`, and sets `Referrer-Policy: no-referrer`. From then on it travels only as `Authorization: Bearer` on requests **to the API origin**, including both SSE streams, which are read with `fetch` plus `eventsource-parser` rather than the browser's `EventSource` (which cannot set headers). The web app therefore has two clients: the **API client**, which always adds the bearer header, and the **storage client**, which sends exactly the pre-signed POST form fields or the signed GET URL and never adds the demo token (S3 authenticates those requests through the form fields or the URL and would reject a foreign `Authorization` header; Spark and `GLTFLoader` fetch signed URLs directly and get no header either). The token is never a query parameter on any request and is never logged. Per-token and global limits: scans per hour, stored bytes, one running GPU job per token, chat calls per minute, CAD attempts per scan, and a hard cap on daily LLM spend.
- **Upload validation happens on the server, not in the browser.** Uploads use a pre-signed **POST** with a fixed key, allowed content types and `content-length-range` (so S3 itself enforces the 500 MB limit), expiring in 10 minutes. `/start` does only a `HEAD` on the object in the API process (existence, size, content type), then enqueues a short **validation job** on the worker that, inside the containment below, reads the magic bytes, runs `ffprobe` (duration 10-90 s, codec, dimensions, stream count) or, for photos, reads image headers and EXIF orientation, and only on success enqueues the pipeline. Passing the probe is a sanity check, not a security boundary: **every** ffmpeg and ffprobe invocation on uploaded media (probe, decode, tone-map, frame extraction) runs with the same containment: a current patched build, no network, a scratch directory as the only writable path, CPU, memory, pids and wall-clock limits, and bounded output dimensions and frame counts. Untrusted media is never parsed in the API process.
- S3 CORS for the exact web origin allows `POST`, `GET` and `HEAD` with the headers the uploader and the asset loaders (Spark, GLTFLoader) send.
- Redis and Postgres bind to `127.0.0.1` on the GPU host; Docker port publishing to `0.0.0.0` is not used because it bypasses the host firewall.
- **SSE keep-alive.** Both streams send an SSE comment line every 20 s; Cloudflare and most proxies close idle responses after about 100 s, and the parts stage takes minutes.
- **CAD code execution.** The judged path runs only the team's own template code, in a normal worker process. The stretch codegen mode runs LLM-written code in a disposable container: read-only image with Python and CadQuery, `--network none`, a `tmpfs` output directory as the only writable path, no scan store or secrets mounted, CPU/memory/pids limits, a 30 s timeout, and the container is deleted after the run. An AST import check is a convenience filter, not the security boundary.
- Validate request bodies at the API boundary with the Pydantic contracts.

---

## 4. Data contracts

All stages talk through these JSON shapes (Pydantic models in `backend/contracts/`, mirrored as TypeScript types in `web/lib/contracts.ts`). **Agree on these first, and commit sample files to `fixtures/` so every person can build against fake data from the start.**

All lengths are millimetres. All coordinates in these contracts are in the **scan frame** (section 3.1): right-handed, millimetres, Z up from the support plane, origin at the scale board or at `object_bottom_center`. The viewer converts to its own frame with the single `scan_to_web` transform; no contract field is ever in web units. Rotations are unit quaternions `[x, y, z, w]`, never Euler angles. All 4x4 matrices are row-major.

**Revisions.** Two monotonic counters, both carried into every CAD request and result: `reconstruction_revision` (scan-level; increments when `/scale` changes the geometry, which rescales the mesh and every part analytically, see section 5) and per-part `measurement_revision` (increments on every `PATCH /parts/{id}` that changes a Measurement or a preferred source, on a merge, and on every `/scale`; label and description edits do not increment it). A CAD result whose stored `(reconstruction_revision, measurement_revision)` no longer matches the current part is marked `stale` by `GET /cad/{id}` and shown as such; the agent re-runs it before offering a download.

### 4.1 Scan
```json
{
  "scan_id": "scn_8f2a",
  "created_at": "2026-11-14T10:00:00Z",
  "input": { "type": "video", "uri": "scans/scn_8f2a/input/video.mov", "duration_s": 34.2, "container": "mov", "codec": "hevc", "hdr": true },
  "status": "processing",
  "reconstruction_revision": 1,
  "stages": [
    { "name": "frames",      "state": "done",    "progress": 1.0, "message": "56 keyframes" },
    { "name": "geometry",    "state": "done",    "progress": 1.0 },
    { "name": "scale",       "state": "done",    "progress": 1.0, "message": "aruco, 41 views" },
    { "name": "object_mask", "state": "running", "progress": 0.6 },
    { "name": "splat",       "state": "pending" },
    { "name": "mesh",        "state": "pending" },
    { "name": "parts",       "state": "pending" }
  ]
}
```
`status` is only `uploaded`, `validating`, `processing`, `ready`, or `failed`; the `stages` array is the single source of truth for what exists. Stage names are exactly `frames`, `geometry`, `scale`, `object_mask`, `splat`, `mesh`, `parts`; stage states are `pending`, `running`, `done`, `failed`, `skipped`. The frontend enables the splat view when `splat` is `done`, measurement when `mesh` is `done`, and the parts panel when `parts` is `done`; a `skipped` stage is shown greyed out. `input.type` is `video`, `photos`, or `mesh` (a self-contained `.glb`, stretch; see 6.1 step 7), and for `mesh` the `frames`, `geometry`, `object_mask` and `splat` stages are `skipped`. `reconstruction_revision` increments whenever the scale changes (section 5, `/scale`).

### 4.2 Reconstruction result (`reconstruction.json`)
```json
{
  "scan_id": "scn_8f2a",
  "reconstruction_revision": 1,
  "splat_uri": "scans/scn_8f2a/splat/scene.spz",
  "mesh_uri": "scans/scn_8f2a/mesh/object.glb",
  "cameras_uri": "scans/scn_8f2a/geometry/cameras.json",
  "keyframes": ["frames/0001.jpg", "frames/0007.jpg"],
  "scale": { "method": "aruco", "board": "charuco_5x7_30mm", "corner_views": 41, "residual_mm": 0.6, "board_tilt_deg": 2.1, "learned_scale_ratio": 1.03 },
  "frame_origin": "board",
  "scan_to_web": { "matrix": [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], "unit_scale": 0.001 },
  "object_bbox_mm": { "min": [-120, -80, 0], "max": [130, 90, 310] },
  "quality": { "frames_used": 56, "total_frames": 56, "mean_depth_confidence": 0.81, "gaussians": 1200000 }
}
```
`splat_uri` is nullable (no Gaussian head, licence not cleared, or mesh import). `scale.method` is `aruco`, `manual` (two clicked points and a typed length; sets scale only, see section 5) or `learned` (the geometry model's own metric estimate, used only when no board was found; flagged in the UI as approximate). `frame_origin` is `board` or `object_bottom_center` (3.1); `board_tilt_deg` is the angle between the board normal and the support plane normal, and above 10 degrees the origin falls back to `object_bottom_center`. `learned_scale_ratio` records how far the model's metric estimate was from the board, as a sanity check. `scan_to_web.matrix` is row-major and equals `Rx(-90°)`.

### 4.3 Part
```json
{
  "part_id": "p3",
  "reconstruction_revision": 1,
  "measurement_revision": 2,
  "label": "slip nut",
  "label_confidence": 0.81,
  "description": "Plastic slip nut joining the J-bend to the tailpiece",
  "mesh_uri": "scans/scn_8f2a/parts/p3.glb",
  "thumbnail_uri": "scans/scn_8f2a/renders/p3.png",
  "centroid_mm": [12.0, 4.5, 140.2],
  "obb": { "center": [12, 4.5, 140], "extent": [62, 62, 28], "rotation_quat": [0, 0, 0, 1] },
  "dimensions": [
    { "name": "outer_diameter", "derived_from": "p3_f1.radius", "factor": 2.0 },
    { "name": "height",         "value": { "scan_mm": 27.6, "snapped_mm": null, "snap_rule": null, "caliper_mm": null, "user_confirmed": false, "preferred": "scan" } }
  ],
  "features": [
    { "feature_id": "p3_f1", "type": "cylinder",
      "axis_point_mm": [12.0, 4.5, 140.2], "axis_dir": [0, 0, 1], "fit_rmse_mm": 0.9, "inlier_fraction": 0.84,
      "radius": { "scan_mm": 30.6, "snapped_mm": null, "snap_rule": null, "caliper_mm": 31.0, "user_confirmed": false, "preferred": "caliper" },
      "length": { "scan_mm": 27.6, "snapped_mm": null, "snap_rule": null, "caliper_mm": null, "user_confirmed": false, "preferred": "scan" } },
    { "feature_id": "p3_f2", "type": "bore",
      "axis_point_mm": [12.0, 4.5, 140.2], "axis_dir": [0, 0, 1], "fit_rmse_mm": 1.1, "inlier_fraction": 0.71,
      "radius": { "scan_mm": 19.4, "snapped_mm": 19.05, "snap_rule": "bore_for_tubular_1_1_2_in", "caliper_mm": null, "user_confirmed": false, "preferred": "snapped" },
      "length": { "scan_mm": 27.6, "snapped_mm": null, "snap_rule": null, "caliper_mm": null, "user_confirmed": false, "preferred": "scan" } },
    { "feature_id": "p3_f3", "type": "thread_suspected", "nominal": "1-1/2 in slip joint", "confidence": 0.55 }
  ],
  "material_guess": { "value": "PVC/ABS plastic", "confidence": 0.6 },
  "source": "auto"
}
```
`source` is `auto` or `user_edited`. Every overridable scalar is a **Measurement** object `{ scan_mm, snapped_mm, snap_rule, caliper_mm, user_confirmed, preferred }`, where `preferred` is `caliper` whenever `caliper_mm` is set, else `snapped` if `snapped_mm` is set, else `scan`; `user_confirmed: true` records that the user accepted the preferred value in chat without measuring (it does not change `preferred`, but `param_sources` reports `user_confirmed` for it). Downstream consumers read the preferred value and record which it was. A `dimensions` entry either holds its own Measurement or is a **read-only view** of a feature Measurement (`derived_from`, with an optional `factor` such as 2.0 for a diameter from a radius), so one physical quantity is never stored twice; `PATCH` targets the feature. Geometric observations (`axis_point_mm`, `axis_dir`, `normal`, `center_mm`, `fit_rmse_mm`, `inlier_fraction`) are plain fields and cannot be overridden. Feature `type` is a discriminated union, each variant fully specified:

| `type` | Observation fields | Measurement fields |
|---|---|---|
| `cylinder`, `bore` | `axis_point_mm`, `axis_dir`, `fit_rmse_mm`, `inlier_fraction` | `radius`, `length` |
| `d_shaft` | as `cylinder` plus `flat_normal` | `radius`, `length`, `flat_depth` |
| `plane` | `point_mm`, `normal`, `fit_rmse_mm`, `inlier_fraction` | `extent_u`, `extent_v` |
| `hole` | `center_mm`, `normal`, `fit_rmse_mm` | `radius`, `depth` |
| `thread_suspected` | `nominal`, `confidence` | none |

`mates_with` in a CAD request references a `feature_id`, never a free-form name. Standard-size comparison is on **diameter**: `2 * radius.scan_mm` against the role-specific table in 6.2 step 4; the snapped radius stored is half the standard diameter.

### 4.4 Scene graph (`scene_graph.json`)
```json
{
  "scan_id": "scn_8f2a",
  "object": { "category": "sink drain P-trap", "summary": "White PVC 1-1/2 in P-trap under a kitchen sink" },
  "parts": ["p1", "p2", "p3", "p4"],
  "relations": [
    { "a": "p3", "b": "p2", "type": "threaded", "confidence": 0.7 },
    { "a": "p4", "b": "p3", "type": "seals", "confidence": 0.6, "note": "slip washer inside nut" }
  ],
  "function": "Traps water to block sewer gas; slip nuts compress washers to seal joints.",
  "failure_hypotheses": [
    { "part_id": "p3", "mode": "worn or misaligned slip washer", "evidence": ["water marks below joint in frame 0042"], "likelihood": 0.6 }
  ]
}
```
Relation `type` is one of: `threaded`, `press_fit`, `fastened`, `hinged`, `sliding`, `seals`, `glued`, `rests_on`, `unknown`.

### 4.5 Repair plan (agent output, rendered as a card)
```json
{
  "plan_id": "pln_01",
  "intent": "fix_leak",
  "diagnosis": { "summary": "Leak at the J-bend slip joint, most likely a worn slip washer.", "part_ids": ["p3", "p4"], "confidence": 0.65 },
  "safety": ["Put a bucket under the trap before loosening the nuts."],
  "tools": ["bucket", "channel-lock pliers (optional)"],
  "steps": [
    { "n": 1, "text": "Hand-loosen slip nut [[part:p3]] counter-clockwise.", "part_ids": ["p3"] },
    { "n": 2, "text": "Replace the slip washer: flat side against the nut, tapered side toward the fitting.", "part_ids": ["p4"] }
  ],
  "bom": [
    { "name": "1-1/2 in slip-joint washer", "spec": "polyethylene, beveled", "qty": 2, "est_price_usd": 3.0, "buy_links": [{ "vendor": "Home Depot", "url": "https://..." }] }
  ],
  "printable": [],
  "followups": ["Want me to show you which way the bevel faces?"]
}
```

### 4.6 CAD request (agent to CAD service)
```json
{
  "cad_id": "cad_07",
  "scan_id": "scn_8f2a",
  "reconstruction_revision": 1,
  "part_measurement_revisions": { "p2": 3 },
  "mode": "template",
  "template": "knob",
  "params": { "shaft_diameter": 6.0, "shaft_flat_depth": 1.5, "outer_diameter": 32.0, "height": 22.0 },
  "param_sources": { "shaft_diameter": "caliper", "shaft_flat_depth": "user_confirmed", "outer_diameter": "scan", "height": "scan" },
  "codegen_prompt": null,
  "mates_with": [{ "part_id": "p2", "feature_id": "p2_f1", "fit": "snug", "radial_clearance_mm": 0.12 }],
  "placement": { "translate_mm": [0, 0, 41.0], "rotation_quat": [0, 0, 0, 1] },
  "material": "PETG",
  "printer_profile": "prusa_mk4_petg_0.2"
}
```
`mode` is `template` (fill a known parametric part; the judged path) or `codegen` (stretch: LLM writes CadQuery code from `codegen_prompt`). `mates_with.feature_id` must resolve to a feature on that part (4.3) of a compatible type (a `knob` mates with a `d_shaft` or `cylinder`). `fit` is `snug` (a knob on a shaft; profile default about 0.1-0.15 mm radial for PETG, set from the tolerance coupon), `sliding` (about 0.2-0.3 mm radial) or `press`; the `radial_clearance_mm` value is copied from the printer profile for that fit type, not chosen by the agent. `param_sources` records where each number came from (`scan`, `snapped`, `caliper`, `user_confirmed`), so the result card can show it; for a `d_shaft` mate both `shaft_diameter` and `shaft_flat_depth` are mating parameters and both need a `caliper` or `user_confirmed` source for the fit check to be `required`. All clearances in these contracts are **radial** (per side); a 0.12 mm radial clearance on a 6.0 mm shaft means a 6.24 mm hole. `printer_profile` names a profile file (build volume, nozzle, perimeters, limits, clearance per fit type).

### 4.7 CAD result
```json
{
  "cad_id": "cad_07",
  "status": "ok",
  "input_revisions": { "reconstruction_revision": 1, "part_measurement_revisions": { "p2": 3 } },
  "stale": false,
  "files": { "stl": "cad/cad_07/model.stl", "step": "cad/cad_07/model.step", "source": "cad/cad_07/model.py", "preview": "cad/cad_07/preview.png" },
  "checks": [
    { "name": "watertight", "kind": "required", "result": "pass" },
    { "name": "fits_build_volume", "kind": "required", "result": "pass" },
    { "name": "mate_p2_f1_diameter_clearance_mm", "kind": "required", "result": "pass", "value": 0.12, "limit": 0.12, "input_source": "caliper" },
    { "name": "mate_p2_f1_flat_clearance_mm", "kind": "required", "result": "pass", "value": 0.12, "limit": 0.12, "input_source": "user_confirmed" },
    { "name": "mate_p2_f1_flat_orientation_deg", "kind": "required", "result": "pass", "value": 0.0, "limit": 2.0 },
    { "name": "mate_p2_f1_engagement_mm", "kind": "required", "result": "pass", "value": 14.0, "limit": 8.0 },
    { "name": "mate_p2_f1_coaxial_offset_mm", "kind": "required", "result": "pass", "value": 0.05, "limit": 0.3 },
    { "name": "min_wall_thickness_mm", "kind": "advisory", "result": "pass", "value": 1.6, "limit": 1.2 },
    { "name": "max_unsupported_overhang_deg", "kind": "advisory", "result": "fail", "value": 62, "limit": 50, "fix": "print upside down or add supports" },
    { "name": "minimum_surface_separation_mm", "kind": "advisory", "result": "pass", "value": 0.15, "note": "unsigned distance to the scanned mesh; cannot prove no interference on its own" },
    { "name": "interference_volume_mm3", "kind": "advisory", "result": "not_computed", "reason": "scanned mesh is not watertight" },
    { "name": "strength_estimate", "kind": "advisory", "result": "not_computed", "reason": "no effective-strength value in profile for this orientation" }
  ],
  "print_settings": { "material": "PETG", "layer_height_mm": 0.2, "infill_pct": 40, "orientation": "flat face down", "supports": false, "est_time_min": 48, "est_material_g": 14, "estimate_method": "volume_formula" },
  "notes": "Shaft hole has 0.12 mm radial clearance from the coupon-calibrated profile; test-fit and scale 101% if tight."
}
```
Every check has the same shape: `name`, `kind` (`required` or `advisory`), `result` (`pass`, `fail`, `not_computed`), and optional `value`, `limit`, `fix`, `reason`, `input_source`. `status` is `ok` (all `required` checks pass), `failed_checks`, or `error`. `advisory` checks never block; they are shown with their values. Thresholds come from the printer profile, not hard-coded constants. **Mate checks are defined per feature variant** (6.4): for a `cylinder` mate, diameter clearance, engagement length and coaxial offset; for a `d_shaft` mate, additionally flat clearance (bore flat depth against shaft flat depth) and flat orientation. A mate check is `required` only when every mating input's source is `caliper` or `user_confirmed`; when any is `scan` **or `snapped`**, the mate checks are reported as `advisory` and the agent must ask the user to confirm the dimension before offering a print (6.3). `stale` becomes `true` when the referenced part's revisions move on (section 4 intro).

### 4.8 Chat stream events (SSE framing over `POST /scans/{id}/chat`)
```
event: token        data: {"text": "The leak is most likely..."}
event: tool_call    data: {"name": "get_part", "args": {"part_id": "p3"}}
event: highlight    data: {"part_ids": ["p3"]}
event: card         data: {"type": "repair_plan", "payload": { ...4.5... }}
event: card         data: {"type": "cad_result", "payload": { ...4.7... }}
event: done         data: {"message_id": "msg_12"}
```
Both SSE routes (chat and progress) are read with `fetch` plus `eventsource-parser`, because the bearer token travels in a header and `EventSource` can neither POST nor set headers. The server sends a `: keep-alive` comment every 20 s on both.

---

## 5. API

All routes require the demo token (3.3). Scan assets are returned as short-lived signed URLs.

| Method and path | Description | Owner |
|---|---|---|
| `POST /scans` | Create a scan. Body: `{ input_type: "video" \| "photos" \| "mesh", mesh_units?: "metres" \| "millimetres" }`. Returns `scan_id` and pre-signed POST form(s) with content-type and size limits. | P4 (route), P5 (pre-signed POST helper) |
| `POST /scans/{id}/start` | Upload finished. API does only a HEAD (existence, size, content type), enqueues the contained worker-side validation job (magic bytes, `ffprobe` or image headers), then the pipeline. For `mesh` input (stretch): `frames`, `geometry`, `object_mask` and `splat` are `skipped`; the GLB is converted from Y-up/`mesh_units` to the scan frame, placed with Z from its support plane and origin at `object_bottom_center`, and perception steps 2-6 run on 24 rendered views of the mesh (colour and depth from the same virtual cameras, written to `geometry/` like real frames); the imported mesh is already isolated, so the object-mask step is skipped. | P4 (route), P5 (validation job) |
| `GET /scans/{id}` | Scan object (4.1). | P4 |
| `GET /scans/{id}/events` | SSE stream of stage progress (read with `fetch`, bearer header). | P4 |
| `GET /scans/{id}/reconstruction` | 4.2 plus signed asset URLs. | P2 |
| `POST /scans/{id}/scale` | Manual scale: `{ point_a_mm, point_b_mm, known_length_mm, expected_revision }` (points in the current scan frame). **Sets uniform scale only**; origin and orientation stay as the reconstruction defined them (`frame_origin` unchanged). A uniform scale `s` about the scan-frame origin is analytic, so nothing is re-segmented and no model is re-run: multiply camera translations, depth, gaussian means (and add `ln(s)` to their log scales), mesh vertices, part GLB vertices, centroids, OBB centres and extents, feature axis points and every Measurement's `scan_mm` by `s`; keep every `part_id`, `feature_id`, merge, rename and label; re-evaluate snaps; keep `caliper_mm` values (they are physical); leave the scene-graph text alone apart from the numbers it quotes. Increments `reconstruction_revision` and every part's `measurement_revision`, marks every CAD result for this scan `stale`, and holds `parts` at `running` until done. Rejects a stale `expected_revision`. | P2 |
| `GET /scans/{id}/parts` | List of Part (4.3). | P3 |
| `PATCH /scans/{id}/parts/{partId}` | Rename or relabel a part (no revision change); enter a `caliper_mm` for any feature Measurement (radius, length, flat depth, hole radius), or set `user_confirmed: true` on one, which sets `source: "user_edited"`, increments the part's `measurement_revision`, and marks CAD results that referenced this part `stale`. Setting `caliper_mm` on a snapped Measurement overrides a false snap. The agent's `set_measurement` tool (6.3) calls this route for values given in chat. | P3 |
| `POST /scans/{id}/parts/merge` | Merge parts: retires both source `part_id`s, creates the merged part (new id, re-fitted features, `measurement_revision` 1) under the current `reconstruction_revision`, recomputes the scene-graph relations for it, and marks CAD results referencing either source part `stale`. | P3 |
| `GET /scans/{id}/scene-graph` | 4.4. | P3 |
| `POST /scans/{id}/measure` | Distance between two 3D points or two parts. | P3 |
| `POST /scans/{id}/chat` | Send a message (plus optional images); SSE-framed streaming response (4.8). | P4 |
| `GET /scans/{id}/chat` | Chat history. | P4 |
| `POST /scans/{id}/cad` | Submit a CAD request (4.6); returns `cad_id`. Rejected with 409 if the request's revisions do not match the current part. | P5 |
| `GET /scans/{id}/cad/{cadId}` | CAD result (4.7), with `stale` computed against the current part revisions. | P5 |
| `GET /demo-scans` | Pre-processed demo scans for the landing page. | P1 |

Ownership of the shared platform (matches section 7): **P4** owns the FastAPI app shell (`backend/app/`: routing, auth and limits, database, queue wiring, SSE helpers) and the contract models; **P5** owns the storage helpers (pre-signed POST, signed GET, CORS), the upload validation job and deployment; **P2** owns the reconstruction handlers (`/reconstruction`, `/scale`) plugged into P4's router shell. Each person owns their own router module inside the shell.

---

## 6. Pipeline details

### 6.1 Reconstruction (feed-forward, no per-scene optimization)

Stage order on the single GPU worker: **frames → geometry → scale → object mask → splat + mesh → parts.** The object mask is P3's SAM 2 function called as a reconstruction sub-stage (it needs frames, not the mesh), which is what breaks the P2/P3 dependency cycle; until P3's code lands, P2 uses a stub that keeps everything inside a box around the board-relative object centre. **Freeing VRAM between models:** DA3's output processor moves depth, confidence and cameras to the CPU but leaves the gaussian tensors on the GPU inside the prediction, and `torch.cuda.empty_cache()` cannot release memory that live tensors still reference. So, before SAM 2 loads, the worker writes the `gs_ply` and the depth/confidence arrays to disk, moves or deletes every retained GPU member of the prediction, deletes the raw output and the model, runs `gc.collect()` and `torch.cuda.empty_cache()`, and asserts `torch.cuda.memory_allocated()` is under a configured threshold (fail the job loudly otherwise). This transition is part of the day-one VRAM measurement.

1. **Frame extraction:** probe the file first (`ffprobe`), then derive the sampling rate from the duration so that after dropping the blurriest 30% (variance of Laplacian) **48-64 keyframes** remain (a 20 s clip needs about 4 fps, a 60 s clip about 1.5 fps). Keep two copies of each keyframe: full resolution for board detection and the agent's close-ups, and the **model grid** copy. Depth Anything 3 resizes the long side to 504 and rounds each side to a multiple of 14, so 1920x1080 becomes 504x280 with slightly different scale factors per axis; `frames.json` records the exact per-axis mapping and the returned `K` applies to that grid. Accept MOV/MP4/WebM with H.264/HEVC/VP8/VP9 and normalise with ffmpeg (the build must decode HEVC) inside the containment described in 3.3. iPhone video is 10-bit HDR by default; tone-map it to 8-bit (`zscale` + `tonemap`) or frames come out washed out. For photo input, normalise EXIF orientation and resize all images to one size first; the model stacks them into one tensor.
2. **Geometry in one pass:** run Depth Anything 3 (`DA3NESTED-GIANT-LARGE-1.1`; the unsuffixed checkpoint is deprecated) with `infer_gs=True`, `ref_view_strategy="middle"` for video, `export_dir=<scan>/geometry`, `export_format="mini_npz-gs_ply"` (the `gs_ply` exporter only runs when the format string includes it and an export directory is set), and `export_kwargs={"gs_ply": {"gs_views_interval": 4}}`. One call returns per-frame camera extrinsics (world-to-camera, OpenCV axes) and intrinsics, per-frame depth and confidence, and the gaussians (the export code reads `prediction.gaussians`; P2 confirms on day one which attribute the installed version exposes and applies the `scale_factor` correction to that object). The alternative geometry model, MapAnything (`facebook/map-anything-apache`, `memory_efficient_inference=True`), returns camera-to-world poses (the adapter inverts them), per-view point maps and a metric scale factor, but no gaussians. **VRAM is the real constraint, not time:** DA3's own streaming measurements show about 21 GB for 60 frames at 504x378 and the nested model adds the metric head, and users report out-of-memory above roughly 100 images with `infer_gs=True` on 24 GB. Start at 48 views on the chosen instance, measure peak memory on day one, and pin the cap in config; the out-of-memory ladder is fewer views, then `infer_gs=False` (mesh only; DA3-Large has no Gaussian head and no metric head, so it is not a substitute), then a 48 GB instance. If the model's confidence is low on most frames, mark the scan low-quality and ask for a slower, better-lit re-capture.
3. **Scale (authoritative):** detect the printed ChArUco board in the **full-resolution** keyframes with `cv2.aruco.CharucoDetector` (OpenCV 4.7+), which gives at least 16 well-spread corners; map each corner into model-grid pixel coordinates with the exact per-axis scale from `frames.json`. The geometry model assumes an undistorted pinhole camera and returns `K` without distortion coefficients, so no undistortion is applied; use **pixel** coordinates with `P = K [R | t]` (never normalised coordinates with `K` as well). Drop observations with a short baseline or large reprojection error, and triangulate each corner across all views that saw it (pairwise `cv2.triangulatePoints` then a least-squares refinement, or a direct multi-view DLT). Fit a robust similarity transform (RANSAC over corner subsets, Umeyama inside) from the recovered corner positions to the board's known coordinates. That one transform maps the model frame to the scan frame and is applied to cameras, depth, mesh and gaussians (3.1). Record `corner_views` and `residual_mm`; a residual above 2 mm (corners are located at 504 px, so about 1 mm per pixel) or fewer than 8 corners seen in 5 or more views flags the scan. Fallbacks, in order: the model's own metric scale (DA3 nested head or MapAnything's `metric_scaling_factor`, recorded as `method: "learned"` and shown as approximate, with rotation from the support plane now and the origin fixed by the object-mask stage, as in 3.1), then the manual two-point `/scale` route (scale only). Accuracy target: within 2% on a known 100 mm test object with the board.
4. **Splat for display:** DA3's gaussians are pixel-aligned, about one per pixel per exported view, so with `gs_views_interval: 4` and 48 views they start at roughly 1.7 M before cropping; without the interval, 48 views would give about 7 M. Apply the nested-model `scale_factor` correction and then the scan-frame similarity (both in 3.1; mind the `[w, x, y, z]` quaternion order of the PLY), crop to the object (step 5), drop gaussians below an opacity threshold, voxel-downsample to a hard cap of **1.5 M**, and pack to `.spz` with `PackOptions.from = UNSPECIFIED` so the scan-frame coordinates are stored unchanged (3.1). At SH degree 0, `.spz` is roughly 20 bytes per gaussian against about 60 for PLY (about 3x smaller, not 10x), so 1.5 M gaussians is about 30 MB. Spark loads `.spz` directly. If DA3's gaussians look poor on the hero object, AnySplat is the alternative feed-forward splat model: it predicts its own poses in its own frame, so run the same ChArUco scale step on its poses (or ICP-align its gaussians to the TSDF mesh) before display. If both disappoint, or the licence gate fails, the viewer shows the vertex-coloured mesh and the splat toggle is hidden. There is no optimization fallback.
5. **Object isolation:** the main-object mask (SAM 2 video predictor, 6.2 step 1, on the model-grid frames) is applied to the depth maps directly, and its 3D bounding box (from the masked depth) plus a 10 mm margin crops the gaussians. In the no-board case this bounding box also fixes the scan-frame origin (3.1) before anything is written. Low-confidence depth pixels are dropped. The table and background are never meshed.
6. **Mesh:** fuse the masked depth maps with their poses in Open3D's `ScalableTSDFVolume`. **Unit boundary: Open3D works in metres, full stop.** The legacy `ScalableTSDFVolume::Integrate` allocates its voxel blocks from a point cloud it builds internally with a **hard-coded** depth scale of 1000 (verified in the 0.20.0 source), while the voxel integrator itself uses the depth as converted by `RGBDImage.create_from_color_and_depth`. Feeding millimetre-valued depth therefore opens blocks at 1/1000 of the right positions and yields an empty or shredded mesh; millimetres inside Open3D are not an option. Convert at the boundary instead: depth to metres and camera translations to metres (the rotation is unchanged), then `RGBDImage.create_from_color_and_depth(color, depth_m_as_float32, depth_scale=1.0, depth_trunc=1.5, convert_rgb_to_intensity=False)`, a volume with `voxel_length=0.001`, `sdf_trunc=0.004` and `color_type=TSDFVolumeColorType.RGB8`, `integrate(rgbd, intrinsic, extrinsic_m)` with the world-to-camera extrinsic (translation in metres) and a colour image of the same size as the depth image; multiply the extracted mesh's vertices by 1000 before writing any scan-frame artifact. Pin the Open3D version and keep a golden TSDF test (a synthetic 100 mm cube rendered to depth from four poses must come back as a 100 mm cube) so a version bump cannot silently break this. Extract with marching cubes, keep the largest connected component, close small holes with `pymeshfix`, then decimate to under 200k triangles for the web (vertex colour survives decimation only coarsely; the mesh view is "clean", not photoreal); keep the full mesh for measurement. Depth from a feed-forward model is smooth and consistent across views, which TSDF handles well, but its resolution limits fine detail; see the accuracy note in 6.2 step 4.
7. **Mesh import (stretch):** `input.type: "mesh"` accepts one self-contained `.glb` from a phone scanning app (Polycam, Scaniverse, Luma). OBJ/PLY are not accepted (external materials, no single-file guarantee). The GLB is Y up in `mesh_units` (declared at `POST /scans`); it is rotated to Z up, converted to millimetres, and placed with the support-plane frame of 3.1. Perception then runs on 24 rendered views of the mesh (colour and depth from virtual cameras on a sphere, written to `geometry/` and `frames/model/` exactly like real frames); the imported mesh is already isolated, so 6.2 step 1 is skipped and steps 2-6 run unchanged. No splat is shown for imported meshes.

### 6.2 Perception
All image-space work (masks, projection, label lifting) uses the **model-grid** frames and the matching `K` from `cameras.json`; the agent's close-ups and evidence crops use the full-resolution frames.
1. **Main-object mask (called by the reconstruction stage, 6.1):** SAM 2 video predictor propagated through the keyframes. The prompt is not the image centre (a P-trap is hollow and off-centre): for hero scans, a stored, verified set of positive (label 1) and negative (label 0) points; for new scans, a box prompt from Grounding DINO with the object name the user typed or Claude inferred from three keyframes, plus one **negative** point (label 0) placed on the board so it is excluded from the mask.
2. **Part proposals:** send 3-5 well-spread keyframes to Claude and ask for the list of visible component names (for a P-trap: tailpiece, slip nut, J-bend, trap arm, washer). Feed those names as text prompts to Grounded SAM 2 (Grounding DINO finds boxes for each name, SAM 2 turns them into masks and tracks them across the video). Do not ask the vision model for pixel coordinates; general VLMs are unreliable at pointing. Fallback when Grounding DINO misses a part: `SAM2AutomaticMaskGenerator` on the keyframes, then ask Claude to label each mask crop and say which masks belong to the same part.
3. **Lift to 3D:** for each mesh vertex, project it into each keyframe with `P = K [R | t]` from `cameras.json` (scan frame, see 3.1), look up the mask label, and take a majority vote; ignore views where the vertex is occluded by comparing against that frame's depth map. Smooth labels over the mesh adjacency graph (a few rounds of majority voting among neighbours). Split the mesh into one `.glb` per part.
4. **Measurement:** per part, compute an oriented bounding box (Open3D `get_oriented_bounding_box`); fit primitives with RANSAC. Open3D has RANSAC planes only (`segment_plane`), so use `pyransac3d` for cylinders and circles, and refine each fit with a least-squares pass (`scipy.optimize.least_squares`) because pyransac3d's cylinder fit is noisy on its own. Every feature records its observation fields and Measurement objects as defined in 4.3. **Snapping** compares the scan **diameter** (`2 * radius.scan_mm`) against a table chosen by the feature's **role**, so that entries within one table are far enough apart to be told apart at this accuracy: exterior `cylinder` features use the tube table (tubular drain 1-1/4" = 31.75 mm and 1-1/2" = 38.1 mm outer diameter, rules `tubular_1_1_4_in` and `tubular_1_1_2_in`); `bore` features use the bore table (the nut and washer bores that accept those tubes, nominal 31.75 and 38.1 mm, rules `bore_for_tubular_1_1_4_in` and `bore_for_tubular_1_1_2_in`). Metric bolt sizes (M3-M10, 1-2 mm apart) and common shafts (6.0 versus 6.35 mm) are **never** snapped; they appear only as suggestions in the agent's wording. A snap fires only when the scan diameter is within **2.0 mm** of one entry in its table and the next nearest entry in that table is at least 6 mm away; with the expected 2-4 mm error a wrong snap is still possible above about 4.3 mm of error, so a snapped value is treated like a scan value for everything that matters physically (confirm-before-printing in 6.3; advisory mate checks in 4.7) and only drives the buy recommendation, where the agent also says "measured about 37 mm, most likely 1-1/2 in; check the size stamped on the nut". A person clears a false snap by entering `caliper_mm`. **Accuracy, honestly:** depth from a 504 px feed-forward model gives roughly 0.5-1 mm per pixel on a 300 mm object and a few pixels of depth noise, so expect fitted diameters within about 2-4 mm, not sub-millimetre. That picks the right template and usually the right tubular size, but not reliably enough to skip the confirmation above. Small features are worse: a 6 mm shaft spans 6-12 pixels, its 1.5 mm flat one to three, and a thread less than one, so with a fit error of several millimetres none of them yields a usable mating dimension; the Part record carries them as `caliper_mm` entered by a person, and the UI shows which source each number came from. Do not claim an automated accuracy figure unless P3's evaluation script (bullet 9 in the P3 workload) shows it on the hero objects; the evaluation reports the false-snap rate explicitly.
5. **Labelling and materials:** render each part (thumbnail plus highlighted context view) and send it to Claude with the object context. Get back label, description, material guess, confidences.
6. **Relations:** candidate pairs are parts whose surfaces are within 2 mm of each other (nearest-neighbour distance between the two part point sets). Classify the relation type with geometry cues (coaxial cylinders suggest threaded or press-fit) plus Claude. Write `scene_graph.json`, including failure hypotheses from visual evidence (water stains, cracks, rust) when the user's question or the images suggest it.

### 6.3 Agent
- **Model:** Claude with tool use, streaming. System prompt: role, safety rules, units (mm, plus inches for US plumbing sizes), citation of part chips, and "never invent a dimension you did not measure; say it is estimated".
- **Tools:**

| Tool | Returns |
|---|---|
| `get_scene_graph()` | 4.4 |
| `get_part(part_id)` | 4.3 |
| `list_parts()` | Short part list |
| `measure(a, b)` | Distance in mm between points or parts |
| `render_view(part_ids, angle)` | Image of the part(s), for visual reasoning. Rendered **on the CPU** (offscreen pyrender/OSMesa or trimesh) in the API or CPU worker, never queued on the GPU worker; for hero scans the renders are precomputed into `renders/` |
| `get_keyframes(query?)` | Relevant raw frames (for example close-ups) |
| `highlight(part_ids)` | Emits a `highlight` event to the viewer |
| `set_measurement(part_id, feature_id, field, caliper_mm?, user_confirmed?)` | Writes a value the user gave in chat through `PATCH /parts/{id}` (section 5), so the Part record and `measurement_revision` change; returns the updated Measurement |
| `search_parts(query, spec)` | Purchasable items with price and link |
| `create_cad(request)` | Starts a CAD job (4.6), waits, returns 4.7 |
| `emit_repair_plan(plan)` | Validates and emits a Repair Plan card (4.5) |

- **Behaviours:**
  - *Explain:* "how does this work?" uses the scene graph plus renders.
  - *Diagnose:* asks at most one or two clarifying questions (where is the water, when does it leak), then ranks failure hypotheses.
  - *Fix:* produces a Repair Plan; uses `search_parts` for every BOM line.
  - *Modify:* decides buy versus print. Prints only when the part is a reasonable FDM part (not pressure-bearing plumbing seals, not electrical, not safety-critical) and says why.
  - *Confirm before printing:* for every template parameter that will mate with the original part and whose source is `scan` **or `snapped`** (no `caliper_mm`, not `user_confirmed`), the agent tells the user the scan value and its uncertainty and asks them to measure it (the value goes in through `set_measurement` as `caliper_mm`) or to accept it (`set_measurement` with `user_confirmed: true`) before calling `create_cad`. A mating dimension is never sent from a scan or snapped value alone; the CAD service reports the mate checks as advisory in that case (4.7). This is the designed path for small features the scan cannot resolve (6.2 step 4), not a failure mode. Non-mating parameters (a knob's outer diameter, a bracket's overall length) may come from the scan and are shown with their source.
  - *Feasibility:* if required CAD checks fail, the agent revises parameters and retries up to 3 times, then explains the trade-off.
- **Parts search:** a curated catalogue JSON for the demo categories (washers, slip nuts, P-traps, faucet cartridges, common screws) with real store links. Optionally add a live product-search API if one is available. Curated is enough for judging and avoids broken results on stage.
- **Before parts are ready:** the same agent works on keyframes only; tools that need parts return "not ready yet".

### 6.4 CAD and fabrication
- **Template library** (parametric CadQuery/build123d functions, each with a typed parameter schema): knob (round/D-shaft/knurled), L-bracket with holes, spacer/washer, hose or pipe adapter (two diameters), cable clip, hook, cap/plug, simple enclosure box. Templates are the judged path.
- **Codegen mode (stretch only):** the LLM writes a CadQuery script against a fixed API (`def build(params) -> cq.Workplane`). It runs in the disposable container described in 3.3; on error, the traceback goes back to the LLM for up to 3 repair attempts. Not built until the template path is proven end to end.
- **Fit to the scan:** parameters come from Part measurements (4.3) through each Measurement's `preferred` value (`caliper`, else `snapped`, else `scan`), with each source recorded in `param_sources`; the clearance for the chosen `fit` type comes from the printer profile, not from the agent. **What the mate checks prove:** for caliper-sourced inputs the generated bore is by construction the shaft dimension plus the profile clearance, so the clearance checks catch template bugs (wrong parameter wiring, unit slips), while the coaxial-offset, engagement and flat-orientation checks verify the placement against the fitted primitive on the original part (axis point and direction from `feature_id`). The physical evidence that the part will fit is the tolerance coupon plus the caliper value, and the result card says so. Mate checks are defined per feature variant: `cylinder` mates get diameter clearance, engagement length and coaxial offset; `d_shaft` mates additionally get flat clearance (bore flat depth against shaft `flat_depth`) and flat orientation against `flat_normal`, because a D-bore with the right diameter and the wrong flat will not mount. They are `required` only when every mating input's source is `caliper` or `user_confirmed`; otherwise `advisory` (4.7). A secondary **advisory** number, `minimum_surface_separation_mm`, is the unsigned distance from sample points on the new part to the scanned part mesh (`trimesh.proximity.closest_point`); it cannot tell a gap from a penetration on its own, so it never decides fit. `trimesh.proximity.signed_distance` and mesh booleans need a watertight mesh and scanned meshes usually are not; `interference_volume_mm3` is therefore reported as `not_computed` unless the scanned part has first been closed and the result verified. Closing (stretch): `pymeshfix` on the part mesh, or voxelise with `mesh.voxelized(pitch).fill()`, convert the `VoxelGrid` back to a surface with `.marching_cubes` (applying the grid's transform), fix normals with `trimesh.repair.fix_normals`, confirm `is_watertight` and positive volume, and only then intersect with `manifold3d` as the boolean engine.
- **Printability checks:** `watertight` and `fits_build_volume` are required; the rest are advisory with thresholds from the selected printer profile: minimum wall thickness with `trimesh.proximity.thickness` on a few thousand surface samples (profile default 1.2 mm for a 0.4 mm nozzle, three perimeters), overhang angle per face (profile default 50 degrees) after choosing the best of 6 axis-aligned orientations, small-feature check (profile default 0.8 mm). Print the hero template with a tolerance coupon (holes at 0.05, 0.10, 0.15, 0.20 and 0.30 mm radial clearance around a 6.0 mm D-shaft) before judging and set the profile's `snug`, `sliding` and `press` clearances from what actually fits.
- **Strength heuristic:** for brackets and hooks, a simple cantilever beam calculation (load from the user or a default 2 kg) with a safety factor of 3 against an **effective** strength taken from the printer/material profile for the chosen print orientation (FDM parts are strongly anisotropic; bulk yield strength overstates them). If the profile has no effective-strength value for that orientation, the check returns `not_computed` with that reason. Shown as an estimate, never as FEA, never a pass/fail gate.
- **Outputs:** STL (print, millimetres), STEP (editing), the CadQuery source, preview PNG, print settings (material, layer height, infill, orientation, supports, time and material estimate from a volume formula labelled as such; PrusaSlicer or OrcaSlicer CLI for a real estimate is stretch).

---

## 7. Team split (5 people)

Each person owns one area end-to-end: code, its API routes, its fixtures, and its part of the demo. Everyone builds against the fixtures in `fixtures/` first, so nobody waits on anyone.

### Summary
| Person | Area | Main output | Depends on (uses fixtures until ready) |
|---|---|---|---|
| P1 | Website and 3D viewer | Everything the user sees | All APIs |
| P2 | Reconstruction and GPU pipeline | Feed-forward geometry, scale, splat, mesh; the GPU worker and its image; hero pre-processing | P3's object-mask function (stubbed until it lands) |
| P3 | Perception (parts and measurement) | Object mask, parts, dimensions, scene graph | P2's mesh and cameras |
| P4 | AI agent, plus the API platform | Chat, diagnosis, repair plans, parts search; FastAPI shell, auth and limits, database, queue and SSE | P3's scene graph, P5's CAD API |
| P5 | CAD and fabrication, plus storage and deployment | Printable, fitted CAD parts with checks, the printed part; S3 pre-signed POST and CORS, upload validation job, HTTPS deployment | P3's part features |

The platform work (API shell, auth, database, queue, storage, deployment) is split off P2 and given to P4 and P5, whose first days are otherwise spent against fixtures; P2 keeps only the GPU pipeline, which is the critical path. Shared, first thing: everyone agrees on section 4 contracts and section 5 routes together; P4 commits the Pydantic models and P1 the TypeScript mirror. Each person commits at least one realistic fixture for their outputs.

---

### P1: Website and 3D viewer (frontend lead)
**Owns:** `web/` (Next.js app), TypeScript contract types, demo-scan gallery, overall look and the demo click-path.

**Workload:**
1. **App shell:** Next.js + Tailwind, routes in 2.1, two typed clients from the contracts (the API client adds the bearer header; the storage client never does), environment config for the API URL, demo token read from the URL fragment on first load, kept in `sessionStorage` and removed from the address bar with `history.replaceState` (3.3).
2. **Capture page:** on phones, `<input type="file" accept="video/*" capture="environment">` as the preferred path (on most phones it opens the native camera; `capture` is a hint, not a guarantee, so the ordinary file picker is always there too), drag-and-drop upload for video/photos on desktop, optional `MediaRecorder` for webcams, capture tips including "turn off HDR video", scale-board PDF download with the print-at-100% check, client-side pre-checks (size, duration) as a courtesy only (the server re-validates), pre-signed POST upload with progress. Test on the exact phones used at judging; expect MOV/HEVC from iPhones.
3. **Status bar:** read `/scans/{id}/events` with `fetch` plus `eventsource-parser` (bearer header); drive each panel from the `stages` array, not from the top-level status; `skipped` stages greyed out; friendly error messages ("Not enough sharp frames; try a slower orbit").
4. **3D viewer (largest task):**
   - react-three-fiber scene with orbit controls, lighting, grid at the scan-frame origin.
   - One parent group that applies `scan_to_web` (3.1); every asset is a child of it. Verify with the asymmetric 100 mm test object on day one and keep it as a golden fixture.
   - Splat rendering with Spark 2.x: one `SparkRenderer` created with the react-three-fiber `gl` and added to the scene (disposed on unmount), plus a `SplatMesh` loading `.spz` under the parent group; mesh rendering of the per-part `.glb` files.
   - Splat/mesh toggle (hidden when `splat_uri` is null), hover and click picking on parts, selection outline, part colours.
   - Exploded-view slider.
   - Two-click measurement tool calling `/measure` (or computed client-side on the mesh); results in millimetres from the scan frame.
   - Manual scale step: when `scale.method` is `learned` or the user asks, two clicks on a known length (credit card long edge 85.60 mm by default) and a `POST /scale` with `expected_revision`.
   - CAD overlay: load the generated STL at `placement` (translation plus quaternion), distinct colour, toggle, clearance badge from the `radial_clearance_*` check with its `input_source`.
   - Camera "focus on part" animation when a part chip is clicked.
5. **Parts panel:** list, thumbnail, label, confidence, and each dimension with its scan value, snapped value and caliper value side by side; rename and merge actions; an "enter caliper value" field that calls `PATCH /parts/{id}`.
6. **Chat panel:** `fetch` streaming with the same SSE parser, markdown, `[[part:pX]]` chips wired to the viewer, image attachments, and the three cards: Repair Plan (steps with checkboxes, safety callout, tools), Parts-to-Buy (price, quantity, links, total), CAD Result (preview, required checks as pass/fail and advisory checks as values, `param_sources` shown per parameter, print settings, STL/STEP download, "show in viewer").
7. **Landing page and demo gallery** with the pre-processed hero scans so the demo works even if live processing is slow.
8. **Polish:** loading skeletons, empty states, mobile layout for the capture page, keyboard shortcuts in the viewer.

**Done when:** a judge can go from landing page to a cached hero scan, click parts, chat, see a Repair Plan and a CAD part overlaid on the model, and download the STL, without help.

**Fallbacks:** if splat rendering is unstable in the browser, ship mesh-only view; if SSE is fiddly, poll every 2 s.

---

### P2: Reconstruction and GPU pipeline
**Owns:** `backend/recon/`, the GPU worker and its Docker image, scan and scale route handlers (the route shell itself is P4's), hero pre-processing.

**Workload:**
1. **GPU environment:** provision the GPU instance, install CUDA, PyTorch, Depth Anything 3 (with the pinned gsplat dependency its Gaussian head needs; build it in the image, not at demo time), MapAnything as the alternative, SAM 2 and Grounded SAM 2 (shared with P3), Open3D, pymeshfix, ffmpeg with HEVC decoding and tone-mapping filters. Pre-download all model weights into the image. One RQ `SimpleWorker` on the GPU queue, concurrency 1, generous job timeouts.
2. **Pipeline orchestration on the worker:** the stage order in 6.1, stage updates and SSE events, freeing the geometry model before SAM 2 loads, calling P3's object-mask function (stubbed until it lands).
3. **Frame extraction** with ffmpeg: probe, normalise container and codec, tone-map HDR, derive the sampling rate from the duration, sharpness filtering, full-resolution and model-grid copies with the per-axis mapping recorded (6.1 step 1).
4. **Feed-forward geometry** (6.1 step 2): one call for poses, depth, confidence and gaussians; low-quality detection from the confidence maps. **Day-one spike:** run the hero video through DA3 and MapAnything on the chosen instance, record peak VRAM at 32, 48 and 64 views, and pin the view cap.
5. **Metric scale** from the ChArUco board with multi-view triangulation and a robust similarity fit (6.1 step 3); the support-plane frame when there is no board; the learned-scale fallback; the `/scale` handler with revisioning (section 5). Accuracy target: within 2% on a known 100 mm test object.
6. **Splat export** (6.1 step 4): nested-model scale correction, quaternion reorder, scan-frame transform, crop, pruning to the gaussian cap, `.spz` packing with `from = UNSPECIFIED` at the version Spark reads; check it in Spark the same day.
7. **Object isolation and mesh** (6.1 steps 5-6): masked depth into Open3D TSDF with the millimetre settings, cleanup with pymeshfix, web-decimated `.glb` plus full mesh.
8. **Pre-process the hero scans** so demo scans are ready before judging, and keep the raw videos so they can be re-run after any pipeline fix.
9. **Stretch:** GLB mesh import (6.1 step 7).

**Done when (judged):** the cached hero scans show a scaled mesh, and a splat if the licence is cleared, in P1's viewer, and the 100 mm test object measures within 2 mm there. **Stretch:** a fresh upload of the hero video reaches `mesh` `done` in under two minutes with live stage progress.

**Fallbacks:** MapAnything if DA3 misbehaves (mesh only); fewer views on out-of-memory; mesh-only view if the splat looks poor; pre-processed demo scans.

---

### P3: Perception (parts, measurement, scene graph)
**Owns:** `backend/perception/`, the object-mask function, parts and scene-graph routes, measurement route.

**Workload:**
1. **Main-object segmentation** with the SAM 2 video predictor on the model-grid frames, exposed as a function P2's worker calls as a reconstruction sub-stage (6.1, 6.2 step 1): Grounding DINO box prompt from the object name, stored verified prompts for the hero scans, one negative (label 0) point on the board. Deliver this first; P2 is blocked on it for cropping, TSDF masking and, in the no-board case, the scan-frame origin.
2. **Part proposal and consistent masks** across keyframes: Claude names the parts, Grounded SAM 2 finds and tracks them; automatic masks plus Claude labelling as the fallback (6.2 step 2).
3. **2D-to-3D label lifting** onto the mesh with `P = K [R | t]` from `cameras.json`, visibility checks and label smoothing; export one `.glb` per part (6.2 step 3).
4. **Geometry and measurement:** OBB, pyransac3d cylinder/circle fits refined by least squares, Open3D plane fits, the typed feature records and Measurement objects of 4.3, the tubular-only snapping table with its 2.0 mm / 6 mm-separation rule (6.2 step 4). Target, measured by the evaluation script in bullet 9: scan diameters of features over 25 mm within 2-4 mm of calipers on the hero objects, zero false snaps on the hero objects, and the false-snap rate on everything else reported as a number rather than promised; small features left unsnapped with a clear "not measurable from scan" state.
5. **Part labelling and materials** with the vision model on rendered thumbnails (6.2 step 5); thumbnail renderer (shared with P4's `render_view` tool).
6. **Relations and scene graph,** including failure hypotheses from visual evidence (6.2 step 6).
7. **Part editing endpoints:** rename and relabel (no revision change), merge (retires the sources, new merged part, CAD referencing them marked stale), and caliper or user-confirmed entry on feature Measurements (`PATCH /parts/{id}`), which sets `source: user_edited` and increments that part's `measurement_revision`; `reconstruction_revision` changes only through `/scale` (section 4 intro, section 5). Split is stretch.
8. **Measure endpoint:** point-to-point and part-to-part distances on the full mesh.
9. **Evaluation:** a small script that compares scan dimensions on the hero objects with caliper readings and reports the snap hit rate, to show honest accuracy numbers during the pitch.

**Done when (judged):** the cached hero scans have correctly named parts (P-trap: tailpiece, slip nuts, J-bend, trap arm, washers where visible), every dimension shows its scan, snapped and caliper values, the snapped sizes match the real parts, and the scene graph is sensible. Every hero result is checked by a person before judging, with caliper values entered where the scan was off.

**Fallbacks:** if 3D lifting is noisy, merge parts by hand and enter caliper values for the dimensions that matter; the user-visible result stays correct because the caliper source is preferred.

---

### P4: AI agent, plus the API platform
**Owns:** `backend/agent/`, chat routes, system prompts, tool layer, parts catalogue; and `backend/app/` (FastAPI shell, auth and limits, database, queue wiring, SSE helpers).

**Workload:**
1. **API platform (first two days, alongside fixtures):** FastAPI app skeleton with a router per owner, demo-token auth dependency and per-token limits (3.3), CORS, Pydantic contract models in `backend/contracts/`, database tables (scans, stages, reconstruction revisions, parts, chat messages, CAD jobs), Redis + RQ queue wiring, SSE helper with the 20 s keep-alive, mock mode that makes every route return fixtures so P1 can build the whole UI early.
2. **Chat endpoint** with SSE streaming of tokens, tool calls, highlights and cards (4.8); chat history persistence; image attachments passed to the model.
3. **Tool layer** (6.3 table): implement each tool against P3 and P5 APIs; until those exist, run against fixtures.
4. **Prompts and behaviours:** explain, diagnose (with limited clarifying questions), fix, modify, confirm-before-printing, with the safety rules and "do not invent dimensions" rule. Include a buy-versus-print decision rubric.
5. **Repair Plan output:** structured output validated against 4.5; part chips in steps; safety notes.
6. **Parts search:** curated catalogue JSON for the demo categories with real links and prices (including both 1-1/2" and 1-1/2" x 1-1/4" reducing slip washers), matched by spec and size from P3's preferred dimension values; optional live search API behind the same tool.
7. **CAD orchestration:** turn "make me a replacement knob" into a 4.6 request (choose template, fill parameters from the preferred Measurement values, record `param_sources`, reference `feature_id`s, and first get every `scan` or `snapped` mating dimension measured or accepted by the user through `set_measurement`), call P5, read the checks, revise and retry on required-check failure, and explain the result.
8. **Before-parts mode:** keyframe-only reasoning so chat is useful within a minute of upload, while perception is still running.
9. **Evaluation set:** 15-25 scripted questions per hero object with expected key facts (correct part, correct size, safe steps). Run it after prompt changes so the demo does not regress.
10. **Guardrails:** refuse or warn on dangerous tasks (gas lines, mains electrical, pressure vessels), recommend a professional where appropriate, and say when confidence is low.

**Done when (judged):** on the cached hero scans, for Scenario A the agent identifies the leaking joint, recommends the correct washer size with a link, and gives safe steps; for Scenario B it asks the user to confirm the shaft dimensions the scan cannot resolve, then produces a CAD request that P5 turns into a part passing the required checks.

**Fallbacks:** if tool calling is unreliable, use a fixed flow per intent (diagnose then plan then BOM) with the LLM filling each step.

---

### P5: CAD and fabrication, plus storage and deployment
**Owns:** `backend/cad/`, CAD routes, template library, printability and fit checks, printer profile, the physical print; and storage (S3 pre-signed POST, CORS, upload validation job) and deployment (HTTPS, Docker, firewall).

**Workload:**
1. **Storage and deployment (first two days, alongside fixtures):** S3 bucket, pre-signed POST with content type and `content-length-range`, signed GET, bucket CORS; the worker-side validation job (`ffprobe` under limits, magic bytes, EXIF); Caddy or Cloudflare Tunnel in front of the API; Redis and Postgres bound to `127.0.0.1`; keys in environment variables; demo token rotation after the event.
2. **CAD service and routes:** accept 4.6, run the job on the CPU worker, return 4.7 with the uniform check schema; store all outputs under `cad/{cad_id}/`.
3. **Template library** (6.4): the templates needed for Scenario B first, then 4-6 more for live "try anything" moments. Each template has a parameter schema with ranges and sensible defaults, and unit tests that build it across the range. Template code is the team's own and runs in the normal worker; no sandbox is needed for the judged path.
4. **Printer profile:** one JSON per printer and material (nozzle, perimeters, minimum wall, overhang limit, small-feature limit, radial clearance per fit type). Checks read their thresholds from it.
5. **Fit check against the scan:** signed radial clearance from P3's fitted primitives by `feature_id` decides pass/fail when the input dimension is caliper-sourced or user-confirmed, advisory when it is scan or snapped; unsigned surface separation against the scanned part mesh is reported as advisory only (6.4). Requests whose revisions do not match the current part are rejected (409), and results are marked `stale` when the part changes afterwards.
6. **Printability checks:** required `watertight` and `fits_build_volume`; advisory wall thickness (`trimesh.proximity.thickness`), overhang with auto-orientation, small features (6.4).
7. **Strength heuristic** for brackets and hooks, labelled as an estimate and never a gate.
8. **Exports and print settings:** STL, STEP, source, preview render, settings and volume-based estimates labelled as such.
9. **Tolerance coupon and physical proof:** print a coupon with 0.05/0.10/0.15/0.20/0.30 mm radial clearance holes around a 6.0 mm D-shaft on the actual printer, set the profile's clearance per fit type from what fits (a knob needs the `snug` value), then print the Scenario B part before judging and bring both the printed part and the original. A part that fits on stage is the strongest demo moment.
10. **Stretch, only after 1-9 work:** codegen mode in the disposable container (3.3) with the error-feedback loop; interference volume via mesh closing and `manifold3d`; slicer CLI for real time estimates.

**Done when (judged):** given the hero part's preferred dimensions, the service produces a template part that passes the required checks, renders in the viewer at the right place, and the printed part actually fits the real object; uploads and the API work over HTTPS behind the demo token.

**Fallbacks:** print the demo part in advance; hide advisory checks that misbehave.

---

## 8. Integration rules
- **Contracts first:** section 4 models live in one place; changing a contract requires telling the people who consume it.
- **Fixtures always work:** every route has a mock mode (an environment flag, owned by P4) that serves fixtures, including a recorded hero conversation with its cards and tool events, with no external call. It exists for two reasons: the UI must never be blocked by a slow pipeline, and it is the basis of the **offline demo build** (section 9): the built frontend plus the API in mock mode running on the presenter laptop at `localhost`, with the hero assets in the local `./data` store and every JavaScript and font dependency bundled rather than fetched from a CDN. `localhost` is a secure context, so camera and HTTPS rules do not get in the way. Keep the fixtures current with the contracts.
- **One repo:** `web/`, `backend/{app,contracts,recon,perception,agent,cad}/`, `fixtures/`, `scripts/`, `profiles/` (printer profiles). Each person owns their folder and reviews changes to it; `backend/app/` and `backend/contracts/` are P4's, storage and deployment files are P5's.
- **End-to-end check:** at regular points, run the full flow on a hero scan and fix whatever broke, before adding new features.
- **Feature freeze** well before judging; the remaining time goes to the demo path only.

## 9. Demo script (about 3 minutes)
Everything the judges see comes from the cached hero scans (1.4). The live capture is a visual only.
1. Problem statement (15 s): "Something breaks, and you don't know what it's called, how it works, or what to buy."
2. Live capture (30 s, **only if the live path was rehearsed successfully the day before; otherwise skip to step 3**): film the P-trap with the board flat beside it for about 20 s, upload, and let the status bar show the first stage events ticking. Do not wait for it: open the cached hero scan of the same object straight away and continue; if the live job happens to finish during the demo, mention it, otherwise nothing depends on it.
3. Understand (30 s): rotate the model (the splat if the licence gate cleared, otherwise the mesh), toggle to parts, exploded view, measure the pipe (shows about 38 mm; the parts panel shows the scan value, the snapped 1-1/2" and the caliper value).
4. Fix (45 s): "It's leaking under here." Agent highlights the slip joint, gives the diagnosis, steps, and a Parts-to-Buy card with the correct washer.
5. Modify and print (45 s): Scenario B object (cached). "Make me a replacement knob." The agent asks to confirm the shaft size, the presenter types the caliper value, the template CAD part appears in the viewer with its required checks passing, download STL. Hold up the printed part and fit it.
6. Close (15 s): accuracy numbers from P3's evaluation (scan versus caliper, false-snap rate), what's next.

If the venue network fails, the cloud API and S3 are unreachable too, so switching the remote API to mock mode would change nothing. The fallback is the **offline demo build** (section 8): frontend and mock-mode API already running on the presenter laptop at `localhost` with the hero assets on local disk, started before judging and rehearsed with Wi-Fi turned off; the presenter switches the browser tab, not a server.

## 10. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Feed-forward geometry is poor on the hero object (shiny, white, textureless plumbing) | Choose matte demo objects; add tape markers for texture; good lighting; compare DA3 and MapAnything on day one; cached hero scans are the judged path. |
| The only splat-capable checkpoints are CC BY-NC | Licence gate on day zero (1.5); if not cleared, MapAnything (Apache) plus the vertex-coloured mesh, and the splat leaves the MVP table. |
| Feed-forward splat looks worse than judges expect | Show the mesh as the main view and the splat as a toggle; AnySplat as the alternative (scaled through the same board step); no optimization fallback by design. |
| Splat, mesh, masks and CAD end up mirrored, rotated or 1000x off in the viewer | One scan frame in millimetres, one `scan_to_web` transform on a parent group, `.spz` packed without coordinate conversion, explicit camera contract, quaternion order and gaussian transform rules (3.1); Open3D fed metres and converted back at the boundary, with a golden TSDF test (6.1 step 6); day-one golden check with an asymmetric 100 mm test object. |
| Measurement accuracy below what the CAD part needs | Scan, snapped and caliper values kept side by side with a preferred source; snapping only for tubular sizes within 2.0 mm with 6 mm separation; snapped values treated like scan values for anything physical; the agent asks the user to confirm mating dimensions; tolerance coupon on the real printer; caliper check of hero values before judging; CAD results marked stale when inputs change. |
| Wrong scale makes CAD parts not fit | Printed ChArUco board with multi-view triangulation and a robust fit; residual and corner-count flags; learned scale and manual `/scale` as fallbacks. |
| Noisy part segmentation | Verified prompts for hero scans; user merge; caliper entry for the dimensions that matter; human check of hero results. |
| LLM invents sizes or unsafe advice | Tools return measured values with their source; "estimated" wording rule; confirm-before-printing; guardrails; evaluation set. |
| Someone finds the public API during judging | Demo token delivered in a URL fragment and sent only as a bearer header (also on both SSE streams); per-token and global limits; LLM spend cap; contained media processing on the worker; databases bound to localhost. |
| Browser splat rendering slow on laptops | Gaussian cap of 1.5 M, `.spz` compression, mesh-only fallback. |
| GPU out of memory | One GPU job at a time; gaussians and depth written to disk and every GPU tensor released, with a memory assertion, before SAM 2 loads; view cap pinned from the day-one measurement (start at 48); `infer_gs=False` for mesh only; SAM 2 base-plus instead of large; 48 GB instance if budget allows. |
| Proxy closes the SSE stream during the minutes-long parts stage | 20 s keep-alive comments on both streams; the client reconnects and re-reads `GET /scans/{id}`. |
| Scope: five people, many experimental integrations | Judged MVP in 1.4 with per-person "done when (judged)" criteria on cached hero artifacts; platform work split across P4 and P5 so P2 holds only the GPU pipeline; everything else is stretch and is not started until the hero path works end to end. |
| Venue network fails (web host, API, S3 and LLM all unreachable) | The offline demo build on the presenter laptop: frontend plus mock-mode API at `localhost`, hero assets on local disk, all JS and fonts bundled; started before judging and rehearsed with Wi-Fi off (sections 8 and 9). |

## 11. Stretch goals
- Assembly/disassembly animation generated from the scene graph and repair steps.
- AR view of the repair steps on a phone (WebXR).
- Live product search across several stores with price comparison.
- Thread pitch estimation from close-up photos.
- Export a full parametric reconstruction of the object (all parts as CAD).
