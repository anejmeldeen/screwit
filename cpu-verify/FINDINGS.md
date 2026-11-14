# X4 findings: real models on the CPU

Host: 32-core Xeon 8488C, no GPU, torch 2.5.1+cpu. Input: 8 real 960x540 frames of a handheld clip (small baselines; see README). Wall times are single runs, 32 threads, weights already cached (first download excluded). Peak RSS is process-wide (`ru_maxrss`).
Raw numbers: `out/*.json`. Convention checks use mean absolute grey difference after warping view A into view B with the model's own depth and pose (lower is better; "identity" is a no-motion control).

## Summary

- All four model families run on CPU with the real weights. No blocker, no upstream source patched.
- CPU is usable as a fallback for small view counts: DA3 nested with the Gaussian head takes about 7 s per view at 8 views (55 s total); MapAnything 3 s per view; SAM 2 video propagation about 4.5 s per frame (base+).
- Confirmed: DA3 returns world-to-camera extrinsics; MapAnything returns camera-to-world; grid 504x280 for 1920x1080 and 960x540 input; the gs_ply layout and `[w,x,y,z]` quaternion order.
- **Not confirmed: the SPEC 3.1 claim that the nested model's gaussians are left unscaled.** Multiplying the gaussian means by `scale_factor` made them match the depth maps worse, not better (details below).
- The real DA3 and MapAnything API differs from what `recon/geometry.py` assumes in several places (section 5), so those two adapters would fail on first GPU run.

## 1. Depth Anything 3

| # | Assumed (SPEC / AGENTS / adapter) | Observed | Result |
|---|---|---|---|
| 1 | `import depth_anything_3 as da3; da3.DepthAnything3` | `depth_anything_3` exports nothing at top level; `from depth_anything_3.api import DepthAnything3` | **fail** (adapter) |
| 2 | Prediction fields depth, conf, extrinsics, intrinsics, processed_images | `depth (N,H,W) f32`, `conf (N,H,W)`, `extrinsics (N,3,4) f32`, `intrinsics (N,3,3)`, `processed_images (N,H,W,3) uint8`, `sky` (None), `is_metric`, `scale_factor`, `gaussians`, `aux` (`{}`) | pass; note extrinsics are **3x4**, not 4x4 (`specs.py` comment says 4x4) |
| 3 | Extrinsics are world-to-camera, `C = -R^T t` | Photo error with w2c hypothesis 12.8 vs 24.2 for c2w and 20.5 for identity (DA3-LARGE); nested 14.4 / 25.4 / 21.9; camera 0 is near identity, centres from `-R^T t` form a smooth path | pass |
| 4 | 1920x1080 input gives 504x280 grid | 1920x1080 -> depth/conf/image `(280,504)`; 960x540 -> `(280,504)`; K has different fx and fy (432.5 vs 428.1), principal point (252,140) | pass |
| 5 | `prediction.scale_factor`, `is_metric` for nested | nested: `is_metric=1`, `scale_factor=3.177` (8 views) and 3.192 (second run, `ref_view_strategy="middle"`; value depends on the run); DA3-LARGE: `is_metric=0`, `scale_factor=None`. Nested depth range 2.1-3.9 vs 0.56-1.39 for DA3-LARGE | pass |
| 6 | `prediction.confidence`, `prediction.metric_scale` (adapter) | Neither exists. Field is `conf`; the metric factor is `scale_factor` | **fail** (adapter) |
| 7 | Nested quirk: gaussians are unscaled, so means x `scale_factor` and log-scale + ln(`scale_factor`) | See "Quirk check" below: not supported by this scene | **not confirmed** |
| 8 | `infer_gs=True` runs on CPU | Gaussian head runs on CPU (only the python-only `gsplat` build was needed for import). Nested: 1,128,960 gaussians for 8 views at 504x280 (one per pixel, slab k = view k); `prediction.gaussians` holds them, `aux` is empty. DA3-LARGE has no head: `gaussians=None` and the exporter raises `AttributeError` | pass (nested); DA3-LARGE as spec says |
| 9 | `export_format="mini_npz-gs_ply"` + `export_dir` writes the PLY | Writes `mini_npz/results.npz` and `gs_ply/0000.ply`, then the API **appends `gs_video`** whenever `infer_gs` is set, which fails on CPU in `render_3dgs` (`RuntimeError: ... written-to tensor refers to a single memory location`, `gs_renderer.py:113`; also gsplat has no CUDA backend here). The PLY is already on disk when it raises | **partial**: PLY written; call raises on CPU; on GPU the video render will also run and cost time and memory |
| 10 | PLY fields | `x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3`, binary, DC-only SH, all float32. 223,672 gaussians for `gs_views_interval=4` over 8 views (2 views exported, then pruned by depth percentile and borders). PLY means, opacity (inverse-sigmoid), `scale_*` (log) and rotations equal the in-memory tensors; `rot_0` is w (`rot_i` equals memory column i, in-memory `Gaussians.rotations` is documented `wxyz`), unit norm | pass (`[w,x,y,z]`, log scales, inverse-sigmoid opacity as SPEC 3.1) |
| 11 | In-memory scales | `gaussians.scales` are linear (`scales_std`, 3e-9..3.6), the PLY stores `log` of them | pass |

Quirk check (`scripts/analyze_da3_gs*.py`, nested, 8 views). The gaussian tensor is 8 slabs of 280x504 and slab k lines up with view k. For each slab I compared the camera-frame z of its means (using that view's extrinsic) with that view's depth map:

- Using extrinsics and depth as returned (already multiplied by `scale_factor`): median depth/gaussian-z ratio 0.62-0.79 per slab, log-ratio std 0.45-0.9.
- Using the "gaussians unscaled" hypothesis (divide extrinsic translation and depth by `scale_factor`): worse for every slab.
- After applying the SPEC correction (means x `scale_factor`) on view 0: median ratio 0.20 vs 0.64 without it.

So the gaussians are not at 1/`scale_factor` of the depth frame, and the proposed correction moves them away from the depth maps. They are also not pixel-consistent with the depth head (wide spread), so this is one scene with small baselines and is not conclusive either way. Whatever P2 decides must be checked on a real scan (gaussian vs TSDF mesh alignment); do not trust the assumption blindly.

Timings (8 views, 504x280):

| Model | Load | Inference | Peak RSS |
|---|---|---|---|
| DA3-LARGE | 2.3 s | 7.1 s (0.9 s/view) | 4.8 GB |
| DA3-LARGE, 2 views from 1920x1080 | 3.7 s | 4.8 s | 3.5 GB |
| DA3NESTED-GIANT-LARGE-1.1 | 16 s | 29 s (3.7 s/view) | 13.1 GB |
| DA3NESTED-GIANT-LARGE-1.1, `infer_gs=True` | 12.5 s | 55 s (6.9 s/view) | 13.4 GB |

## 2. MapAnything (`facebook/map-anything-apache`)

| # | Assumed | Observed | Result |
|---|---|---|---|
| 1 | `model.infer(list_of_numpy_images, memory_efficient_inference=True)` | Needs `views = mapanything.utils.image.load_images(paths)` (list of dicts with `img, true_shape, idx, instance, data_norm_type`). `infer(views, memory_efficient_inference=True, use_amp=False)` ran on CPU | **fail** (adapter passes numpy arrays) |
| 2 | Result is one dict of stacked arrays (`preds["camera_poses"]`, `["depth"]`, `["confidence"]`) | Returns a **list with one dict per view**, each with leading batch dim 1: `pts3d, pts3d_cam, ray_directions, depth_along_ray, depth_z (1,H,W,1), cam_trans, cam_quats, metric_scaling_factor (1,1), conf (1,H,W), non_ambiguous_mask(+_logits), img_no_norm, intrinsics (1,3,3), camera_poses (1,4,4), mask (1,H,W,1)`. No `depth` and no `confidence` keys | **fail** (adapter) |
| 3 | `camera_poses` are camera-to-world, OpenCV; adapter inverts | Using `inv(camera_poses)` as world-to-camera: photo error 13.7 vs 23.0 for using it directly vs 19.1 identity; view 0 pose is identity, translations form a smooth trajectory | pass |
| 4 | Model grid 504x280 | 518x294 (long side 518, multiple of 14). K0 fx 211.9, fy 213.6, cx 256.1, cy 147.7 | **differs** from DA3 grid; `frames.json`/K must carry the grid per model |
| 5 | `metric_scaling_factor` | Per view `(1,1)`, identical for all views (1.0766); depth range 0-2.74 (0 = masked pixels) | pass |
| 6 | No gaussians | Right: no gaussian head | pass |

Timing: load 30 s, inference 25 s for 8 views (3.2 s/view, `use_amp=False`), peak RSS 9.6 GB.

## 3. SAM 2 (`sam2.1_hiera_base_plus`, config `configs/sam2.1/sam2.1_hiera_b+.yaml`)

| # | Assumed | Observed | Result |
|---|---|---|---|
| 1 | Image predictor with a box and a negative point | `SAM2ImagePredictor(build_sam2(cfg, ckpt, device="cpu"))`; `set_image(np.uint8 HxWx3)`; `predict(point_coords=(1,2), point_labels=[0], box=(1,4), multimask_output=False)` returns `masks (1,H,W) float32`, `scores (1,)`, `logits (1,256,256)`; the negative point fell outside the mask | pass |
| 2 | `build_sam2_video_predictor(cfg, ckpt)` without a device | Default `device="cuda"` in both builders, so this fails on a CPU host unless `device="cpu"` (or CUDA) is passed | **fail** on CPU (adapter) |
| 3 | `init_state(video_dir)`; `add_new_points_or_box(state, frame_idx=, obj_id=, points=, labels=, box=)`; `propagate_in_video(state)` | All as assumed; `init_state(video_path=dir)` needs a directory of JPEGs named with integers (`00000.jpg`). `add_new_points_or_box` returns `(frame_idx, [obj_id], logits (1,1,H,W))`; `propagate_in_video` yields `(frame_idx, obj_ids, logits (n_obj,1,H,W))`; mask is `logits > 0` | pass |
| 4 | Propagation over 8 frames | Ran; mask area stable (2.82%-2.91%) across the 8 frames | pass |
| 5 | `SAM2AutomaticMaskGenerator(build_sam2(...))` | Works (`points_per_side=16`: 20 masks, keys `segmentation, area, bbox, predicted_iou, point_coords, stability_score, crop_box`); adapter default 32 points per side is about 4x slower | pass |

Timings: load 3.2 s, `set_image` 7.7 s, box predict 0.5 s; video load 4.8 s, `init_state` (8 frames) 12.2 s, prompt 1.2 s, propagate 8 frames 35.7 s (4.5 s/frame); automatic masks 14 s at 16 points per side. Peak RSS 4.3 GB. The CPU run used the default float32 path with no autocast.

## 4. Grounding DINO

| # | Assumed | Observed | Result |
|---|---|---|---|
| 1 | HF `IDEA-Research/grounding-dino-tiny` with `post_process_grounded_object_detection(outputs, input_ids, box_threshold=, text_threshold=, target_sizes=)` | Works on CPU (load 4.0 s, inference 8.1 s first call). Result keys `scores, boxes, text_labels, labels`; boxes are xyxy in pixels of `target_sizes`; with transformers 4.49.0 `labels` are strings, so the adapter's `results["labels"]` works. In later transformers (>= 4.51), `labels` are expected to be ints and `text_labels` holds the strings (not run here); also **transformers 5.x refuses to load with torch 2.5.1** (needs >= 2.6). Pin transformers and use `text_labels` | pass with pin |
| 2 | Prompt `"bed. pillow."` (lower case, period-separated) | The processor accepts `"a bed. a pillow."`; labels come back as the phrase text (`"a bed"`) | pass |
| 3 | Grounded-SAM-2 repo path on CPU | `groundingdino` from the repo builds without CUDA (`pip install --no-build-isolation -e grounding_dino`) and runs with `load_model(config, ckpt, device="cpu")` and `predict(model, img, "bed. pillow.", 0.35, 0.25, device="cpu")`: load 3.8 s, infer 2.8 s, returns normalised cxcywh boxes, scores and phrases. Needs the repo root on `sys.path` (`grounding_dino.groundingdino...` imports), plus `supervision` and `yapf` | pass; HF port is simpler and is what the adapter uses |
| 4 | Box into SAM 2 | A Grounding DINO box passed as `box=` to SAM 2 produced a mask (score 0.98) | pass |

## 5. Adapter mismatches (read-only review)

Read: `recon/backend/screwit/recon/geometry.py`, `gaussians.py` (`read_ply`, `apply_scale_factor`), `perception/backend/screwit/perception/adapters.py`. Not edited.

`recon/geometry.py` `DepthAnything3Adapter`:
1. Import: use `from depth_anything_3.api import DepthAnything3`.
2. `.to(device)` is missing; `from_pretrained` returns a CPU/default-device model (README calls `.to(device)`).
3. `prediction.confidence` -> `prediction.conf`; `prediction.metric_scale` does not exist (use `scale_factor`; it is `None` for non-nested models, so `float(...)` will raise).
4. `extrinsics` is `(N,3,4)`; `GeometryResult.extrinsics` is `(N,4,4)`, so append the `[0,0,0,1]` row.
5. Depth is not millimetres: units are model units (metric metres for the nested model, arbitrary for others); the field is named `depth_mm`.
6. The model grid is 504x280 for a 16:9 input, produced by the model; use `prediction.depth.shape[1:]` and `processed_images`, not the input size (the adapter does take `h, w` from depth, which is right).
7. `export_format="mini_npz-gs_ply"` makes the API append `gs_video` (renders a video, fails on CPU, slow on GPU). Call the exporter pieces yourself or catch the failure after the PLY exists; prefer reading `prediction.gaussians` directly (means, scales, rotations wxyz, harmonics, opacities as tensors, linear scales and probability opacity) rather than the pruned, subsampled PLY.
8. `read_ply`: reads float-only binary PLY with `rot_0 = w`, log scales, inverse-sigmoid opacity: consistent with the file DA3 wrote (the file also has `nx ny nz`, which it skips). Pass.
9. `apply_scale_factor`: see quirk check; unverified and possibly wrong direction.

`MapAnythingAdapter`:
10. Must call `load_images(paths)` and pass its list of view dicts; `infer` returns a list of per-view dicts, not a dict of stacked arrays. Keys: `camera_poses (1,4,4)`, `intrinsics (1,3,3)`, `depth_z (1,H,W,1)`, `conf (1,H,W)`, `metric_scaling_factor (1,1)`. The adapter's `preds["depth"]`, `preds["confidence"]`, `preds.get(...)` calls do not match.
11. Output grid is 518x294, not 504x280.
12. `memory_efficient_inference=True` is accepted; on CPU use `use_amp=False` (the README default is bf16 AMP).

`perception/adapters.py`:
13. `build_sam2_video_predictor(cfg, ckpt)` and `build_sam2(...)` default to `device="cuda"`; pass a device.
14. Calls `init_state(video_dir)` (positional), `add_new_points_or_box(state, frame_idx=, obj_id=, points=, labels=, box=)`, `propagate_in_video(state)`: match. `labels` should be an int32 array. `mask_logits[0] > 0` then `.squeeze()` gives `(H,W)`, fine for one object. Video directory must hold integer-named JPEGs.
15. `GroundingDinoAdapter`: `results["labels"]` are strings only in transformers 4.x up to the version tested here (4.49.0); use `text_labels`. Needs transformers < 5 with torch < 2.6.
16. `Sam2AutomaticMasksAdapter` defaults to 32 points per side (not timed; 16 points took 14 s, so expect several times that); `SAM2AutomaticMaskGenerator(model)` mask dicts match the assumption (`segmentation`).

## 6. Is CPU inference usable as a no-GPU fallback?

For the demo, only for a reduced view count, not the 48-64 views of SPEC 6.1.

Scaling the measured per-view costs linearly (attention cost grows faster than linear, so treat these as lower bounds): 16 views with the nested model plus Gaussian head about 2 minutes; 48 views about 6 minutes or more, with memory not measured beyond 8 views (13.4 GB at 8). DA3-Large or MapAnything for geometry at 16 views is about 15-50 s. SAM 2 video propagation is about 4.5 s per frame (8 frames: 36 s; 48 frames about 3.6 min). Grounding DINO is 3-8 s per call. A 12-16 view CPU run of the full chain (geometry, one SAM 2 propagation, a few Grounding DINO calls) fits in roughly 3-5 minutes on this 32-core host; on a typical laptop expect several times longer. Untested: 48 views, bf16 on CPU, quantisation, and a dev machine with fewer cores.

Not verified here: the actual accuracy of gaussians against a TSDF mesh on a ChArUco scan (no scan available), GPU behaviour, `DA3-GIANT-1.1` (not run), SAM 2 large, and anything with more than 8 views.
