# X8: does the real pipeline work on hard objects?

Read `briefs/common.md` and `AGENTS.md` first. You are the X8 builder. Your worktree is
`/local/home/brancli/.kiro/crew/workspace/screw-it-wt/hard` on branch `ws/hard` (made from `main`).

## Why

The only real-model test so far is a cube and a cylinder next to the board on a plain background
(`fixtures/media/android_h264.mp4`): board fit 0.31 mm, cube measured 103.0 x 101.6 x 101.0 mm. The
hero P-trap and knob scans are hand-built fixtures and test the software, not reconstruction. The
user asked: how do we know this works for a sink and other real objects? Find out, with numbers,
on renders whose geometry we know exactly. Report failures plainly; a failure found here is the
point of the work.

## What to build (you own these paths)

- `scripts/make_hard_scenes.py`: renders orbit or arc videos of hard scenes with exact ground truth.
- `scripts/hard_eval.py`: runs the real live pipeline on each video and scores it against the truth.
- `hard/README.md` (method, licences and attribution, how to rerun) and `hard/RESULTS.md` (the numbers).
- `backend/tests/hard/`: fast tests of the scorer and the scene generator (no model weights).
- Large outputs (videos, ground truth, run folders, model caches) go under
  `/local/home/brancli/screwit-hard/`, never in git. Keep only small JSON summaries in `hard/`.

Do not edit any other path. If the pipeline needs a fix, write it up in your report (file, line,
cause, the measured effect, a suggested change), and do not make it yourself.

## Scenes (at least these)

1. **Google Scanned Objects** (CC BY 4.0; attribute each model in `hard/README.md`). Download 5 to 8
   varied models from Gazebo Fuel (verified reachable from this host:
   `https://fuel.gazebosim.org/1.0/GoogleResearch/models?per_page=...`, files under
   `.../models/<name>/1/files/meshes/model.obj` plus its texture). Pick a spread: a mug with a handle,
   a shoe, something with thin parts, something glossy or white, a kitchen tool. Each sits on a
   textured tabletop (wood-like, not flat colour) with the board flat beside it, scaled to a known
   real size.
2. **Under a sink** (procedural, the user's main case): cabinet interior (floor, back and side walls),
   the underside of a basin, tailpiece, P-trap J-bend, two slip nuts, a slip-joint washer edge, a
   shutoff valve and supply line, plus clutter (a box, a bottle). Materials: chrome-like (strong
   highlights that move with the camera) and white plastic. Dim, uneven lighting. The camera moves
   in an arc in front of the cabinet, as a real person can, not a full orbit. Board on the cabinet
   floor.
3. **Knob on a panel** with a D-shaft behind it: a thin feature for measurement.
4. A **portrait** variant of one scene, rendered upright in portrait and stored the way phones store
   portrait video (sideways samples plus a 90 degree display rotation). Check it decodes upright
   through `screwit.recon.frames.extract_video_frames` before using it.

Rendering: any CPU renderer that works here (no GPU, no display). Shading must include texture,
diffuse lighting and specular highlights, because shiny surfaces are a known depth-model weakness.
Encode with the imageio-ffmpeg binary as H.264 1080p30, 20 to 30 s, so the real decode path runs.
Ground truth per scene, in the scan frame the board defines (`screwit.recon.scale.board_scan_points`:
origin at the printed origin corner, X along the arrow, Y up the page, Z up): every object mesh,
per-frame world_to_camera and K in the displayed image frame, per-frame object and part label masks,
and a list of key dimensions (for example tube outer diameter, nut width, knob diameter, shaft
diameter, mug height).

## How to run the real models

- CPU venv with the real weights: `/local/home/brancli/.kiro/crew/workspace/screw-it-wt/cpu/cpu-verify/.venv/bin/python`,
  `HF_HOME=/local/home/brancli/.kiro/crew/workspace/screw-it-wt/cpu/cpu-verify/.cache/hf`, `SCREWIT_DEVICE=cpu`
  (16-view cap; Depth Anything 3 nested takes about 55-60 s per scan on this host). SAM 2 checkpoint:
  `.../cpu-verify/.cache/ckpt/sam2.1_hiera_base_plus.pt`. Read `cpu-verify/FINDINGS.md`.
- A working example driver is `/local/home/brancli/.kiro/crew/workspace/screw-it-tools/live_cpu_run.py`
  (runs `ReconPipelineImpl` + `PerceptionImpl(provider="fake")`, caches the geometry result as a
  pickle so later stages can be rerun quickly). Helpers: `gt_compare.py`, `mesh_parts.py`,
  `portrait_look.py` in the same folder. Use them as a starting point; your harness lives in `scripts/`.
- Another builder is measuring web performance on this host at the same time. Run every model job
  under `nice -n 19 taskset -c 16-31` with `OMP_NUM_THREADS=16`, one job at a time.
- No Claude credentials exist: part names use the fake labeller. The CPU live path currently makes a
  single part per object (no real part segmenter is wired); measure what exists and say so. If you
  can drive the existing Grounding DINO + SAM 2 CPU adapters (`screwit.perception.adapters`) from your
  harness with ground-truth part names standing in for Claude's, do it and report it as a separate
  column, clearly labelled.

## Score each scene

- Board: found or not, corner views, residual mm, tilt.
- Scale: error of each key dimension measured on the scan versus truth (mm and %).
- Object mask: IoU against the true object mask per frame (mean, worst).
- Mesh: accuracy and completeness against the true mesh in the scan frame (distance percentiles,
  share of the true surface within 2 mm and 5 mm), triangle count, holes.
- Splat: whether it came from the gaussian head or the depth fallback, gaussian count.
- Parts and measurements: parts found, feature types, each measured dimension versus truth.
- Wall time per stage.
- One image per scene: a decoded frame, the predicted mask over it, and a render of the scanned mesh
  next to the true mesh.

## Done

`hard/RESULTS.md` has one table row per scene with the numbers above and a short, plain verdict per
scene (works / works with caveats / fails, and why). `ruff` is clean, `pytest backend/tests/hard`
passes, everything in your paths is committed on `ws/hard`. Your report lists each pipeline failure
you found with its cause and a suggested fix, ranked by how much it hurts a real under-sink scan.
