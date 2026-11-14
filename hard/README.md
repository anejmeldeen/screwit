# X8: does the real pipeline work on hard objects?

The only real-model test before this was a cube and a cylinder on a plain background. This
workstream renders *hard* scenes whose geometry is known exactly, runs the **real** reconstruction
pipeline on them (Depth Anything 3 nested on CPU, the fake part labeller because no Claude
credentials exist), and scores every stage against the ground truth. `RESULTS.md` has the numbers
and a per-scene verdict.

A failure found here is the point of the work, not a defect in it.

## What runs

- `scripts/make_hard_scenes.py` renders each scene and writes exact ground truth.
- `scripts/hard_eval.py` runs the live pipeline on a scene's video and scores it.
- `backend/tests/hard/` are fast tests of the scorer and generator (no model weights).
- Large outputs (videos, ground truth, run folders, model caches) live under
  `/local/home/brancli/screwit-hard/`, never in git. Only the small `*.json` score summaries here
  and this doc + `RESULTS.md` are committed.

## Scenes

| scene | what it is | object | camera | why it is hard |
|---|---|---|---|---|
| `gso_real_mug` | real GSO mug, on a wood tabletop beside the board | mug + handle | orbit | small, glossy, thin handle |
| `gso_real_shoe` | real GSO bootie | shoe | orbit | organic shape, no flat faces |
| `gso_real_glass` | real GSO plant saucer | glossy saucer | orbit | shallow, glossy, low relief |
| `gso_real_toy` | real GSO toy (kitchen-tool-like) | moulded plastic toy | orbit | concavities, thin parts |
| `under_sink` | procedural cabinet: basin underside, tailpiece, P-trap, two slip nuts, slip-joint washer, shutoff valve + supply line, a box and a bottle | the plumbing | hand-held arc in front | chrome highlights that move with the camera, white plastic, dim uneven light, clutter |
| `under_sink_portrait` | the same under-sink scene shot upright in portrait | the plumbing | arc | phone portrait storage (sideways samples + 90° display rotation) |
| `knob_panel` | a knob on a panel with a D-shaft behind it | knob + D-shaft | arc | a thin feature (the D-shaft flat) to measure |

The primitive `gso_mug`, `gso_shoe`, `gso_thin_tool`, `gso_glossy_bottle` scenes exist as
exact-dimension stand-ins for when Gazebo Fuel is unreachable; the `--fetch-gso` scenes above use
real geometry and are what the results report.

## Ground truth

Everything is in the **scan frame** the board defines (`screwit.recon.scale.board_scan_points`):
origin at the printed origin corner, X along the arrow, Y up the page, Z up out of the paper. Per
scene, under `<out>/<scene>/truth/`:

- `meshes/<object>.ply` — every object mesh (millimetres).
- `cameras.json` — per-frame `world_to_camera` (row-major, OpenCV axes) and `K`, in the **displayed**
  image frame (for the portrait scene the 90° display rotation is already applied).
- `masks/<object>/NNNN.png` and `parts/<part>/NNNN.png` — per-frame object and part label masks.
- `scene.json` — key dimensions (tube OD, nut width, knob/shaft diameter, mug height, …), materials,
  the part list, the frame/portrait metadata, and a ChArUco `board_detection` count (how many
  rendered frames a real detector sees the board in).

The ground truth is self-validating: a real ChArUco detector run on the rendered frames reprojects
its detected corners onto these cameras to **under 1 px mean** (checked in
`backend/tests/hard/test_make_hard_scenes.py` and recorded per scene), so the cameras, board
placement and object meshes are mutually consistent.

## Rendering method

A small numpy software rasterizer (no GPU, no OpenGL, no display): triangle meshes with a z-buffer,
per-face Phong shading (ambient + diffuse + specular point lights), and perspective-correct bilinear
texture sampling for the board. Shiny surfaces (chrome, glossy white) use a high specular exponent so
highlights move with the camera, because specular highlights are a known depth-model weakness.
Videos are encoded with the `imageio-ffmpeg` binary as **H.264, 1920×1080, 30 fps, 20 s**, so the
real decode path (`screwit.recon.frames.extract_video_frames`) runs. The portrait scene stores
landscape samples plus a 90° display matrix (as phones do) and is verified to decode upright through
that extractor before use.

## Licences and attribution

The real objects are **Google Scanned Objects**, downloaded from Gazebo Fuel, each licensed
**CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Attribution:

| scene | model | source |
|---|---|---|
| `gso_real_mug` | *Cole Hardware Mug Classic Blue* | Google Research, Gazebo Fuel, CC BY 4.0 |
| `gso_real_shoe` | *Womens Teva Capistrano Bootie* | Google Research, Gazebo Fuel, CC BY 4.0 |
| `gso_real_glass` | *Ecoforms Plant Saucer SQ8COR* | Google Research, Gazebo Fuel, CC BY 4.0 |
| `gso_real_toy` | *Vtech Roll & Learn Turtle* | Google Research, Gazebo Fuel, CC BY 4.0 |

Model URL pattern: `https://fuel.gazebosim.org/1.0/GoogleResearch/models/<name>`. The downloaded OBJ
geometry is used as-is (metres → millimetres, scaled to a chosen real size); the flat rasterizer does
not sample the OBJ texture, so each model is shaded with a representative material (noted in its
`scene.json`). The ChArUco board, cabinet, tabletop and plumbing are generated here and are not GSO.

## How to rerun

All commands run from the worktree root
`/local/home/brancli/.kiro/crew/workspace/screw-it-wt/hard`. The model runs are CPU-only and slow
(~45–75 s of geometry and ~2–4 min of TSDF meshing per scene); run one at a time, pinned off the
cores another builder uses:

```bash
VENV=/local/home/brancli/.kiro/crew/workspace/screw-it/backend/.venv/bin/python
CPUVENV=/local/home/brancli/.kiro/crew/workspace/screw-it-wt/cpu/cpu-verify/.venv/bin/python
CPU=/local/home/brancli/.kiro/crew/workspace/screw-it-wt/cpu/cpu-verify

# 1. Render scenes + ground truth (fast, no weights). Real GSO needs Gazebo Fuel reachable:
nice -n 19 taskset -c 16-31 env OMP_NUM_THREADS=16 PYTHONPATH=backend \
  $VENV scripts/make_hard_scenes.py --fetch-gso --out /local/home/brancli/screwit-hard
nice -n 19 taskset -c 16-31 env OMP_NUM_THREADS=16 PYTHONPATH=backend \
  $VENV scripts/make_hard_scenes.py --all --out /local/home/brancli/screwit-hard

# 2. Run the real pipeline and score (uses the cpu-verify venv that has the real weights):
nice -n 19 taskset -c 16-31 env OMP_NUM_THREADS=16 SCREWIT_DEVICE=cpu HF_HOME=$CPU/.cache/hf \
  PYTHONPATH=backend $CPUVENV scripts/hard_eval.py --all \
  --out /local/home/brancli/screwit-hard --summary-dir hard

# Rescore from a cached run without rerunning the model:
...  $CPUVENV scripts/hard_eval.py --scene gso_real_mug --out /local/home/brancli/screwit-hard --reuse

# 3. Fast tests + lint (shared venv):
PYTHONPATH=backend $VENV -m pytest backend/tests/hard -q
$VENV -m ruff check scripts/make_hard_scenes.py scripts/hard_eval.py backend/tests/hard \
  --config backend/pyproject.toml
```

The geometry result is pickled per scene (`<out>/<scene>/run/geometry.pkl`), so re-running the eval
reuses the DA3 pass and only redoes the cheap stages; `--reuse` skips the pipeline entirely and
rescans the persisted artifacts.

## Scoring (what each number means)

Per scene, `hard_eval.py` writes `<out>/<scene>/run/score.json` and copies a summary here:

- **Board** — method (`aruco` = found), corner views, residual mm, tilt.
- **Scale** — each key dimension measured on the scan mesh vs truth (mm and %). The CPU live path
  makes a single part per object, so this is a coarse overall-size check; per-part numbers are under
  *Parts*.
- **Mask** — IoU of the scan mesh's silhouette against the true object silhouette, rendered through
  the pipeline's **own recovered camera** per keyframe (mean and worst). The pipeline does not
  persist its per-frame object masks, so this silhouette IoU is the mask-quality proxy.
- **Mesh** — accuracy (scan→true) and completeness (true→scan) distance percentiles in the scan
  frame, share of the true surface within 2 mm and 5 mm, triangle count, broken faces. Reported over
  the whole scan mesh **and** over only the scan surface inside the true object's bbox grown 15 mm
  (`object_region`), which separates reconstruction quality from mask leakage.
- **Splat** — gaussian head vs depth fallback, gaussian count.
- **Parts** — parts found, feature types, each measured dimension vs truth (fake labeller).
- **Time** — wall time per stage.
- One composite image per scene (`run/composite.png`): a decoded frame with the scan silhouette
  overlaid, beside top views of the scan mesh and the true mesh.
