# Screw It: hand-off notes

Written 2026-11-14 by Kiro (integration lead) for whoever picks this up. `SPEC.md` is the design;
`AGENTS.md` lists the rules and the decisions that override the spec. This file gives the current
state, how to run everything, what is proven, what is not, and what is in flight.

## What it is

A website: film an object with a phone next to a printed scale board. The server turns the video
into a 3D scan with a feed-forward model (no training, no per-scene optimisation), splits it into
parts with measurements, and a Claude agent answers questions about it: what is wrong, how to fix
it, which parts to buy (with links), and printable CAD for a replacement part (built-in templates,
or new shapes written as CadQuery and run in a locked-down container).

## Repo layout

| Path | What |
|---|---|
| `backend/screwit/contracts/` | Pydantic contracts (`models.py`, `sse.py`); everything in and out goes through them |
| `backend/screwit/app/` | FastAPI app, routes, auth (loopback only), signing, jobs, registry (mock vs live wiring) |
| `backend/screwit/recon/` | frames (ffmpeg), geometry (DA3 / MapAnything adapters + CPU fake), scale (ChArUco board), gaussians/SPZ, TSDF mesh, pipeline |
| `backend/screwit/perception/` | object mask, parts, measurement fits, snapping, labels, scene graph |
| `backend/screwit/agent/` | Claude agent loop, tools, prompt, offline agent (no-key fallback) |
| `backend/screwit/cad/` | CAD templates, placement on the mated feature, checks, preview, `CadRunner` (template and codegen modes) |
| `backend/screwit/codegen/` | new-shape designer: AST pre-check, docker sandbox runner, 3 candidates + repairs + pick |
| `backend/screwit/optimize/` | phone variants: mesh decimate + meshopt (gltfpack), splat thinning to 400k |
| `backend/screwit/storage/` | local disk and S3 storage, upload validation (contained ffmpeg) |
| `web/` | Next.js app: landing, new scan/upload, workspace (viewer, parts, chat), `/api` runtime proxy |
| `fixtures/` | hero scans `scn_trap` (P-trap) and `scn_knob` (knob on a D-shaft), test media, schema |
| `e2e/` | Playwright browser tests (5) |
| `perf/` | Lighthouse + Playwright performance suite (`bash perf/run.sh`) |
| `gpu/` | GPU verification kit (nothing launches cloud resources by itself) |
| `cpu-verify/` | real models on CPU: `FINDINGS.md` has the measured facts |
| `deploy/` | compose, S3 bucket template, media + codegen sandbox Dockerfiles |
| `briefs/` | the task briefs each builder worked from |
| `scripts/` | fixture generators, scale-board PDF, test media, mock chat |

## How to run

```
cd backend && ~/.local/bin/uv venv -p 3.12 .venv && ~/.local/bin/uv pip install -p .venv/bin/python -e '.[cad,geometry,agent,storage,queue,dev,e2e]'
cd web && npm ci && npm run build
cd backend/screwit/optimize/toolchain && npm ci        # gltfpack for phone meshes
cd perf && npm ci                                       # Lighthouse + Playwright
```

Servers (loopback only; they refuse other addresses):

```
cd backend && SCREWIT_MODE=mock SCREWIT_PORT=8000 SCREWIT_DATA_DIR=$(mktemp -d) .venv/bin/python -m screwit.app
cd web && PORT=3000 SCREWIT_API_ORIGIN=http://127.0.0.1:8000 npm run start
```

`SCREWIT_MODE=live` runs the real pipeline (`SCREWIT_DEVICE=cpu` for the 16-view CPU path).
`SCREWIT_LLM=openrouter|bedrock|anthropic|fake` and `SCREWIT_MODEL` pick the LLM (credentials only from
the environment: `OPENROUTER_API_KEY`, AWS credentials, or `ANTHROPIC_API_KEY`). Check a provider with
`PYTHONPATH=backend backend/.venv/bin/python scripts/llm_smoke.py` (text, image + JSON, streamed tool call).
Remote viewing: SSH local port forward or an Amazon Tunnels share. There is no demo token
(self-minted auth is forbidden).

Tests:

```
backend/.venv/bin/python -m pytest -q backend gpu/tests -p no:warnings
PYTHONPATH=backend backend/.venv/bin/python -m pytest backend/tests/integration -q
backend/.venv/bin/ruff check -q backend/screwit backend/tests scripts --config backend/pyproject.toml
cd web && npx tsc --noEmit && npm run -s lint && npx vitest run && npm run -s build
# browser (this host lacks libatomic; the exports point at a user-local copy and Chromium)
export LD_LIBRARY_PATH=<tools>/lib PLAYWRIGHT_BROWSERS_PATH=<tools>/ms-playwright
PYTHONPATH=backend backend/.venv/bin/python -m pytest -c e2e/pytest.ini e2e -q
bash perf/run.sh          # ports 8000 and 3000 must be free; report in perf/out/report.md
```

On the original host `<tools>` is `/local/home/brancli/.kiro/crew/workspace/screw-it-tools`.

Real models on CPU: a separate venv with torch 2.5.1+cpu, depth-anything-3, map-anything, SAM 2,
transformers; see `cpu-verify/README.md` (exact clone commits and install steps) and
`cpu-verify/requirements.lock.txt`. Weights download from Hugging Face on first use
(`depth-anything/DA3NESTED-GIANT-LARGE-1.1`, `facebook/map-anything-apache`,
`IDEA-Research/grounding-dino-tiny`, SAM 2.1 hiera base+ checkpoint). `tools/live_cpu_run.py` in
the hand-off archive drives one video through the live pipeline and caches the geometry result.

## Status at hand-off

Last full results on `main`:

- Backend + GPU kit: 494 passed, 3 skipped; integration 22 passed, 2 skipped (both now pass: hero splats exist); ruff clean.
- Web: tsc, lint, build clean; vitest 39/39.
- Browser e2e: 5/5 (landing, trap workspace and parts, trap repair plan + parts to buy, knob confirm then CAD result).

Phone performance (lab proxy, mock mode):

| Budget | Measured | Status |
|---|---|---|
| Landing LCP < 2.5 s, JS < 200 KB, no three/Spark | 1.98 s, 114 KB, none | pass |
| Workspace shell < 2 s | 1.4 s (mobile preset) | pass |
| Workspace JS | ~400 KB gzipped (was 1377 KB) | Spark loads only on splat view |
| CLS | ~0.02 (was 0.42) | pass |
| TBT | ~685 ms vs 300 target | over: three.js parse when the viewer mounts |
| First mesh on slow 4G < 5 s | 7.4 s (Lighthouse slow-4G devtools values) | over; was 33.5 s |
| Orbit fps | 59.9 p50, software renderer, informational | real phone check pending |
| Chat first token < 2 s | 0.58-0.75 s | pass |
| Phone assets | mesh 972 tris meshopt; splat 20k gaussians | pass |

Real models on CPU, X2's synthetic Android clip (cube 100 mm + cylinder 38.1 mm next to the board):
frames 8 s, Depth Anything 3 nested 55-60 s (16 views), board fit 0.31 mm over 24 views, mesh
about 12-20 s, parts about 34 s; cube measured 103.0 x 101.6 x 101.0 mm (two sides within 2%, one 3%).

## Decisions worth knowing (more in AGENTS.md)

- Feed-forward only (DA3 nested; MapAnything as the Apache-licensed alternative). DA3 Gaussian-head
  checkpoints are CC BY-NC: a licence gate before any commercial use.
- The SPEC's "multiply DA3 gaussians by scale_factor" was not confirmed on real weights. The pipeline
  fits a gaussian-to-depth similarity at runtime; when it fits poorly (it did on the real clip, 49%
  inliers), the splat is built from the depth maps instead so splat and mesh agree.
- Scale board: `backend/screwit/recon/board.py` is the only definition (ChArUco 5x7, 30 mm squares,
  22.5 mm markers, DICT_4X4_50). Board coordinates are right-handed from the printed origin corner;
  the fit refuses a reflection (the earlier sign fix mirrored every scan).
- CPU object mask (no SAM 2 on the live CPU path yet): support plane by RANSAC, accepted only if the
  cameras are above it; keep clusters near the point the cameras orbit; put back the base band.
  Falls back to "all valid depth" and says why in the progress message.
- Poor cylinder fits (inliers < 50% or RMSE > 25% of radius) report box dimensions instead.
- Meshes are decimated with Open3D before pymeshfix (pymeshfix took 11 minutes at 1M triangles).
- Chat: real Claude when credentials exist; otherwise an offline agent grounded in the scan data;
  the old scripted replay only with `SCREWIT_CHAT_REPLAY=1`.
- New shapes: CadQuery code runs only in `docker run --rm --network none --read-only --cap-drop ALL
  --security-opt no-new-privileges --user 65534:65534 --pids-limit 128 --memory 1g --cpus 1`, 30 s,
  killed by name on timeout; only `model.stl`/`model.step` are read back, as regular files under
  64 MB. Mating dimensions must come from calipers or the user (a required check).
- Phones start in mesh view with `mesh_mobile_uri` and never fetch the splat unasked.

## Known problems

1. **iPhone portrait clip: fixed.** The old test clip decoded sideways (an upright landscape render with
   a 90 degree tag). It is now a real portrait video (sideways samples + tag, upright when decoded).
   Real DA3 on CPU: board fit 1.06 mm (was 45.6), tilt 0.37 deg, cube 103.7 x 102.1 x 101.5 mm.
   Two follow-on fixes: the splat is rebuilt from depth when the gaussian head leaves almost nothing in
   the object box, and the phone-mesh bbox guard allows 1% of the diagonal.
2. **First mesh 7.4 s on slow 4G, TBT ~685 ms.** Mostly three.js download and parse. Fix in flight.
3. **Nothing tested on a real object.** The real-model test is a cube and cylinder on a plain
   background. Shiny chrome, white porcelain, thin pipes in shadow and clutter (a real under-sink
   scan) are untested. The hard-object evaluation is in flight.
4. **CPU live path makes one part per object.** The Grounding DINO + SAM 2 part segmenter is verified on
   CPU but not wired into the live path; part names need Claude.
5. **Offline agent's safety notes are generic** ("disconnect power" for a sink).
6. **Not verified at all:** real Claude (provider code is unit-tested with a stub; no key yet), GPU runs, real phone videos, printed fit
   (tolerance coupon, knob).

## In flight at hand-off (branches, not yet merged)

| Branch | Work | Brief / instructions |
|---|---|---|
| `ws/web` | phone first-mesh under 5 s and TBT toward 300 ms: start viewer + mesh downloads with the shell, trim drei/three on the mesh path, measure each change | `briefs/P1-web.md`; measure with `perf/run.sh` |
| `ws/hard` | X8: render hard scenes (Google Scanned Objects, an under-sink cabinet, a knob, a portrait video) with exact truth, run the real CPU chain, score it | `briefs/X8-hard-objects.md` |

Each branch may hold more commits than `main`. Merge only after re-running its tests and checking
that it touched only its own paths (`git diff --name-only $(git merge-base ws/X main) ws/X`).

## What the next person needs from the user

- **Real videos:** board printed at 100% (check a square is 30 mm), laid flat beside the object;
  1080p30, 20-30 s, slow arc or circle; a mug or box, an object with parts, and under a sink with
  the cabinet open; 2-3 caliper or ruler readings per object in a notes file. On the original host
  they go in `/local/home/brancli/screwit-videos/`.
- **Claude via OpenRouter (the user's choice):** `export OPENROUTER_API_KEY=sk-or-...` in the shell
  (never in a file), `SCREWIT_LLM=openrouter`, `SCREWIT_MODEL=anthropic/<model>`; run
  `scripts/llm_smoke.py` first. Scan frames and renders are sent to OpenRouter and the model vendor.
- GPU and 3D-printer tests were ruled out by the user for now.

## Next steps, in order

1. Merge the two in-flight branches once green; rerun everything above.
2. Run the live pipeline on the user's real videos; report scan versus caliper per dimension.
3. Wire the Grounding DINO + SAM 2 part segmenter into the live CPU path (part names from Claude).
4. With credentials: run the agent eval set and three new-shape designs against real Claude.
5. Fix whatever the hard-object evaluation ranks worst for an under-sink scan.
