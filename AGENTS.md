# Screw It: rules for everyone writing code here

`SPEC.md` is the design. This file records how the repo is built and the decisions that override the spec.

## Layout and ownership

| Path | Owner | Notes |
|---|---|---|
| `backend/screwit/contracts/`, `backend/screwit/interfaces.py`, `backend/screwit/llm.py`, `fixtures/*.json`, `fixtures/scans/*/*.json`, `fixtures/schema/`, `profiles/`, `backend/pyproject.toml`, `AGENTS.md`, `SPEC.md` | integration lead | Read-only for workstreams. Need a change? Say so in your report; work around it locally meanwhile. |
| `web/` | P1 web | |
| `backend/screwit/recon/`, `backend/tests/recon/`, `scripts/make_fixture_splats.py`, `fixtures/scans/*/splat/` | P2 recon | |
| `backend/screwit/perception/`, `backend/tests/perception/`, `scripts/make_fixture_assets.py`, `fixtures/scans/*/{parts,mesh,renders,geometry}/` | P3 perception | |
| `backend/screwit/app/`, `backend/screwit/agent/`, `backend/tests/app/`, `backend/tests/agent/`, `catalogue/` | P4 platform and agent | |
| `backend/screwit/cad/`, `backend/screwit/storage/`, `backend/tests/cad/`, `backend/tests/storage/`, `deploy/`, `scripts/make_fixture_cad.py`, `fixtures/scans/*/cad/` | P5 CAD, storage, deploy | |
| `gpu/` | X1 GPU kit | |
| `scripts/make_test_media.py`, `fixtures/media/`, `backend/tests/media/` | X2 test media | |
| `e2e/`, `backend/tests/integration/` | X3 end-to-end tests | |
| `cpu-verify/` | X4 real models on CPU | |
| `backend/screwit/optimize/`, `backend/tests/optimize/`, `scripts/make_fixture_mobile.py`, `fixtures/scans/*/mobile/` | X5 phone asset optimizer | |
| `perf/` | X6 performance tests | |
| `backend/screwit/codegen/`, `backend/tests/codegen/`, `deploy/codegen-sandbox/` | X7 new shapes (codegen) | |

Never edit a path you do not own. Never run `git checkout`, `git reset`, `git clean` or `git stash` in the main tree (`/local/home/brancli/.kiro/crew/workspace/screw-it`); work only in your own worktree.

## Toolchain

- Python 3.12 virtualenv at `backend/.venv` in the main tree, shared by every worktree. Run code from a worktree with
  `PYTHONPATH=<worktree>/backend /local/home/brancli/.kiro/crew/workspace/screw-it/backend/.venv/bin/python -m pytest <worktree>/backend/tests/<area>`.
  `PYTHONPATH` takes precedence over the editable install, so you test your own copy.
- Need a new Python package? `~/.local/bin/uv pip install -p <main>/backend/.venv/bin/python <pkg>` and list it in your report; the integration lead adds it to `pyproject.toml`.
- No GPU on the dev machine. Depth Anything 3, MapAnything and SAM 2 run only behind adapters; their real calls carry `@pytest.mark.gpu` and are skipped. Every stage must also have a CPU fake so the pipeline is testable end to end.
- `ffmpeg` is not installed system-wide; use the binary from `imageio-ffmpeg` (`imageio_ffmpeg.get_ffmpeg_exe()`) in dev and tests.
- Node 22, npm 10 for `web/`.
- Headless browser (Playwright, Lighthouse): the host lacks `libatomic.so.1`, so export
  `LD_LIBRARY_PATH=/local/home/brancli/.kiro/crew/workspace/screw-it-tools/lib` and
  `PLAYWRIGHT_BROWSERS_PATH=/local/home/brancli/.kiro/crew/workspace/screw-it-tools/ms-playwright`
  (Chromium headless shell 159 is installed there and verified to launch). For Lighthouse, point `CHROME_PATH` at a Chromium binary under that directory.
- `ruff check` must be clean for Python; `tsc --noEmit` and `eslint` for TypeScript.

## Conventions (from SPEC 3.1 and 4)

- Scan frame: right-handed, millimetres, Z up from the support plane. Web frame: three.js Y up, metres, reached only through `scan_to_web` on one parent group.
- Contracts quaternions `[x, y, z, w]`; PLY and 3DGS `[w, x, y, z]`; spz `GaussianCloud` `[x, y, z, w]`. Reorder explicitly at every boundary.
- `cameras.json` stores `world_to_camera`, row-major, OpenCV axes.
- Open3D is fed metres and converted back (SPEC 6.1 step 6). Open3D is pinned at 0.19.0.
- Validate every input and output with the Pydantic contracts. Never return a dict that has not been through a contract model.

## Decisions that override the spec

1. **No self-minted demo token (overrides SPEC 3.3 "Access gate").** Amazon security policy forbids inventing our own authentication. Instead:
   - Both servers bind `127.0.0.1` by default and refuse to start on a non-loopback address. The Next.js server proxies `/api/*` to FastAPI, so the browser talks to one origin.
   - The FastAPI gate (`screwit.app.auth`) accepts a request only if it arrived on a loopback socket. Reaching the demo from elsewhere is done by an SSH local port forward to the presenter's laptop, or an Amazon Tunnels share (Midway-authenticated) when hosted on a Cloud Desktop.
   - A `midway` mode exists as a fail-closed stub that rejects every request with 401 until a real Midway integration is wired; it never admits anyone.
   - Consequence: no bearer header anywhere. The two-client rule stays: the API client calls `/api`, the storage client sends exactly the pre-signed form fields or signed URL and nothing else.
2. **SSE:** chat still uses `fetch` plus `eventsource-parser` (it is a POST). Progress may use the same client for uniformity.
3. **Storage in dev** is local disk (`data/` in the main tree, gitignored) with pre-signed POST semantics emulated by the API; S3 (private, Block Public Access on, presigned POST with `content-length-range`) when deployed.
4. **LLM provider** is chosen by `SCREWIT_LLM` = `openrouter` (Claude through OpenRouter's Anthropic-compatible Messages endpoint `https://openrouter.ai/api`, key from `OPENROUTER_API_KEY`, model ids like `anthropic/claude-sonnet-4.5`), `bedrock` (Claude on Amazon Bedrock via the `anthropic` SDK's `AnthropicBedrock`, AWS credentials from the environment), `anthropic` (API key from `ANTHROPIC_API_KEY`), or `fake` (deterministic, used by tests and mock mode). `SCREWIT_MODEL` is required for every real provider. No credential is ever written to a file in this repo.
5. **Real-model facts override SPEC wherever they differ** (`cpu-verify/FINDINGS.md`, measured on the real weights). In particular SPEC 3.1's "multiply DA3 nested gaussian means by `scale_factor`" was not confirmed: the pipeline instead fits the gaussian-to-depth similarity at runtime from the pixel-aligned gaussians. DA3 extrinsics are `(N,3,4)`; MapAnything's grid is 518x294 and its `infer` returns a list of per-view dicts; SAM 2 builders default to CUDA, so the device is always passed explicitly.
6. **CPU fallback:** `SCREWIT_DEVICE=cpu` runs the real models on the CPU with at most 16 views (about 3-5 minutes for the full chain on a 32-core host, measured at 8 views and scaled).

## The scale board (one definition)

`backend/screwit/recon/board.py` (P2) is the only definition of the printed board: ChArUco, 5 x 7 squares, 30 mm squares, 22.5 mm markers (ratio 0.75), `DICT_4X4_50`, name `charuco_5x7_30mm`. Test media, the printable PDF the web app links, and the detector all use exactly this. The PDF in `web/public/scale-board.pdf` is generated from `board.py` at 100% scale, never drawn by hand.

## The demo object can be anything

The P-trap and the knob are only the two hero fixtures. Code must work for any object:
- Never hard-code plumbing (or any object category) in perception, agent, CAD or UI logic. Part names come from Claude; categories, relations and failure modes are data.
- Snapping tables are data keyed by feature role; an object with no matching table simply gets no snaps.
- The agent prompt and tools are object-agnostic; plumbing knowledge lives only in the parts catalogue and the fixtures.
- UI copy says "object", never "sink" or "trap", outside the demo-scan titles.

## Performance budgets (phones are a first-class target)

Reference phone: a mid-range device from about 2021 (iPhone 12 or Pixel 6 class). Lab proxy: Lighthouse mobile preset and Playwright with 4x CPU throttling and a slow-4G network profile. Lab numbers are a proxy; the final check is on real phones (a test page reports frame times so a person can read them off the device).

| What | Budget |
|---|---|
| Landing page, mobile preset | LCP under 2.5 s; JS under 200 KB gzipped; no three.js or Spark loaded |
| Workspace first paint (shell, panels, status) | under 2 s; viewer code loaded by dynamic import after the shell |
| First 3D view of a hero scan on slow 4G | mesh visible under 5 s; splat may stream in afterwards |
| Viewer frame rate on the reference phone | 30 fps or better while orbiting; render on demand when idle (r3f `frameloop="demand"`), device pixel ratio capped at 2 |
| Phone assets | splat at most 400k gaussians (`splat_mobile_uri`), mesh at most 50k triangles with meshopt or Draco compression (`mesh_mobile_uri`); desktop keeps the full assets. The client picks by device capability and offers a toggle |
| Chat | first token on screen under 2 s after send (mock mode: immediate) |
| Upload | guidance to record 1080p, not 4K (about 30-40 MB for 30 s of HEVC); visible progress; resumable or retryable on failure |
| Server pipeline | unchanged from SPEC: geometry in seconds on a GPU; CPU fallback timings come from X4 |

Every budget gets an automated check where one is possible, and a number in the final report either way.

## Done means

Your area's tests pass, `ruff` (or `tsc`/`eslint`/`next build`) is clean, everything is committed on your branch, and your report lists: what you built, how you verified it (commands and results), what you could not verify and why, any dependency you installed, and any contract change you need.
