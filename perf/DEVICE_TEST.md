# `/perf` device test page — spec for P1 to implement

Lab numbers (Lighthouse mobile preset, Playwright with 4x CPU + slow-4G) are a proxy. AGENTS.md
"Performance budgets" says the final check is on real phones: "a test page reports frame times so
a person can read them off the device." This file specifies that page. **P1 implements it; X6 (this
brief) does not.** It lives in `web/` and is P1-owned.

## Route

`/perf` (and `/perf/[scanId]`, defaulting to `scn_trap`). Not linked from the landing page or the
demo gallery; reachable only by typing the URL, so it never ships in the judged demo path. Behind
the same loopback gate as every other route — no new auth.

## What it loads

1. The hero scan's reconstruction (`GET /api/scans/{id}/reconstruction`) and the real viewer
   component — the same `SparkRenderer` + `SplatMesh` + mesh/part GLB path as `/scan/[id]`, under
   the one `scan_to_web` parent group. This page must exercise the production viewer, not a stub,
   or the numbers mean nothing.
2. A device-capability pick between the full assets and the mobile assets (`splat_mobile_uri` /
   `mesh_mobile_uri` when present), with a visible toggle, so a tester can compare both on the same
   phone.

## What it shows on screen (large, readable at arm's length)

- **Live FPS**: a running median and 5th-percentile (worst) fps over a rolling 2 s window, updated
  every frame from `requestAnimationFrame` deltas. Start a 5 s "orbit test" button that spins the
  camera and reports p50 / p95 fps for that window, matching how the Playwright suite measures it.
- **Marks**: the elapsed time of each `performance.mark` below, in ms from navigation, as they fire.
- **Device line**: `devicePixelRatio` (and whether the viewer capped it at 2), viewport size, UA,
  the WebGL `RENDERER` string (`WEBGL_debug_renderer_info`), and which asset tier is loaded
  (full vs mobile).
- **Asset line**: the splat gaussian count and mesh triangle count actually loaded, and the bytes
  transferred for each, read from the `PerformanceResourceTiming` entries for those URLs.
- **Chat first token**: a "send test message" button that posts a fixed prompt and shows the ms to
  the first streamed token (mock mode: should be immediate).

## Marks this suite consumes (please emit exactly these names)

The Playwright suite (`perf/playwright/run.mjs`) and this page both read these `performance.mark`
names. Emit them from the production viewer so lab and device numbers line up. The web app already
emits `viewer:first-mesh`, `viewer:first-splat` and `chat:first-token` (see `web/lib/perf.ts`);
`shell:ready` is the one still to add.

| Mark name | Status | Fire when |
|---|---|---|
| `shell:ready` | **to add (P1)** | the workspace shell is painted: panels, status bar and the empty viewer canvas are on screen, before any 3D asset has loaded (this is the "workspace first paint" budget, under 2 s). Until it lands the suite falls back to `[data-testid=status-bar]` becoming visible |
| `viewer:first-mesh` | emitted | the first mesh primitive is actually drawn in the r3f canvas (first frame where the mesh is in the scene and rendered), not when the GLB finishes downloading — this is the "mesh visible under 5 s" budget |
| `viewer:first-splat` | emitted | the first splat is drawn (optional; the splat may stream in after the mesh). Null is fine when no splat tier is shown |
| `chat:first-token` | emitted | the first streamed assistant token is on screen — this is the "chat first token under 2 s" budget |

Emit them with `performance.mark(name)` on the real render path (e.g. inside the r3f frame callback
guarded by a "have I drawn this yet" flag), not from a `useEffect` that only means "mounted".

## Hooks the Playwright suite relies on (so the lab run can drive the page)

These let the headless run find and drive the UI without guessing selectors. The web app already
exposes all of them:

- `data-testid="status-bar"` — the shell-ready DOM fallback (visible once the status bar paints).
- `data-testid="viewer-canvas"` — the orbit-drag target; a pointer drag across its centre orbits
  the camera, and a tiny move nudges a render under r3f `frameloop="demand"`.
- `data-testid="chat-input"` and `data-testid="chat-send"` — the chat box and send button.
- The suite detects the splat/mesh toggle by the button text "Mesh" and clicks it so the mesh layer
  mounts (the hero scan defaults to the splat view). A stable `data-testid` on that toggle would be
  more robust than matching text — add one if convenient.

A `data-testid="assistant-message"` on each streamed assistant bubble would let the suite confirm
the first token by DOM as well as by the `chat:first-token` mark; today it falls back to the
assistant markdown container (`.prose`).

If any selector is inconvenient, tell X6 and we will update the suite — they live in one place
(`perf/playwright/run.mjs`).

## This host has no GPU — fps and first-mesh are read off a real phone

Chromium on the lab host renders WebGL in **software** (SwiftShader), so orbit fps and the slow-4G
first-mesh time are confounded by software decode and are **not** graded against the 30 fps and
under-5 s budgets — the suite reports them as informational with that caveat and SKIPs the budget.
`/perf` on an actual iPhone 12 / Pixel 6 is the authority for both. CPU throttling and an emulated
network are not a real mid-range GPU or a real radio either; the lab suite catches regressions
cheaply and gates CI, the phone gives the number judging rests on.
