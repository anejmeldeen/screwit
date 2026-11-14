# `perf/` — Screw It performance suite (X6)

Proves every budget in `AGENTS.md` → "Performance budgets" with numbers. It is built to run as
soon as the web (P1) and API (P4) branches merge; until then each suite **SKIPs cleanly** with a
reason rather than erroring, and the report still writes.

## Run it

```bash
perf/run.sh
```

Starts the FastAPI app in mock mode and the production web build (both `127.0.0.1` only), waits for
readiness, runs the suites, writes `perf/out/report.json` and `perf/out/report.md`, and stops both
servers (a trap kills every process it started, even on error or Ctrl-C).

Environment knobs:

| Var | Default | Meaning |
|---|---|---|
| `BACKEND_DIR` | main tree `backend/` | where `python -m screwit.app` runs (mock mode) |
| `WEB_DIR` | main tree `web/` | where `npm run build && npm run start` runs |
| `VENV_PY` | main tree shared venv | Python interpreter |
| `API_PORT` / `WEB_PORT` | 8000 / 3000 | loopback ports |
| `STRICT=1` | off | non-zero exit if any **required** budget is missing or over |
| `SKIP_BROWSER=1` | off | skip Chromium install + Lighthouse/Playwright (API + assets only) |

## Suites

1. **Lighthouse** (`lighthouse/run.mjs`, pinned `lighthouse@12.2.1`), headless Chromium, mobile
   preset, on `/` and `/scan/scn_trap`: LCP, TBT, CLS, total JS transferred, and a network-log check
   that the landing page loads **no three.js and no Spark**.
2. **Playwright** (`playwright/run.mjs`, pinned `playwright@1.48.2`), iPhone 12 emulation, 4x CPU
   throttling and a slow-4G profile via CDP: time to workspace shell, time to first mesh, orbit fps
   (p50 / worst-case p95) over 5 s, and chat first-token latency in mock mode. Also a desktop
   unthrottled comparison run. Reads `performance.mark`s from the viewer — see `DEVICE_TEST.md`.
3. **API latency** (`api_latency.py`): p50/p95 for every mock-mode GET route over 50 requests, and
   SSE first-event latency on the chat POST and progress GET (keep-alive comments skipped).
4. **Asset sizes** (`asset_sizes.py`): for each hero scan, if `splat_mobile_uri` / `mesh_mobile_uri`
   are present, counts gaussians (`.spz`/`.ply` header) and triangles (`.glb` accessors) against the
   budgets; if absent, reports them missing (FAIL in strict mode, flagged otherwise).

`collect.py` grades every suite's numbers against `budgets.py` and writes the report with
`report.py`. `budgets.py` is the single source of truth for the budget table and comparison logic.

## Tests (`perf/tests/`)

The parts that do **not** depend on other teams — the report writer, the budget comparison logic
(with fake numbers), the asset parsers (synthetic files) and the latency percentile helper:

```bash
PYTHONPATH=perf /local/home/brancli/.kiro/crew/workspace/screw-it/backend/.venv/bin/python \
  -m pytest perf/tests -q
```

## Outputs

`perf/out/` (gitignored): `report.json`, `report.md`, `lighthouse.json`, `playwright.json`, and
`logs/`. The committed source carries no numbers; numbers are produced by a run.

## What the device page needs from P1

`DEVICE_TEST.md` specifies the `/perf` page for P1 to build and lists the exact `performance.mark`
names and `data-testid` hooks this suite consumes.
