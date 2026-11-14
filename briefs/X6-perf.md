# X6 performance tests (worktree screw-it-wt/perf, branch ws/perf)

Read: briefs/common.md, AGENTS.md (especially "Performance budgets"), SPEC.md 2, 4.8, 5, 8, and briefs/P1-web.md, briefs/P4-platform.md (what is being built).

You own only perf/ (top-level). Create nothing elsewhere.

Purpose: prove every budget in AGENTS.md "Performance budgets" with numbers, ready to run as soon as the web and API branches merge. Until then, tests must SKIP cleanly with a reason, never error.

Build, everything binding 127.0.0.1 only and every process you start stopped at the end:
1. perf/run.sh: starts the FastAPI app in mock mode (`python -m screwit.app`, env SCREWIT_MODE=mock, from the backend dir given by BACKEND_DIR, default the main tree) and the production web build (`npm run build` then `npm run start` in WEB_DIR, default the main tree's web/), waits for readiness, runs the suites below, writes perf/out/report.json and report.md (gitignored outputs), stops both.
2. Lighthouse (pinned npm version, headless Chromium) mobile preset on the landing page and on /scan/scn_trap: record LCP, TBT, CLS, total JS transferred, and fail if over budget. Also check that the landing page does not load three.js or Spark (inspect the network log for their chunks).
3. Playwright (pinned) with mobile emulation (an iPhone 12-class viewport and user agent), 4x CPU throttling via CDP, and a slow-4G network profile via CDP: time to workspace shell, time until the first mesh is drawn (use a `performance.mark` the viewer emits; list the mark names you need from P1 in your report, e.g. `viewer:first-mesh`, `viewer:first-splat`), frame times while orbiting for 5 s (requestAnimationFrame deltas collected in page; report p50/p95 fps), and chat first-token latency in mock mode. Also run desktop unthrottled for comparison.
4. Asset size checks: fetch reconstruction for each hero scan; if splat_mobile_uri / mesh_mobile_uri are present, assert their sizes and counts against the budgets; if absent, report them as missing (fail in strict mode, flag otherwise).
5. API latency: p50/p95 for every GET route in mock mode over 50 requests, and SSE first-event latency.
6. A device page spec for P1 to implement (do not implement it): describe in perf/DEVICE_TEST.md a route like /perf that loads a hero scan and shows live fps and the marks, so a person can read the numbers off a real phone. List exactly what you need P1 to expose.
7. perf/tests/: pytest that runs the parts that do not depend on other teams now (report writer, budget comparison logic with fake numbers) using the main backend venv.

Install Chromium for Lighthouse/Playwright under the default cache; if downloads fail, say exactly what failed.

Run everything now and report what passes, what skips and why, and the instrumentation you need from P1.
