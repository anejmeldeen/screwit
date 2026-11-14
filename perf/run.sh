#!/usr/bin/env bash
# Screw It performance suite orchestrator (brief step 1).
#
# Starts the FastAPI app in mock mode and the production web build, both bound to 127.0.0.1 only,
# waits for readiness, runs the Lighthouse / Playwright / API-latency / asset-size suites, writes
# perf/out/report.{json,md}, and stops both servers. Every process started here is killed on exit
# (including on error or Ctrl-C) by the trap below.
#
# When the web or API build is not present yet (the P1/P4 branches have not merged into the tree
# this points at), the corresponding server simply does not start and its suites SKIP cleanly; the
# report still writes with those budgets marked SKIP. Nothing here errors on a missing dependency.
#
# Env:
#   BACKEND_DIR   backend dir to run `python -m screwit.app` from (default: main tree backend)
#   WEB_DIR       web dir to `npm run build && npm run start` in   (default: main tree web/)
#   VENV_PY       python interpreter (default: main tree shared venv)
#   API_PORT      FastAPI port   (default 8000)
#   WEB_PORT      Next.js port   (default 3000)
#   STRICT=1      fail the run if any required budget is missing/over
#   SKIP_BROWSER=1  skip Chromium install + Lighthouse/Playwright (API + assets only)
set -u

PERF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_TREE="/local/home/brancli/.kiro/crew/workspace/screw-it"

BACKEND_DIR="${BACKEND_DIR:-$MAIN_TREE/backend}"
WEB_DIR="${WEB_DIR:-$MAIN_TREE/web}"
VENV_PY="${VENV_PY:-$MAIN_TREE/backend/.venv/bin/python}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
BIND="127.0.0.1"
API_BASE="http://$BIND:$API_PORT"
WEB_BASE="http://$BIND:$WEB_PORT"
OUT_DIR="$PERF_DIR/out"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Refuse to run against a server we did not start: a stale listener on either port made earlier
# runs silently measure an old build (the new server failed with EADDRINUSE).
for port in "$API_PORT" "$WEB_PORT"; do
  if (exec 3<>"/dev/tcp/$BIND/$port") 2>/dev/null; then
    echo "port $port on $BIND is already in use; stop that server or set API_PORT/WEB_PORT" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------------------
# Browser toolchain. The integration lead stages a shared set under screw-it-tools/ with
# libatomic.so.1 preloaded (the Playwright driver needs it on this host). Honour an externally
# provided LD_LIBRARY_PATH / PLAYWRIGHT_BROWSERS_PATH; otherwise auto-detect the shared dir and
# fall back to the default per-user cache when it is absent.
# ---------------------------------------------------------------------------
TOOLS_DIR="${SCREWIT_TOOLS_DIR:-/local/home/brancli/.kiro/crew/workspace/screw-it-tools}"
if [[ -d "$TOOLS_DIR/lib" ]]; then
  export LD_LIBRARY_PATH="$TOOLS_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" && -d "$TOOLS_DIR/ms-playwright" ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="$TOOLS_DIR/ms-playwright"
fi

API_PID=""
WEB_PID=""
# A fresh throwaway data dir per run (the lead requires a fresh SCREWIT_DATA_DIR for every server).
# Under $KIROCREW_SCRATCH when set (session-owned, auto-reclaimed), else a mktemp fallback.
API_DATA_DIR=""

cleanup() {
  local code=$?
  if [[ -n "$WEB_PID" ]] && kill -0 "$WEB_PID" 2>/dev/null; then
    echo "stopping web server (pgid $WEB_PID)"
    # WEB_PID is the setsid session leader; its PGID equals its PID, so kill the whole group
    # (npm + the next-server grandchild) in one signal.
    kill -- "-$WEB_PID" 2>/dev/null || kill "$WEB_PID" 2>/dev/null
    pkill -P "$WEB_PID" 2>/dev/null || true
    wait "$WEB_PID" 2>/dev/null || true
  fi
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    echo "stopping API server (pid $API_PID)"
    kill "$API_PID" 2>/dev/null
    wait "$API_PID" 2>/dev/null || true
  fi
  # Remove the per-run data dir (only ever a mktemp dir we created).
  if [[ -n "$API_DATA_DIR" && -d "$API_DATA_DIR" ]]; then
    rm -rf "$API_DATA_DIR" 2>/dev/null || true
  fi
  exit $code
}
trap cleanup EXIT INT TERM

wait_ready() {
  # wait_ready <url> <seconds>
  local url="$1" secs="$2" i=0
  while (( i < secs )); do
    if curl -fsS -o /dev/null --max-time 2 "$url" 2>/dev/null; then return 0; fi
    sleep 1; ((i++))
  done
  return 1
}

# ---------------------------------------------------------------------------
# 1. FastAPI app in mock mode (loopback only).
# ---------------------------------------------------------------------------
API_UP=0
# `python -m screwit.app` runs the package __main__ (P4 brief: app/__main__.py). Probe for that
# module specifically — the empty `screwit.app` package exists before P4 merges but has no __main__.
if PYTHONPATH="$BACKEND_DIR" "$VENV_PY" -c \
      "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('screwit.app.__main__') else 1)" 2>/dev/null; then
  # Fresh, empty data dir for this run; the mock app writes its SQLite + any scratch here.
  API_DATA_DIR="$(mktemp -d "${KIROCREW_SCRATCH:-${TMPDIR:-/tmp}}/screwit-perf-data.XXXXXX")"
  # Serve fixtures from the repo root that owns $BACKEND_DIR (main tree by default).
  APP_FIXTURES="$(cd "$BACKEND_DIR/.." && pwd)/fixtures"
  echo "starting FastAPI (mock) from $BACKEND_DIR on $API_BASE (data dir: $API_DATA_DIR)"
  SCREWIT_MODE=mock SCREWIT_BIND="$BIND" SCREWIT_PORT="$API_PORT" \
    SCREWIT_DATA_DIR="$API_DATA_DIR" SCREWIT_FIXTURES_DIR="$APP_FIXTURES" \
    PYTHONPATH="$BACKEND_DIR" \
    "$VENV_PY" -m screwit.app >"$LOG_DIR/api.log" 2>&1 &
  API_PID=$!
  # Poll readiness, but stop early if the process has already exited (fast failure).
  i=0
  while (( i < 30 )); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "API process exited early; see $LOG_DIR/api.log"
      API_PID=""
      break
    fi
    if curl -fsS -o /dev/null --max-time 2 "$API_BASE/api/demo-scans" 2>/dev/null; then
      API_UP=1; echo "API ready"; break
    fi
    sleep 1; ((i++))
  done
  [[ "$API_UP" -eq 0 && -n "$API_PID" ]] && echo "API did not become ready in 30s; see $LOG_DIR/api.log"
fi
if [[ "$API_UP" -eq 0 ]]; then
  echo "FastAPI app not available (screwit.app has no runnable __main__, or did not start) — API/SSE budgets will SKIP"
fi

# ---------------------------------------------------------------------------
# 2. Production web build (loopback only).
# ---------------------------------------------------------------------------
WEB_UP=0
if [[ -f "$WEB_DIR/package.json" ]]; then
  if [[ ! -d "$WEB_DIR/node_modules" ]]; then
    echo "installing web deps in $WEB_DIR"
    (cd "$WEB_DIR" && npm ci >"$LOG_DIR/web-npm-ci.log" 2>&1) \
      || (cd "$WEB_DIR" && npm install >"$LOG_DIR/web-npm-ci.log" 2>&1) \
      || echo "web npm install failed; see $LOG_DIR/web-npm-ci.log"
  fi
  echo "building web in $WEB_DIR"
  if (cd "$WEB_DIR" && npm run build >"$LOG_DIR/web-build.log" 2>&1); then
    echo "starting web (npm run start) on $WEB_BASE"
    # Own process group (setsid) so cleanup can kill npm AND its next-server grandchild together.
    setsid bash -c "cd '$WEB_DIR' && PORT='$WEB_PORT' SCREWIT_API_ORIGIN='$API_BASE' npm run start" \
      >"$LOG_DIR/web.log" 2>&1 &
    WEB_PID=$!
    if wait_ready "$WEB_BASE/" 40; then
      WEB_UP=1; echo "web ready"
    else
      echo "web did not become ready in 40s; see $LOG_DIR/web.log — continuing (web suites will SKIP)"
    fi
  else
    echo "web build failed; see $LOG_DIR/web-build.log — web suites will SKIP"
  fi
else
  echo "web/ has no package.json (P1 branch not merged) — Lighthouse/Playwright will SKIP"
fi

# ---------------------------------------------------------------------------
# 3. Browser install + Lighthouse + Playwright (only if web is up).
# ---------------------------------------------------------------------------
LH_OUT="$OUT_DIR/lighthouse.json"
PW_OUT="$OUT_DIR/playwright.json"
rm -f "$LH_OUT" "$PW_OUT"

if [[ "$WEB_UP" -eq 1 && "${SKIP_BROWSER:-0}" != "1" ]]; then
  if [[ ! -d "$PERF_DIR/node_modules" ]]; then
    echo "installing perf npm deps (pinned)"
    (cd "$PERF_DIR" && npm ci >"$LOG_DIR/npm-ci.log" 2>&1) \
      || (cd "$PERF_DIR" && npm install >"$LOG_DIR/npm-ci.log" 2>&1) \
      || echo "npm install failed; see $LOG_DIR/npm-ci.log — web suites will SKIP"
  fi
  echo "installing Chromium for Lighthouse/Playwright (browsers path: ${PLAYWRIGHT_BROWSERS_PATH:-default cache})"
  (cd "$PERF_DIR" && npx --yes playwright install chromium >"$LOG_DIR/chromium-install.log" 2>&1) \
    || echo "Chromium download FAILED; see $LOG_DIR/chromium-install.log — web suites will SKIP"

  # Lighthouse drives Chrome via chrome-launcher; point CHROME_PATH at the Playwright chromium
  # binary our pinned version resolves, so both tools use the same verified build.
  CHROME_PATH_RESOLVED="$(cd "$PERF_DIR" && node -e "try{console.log(require('playwright').chromium.executablePath())}catch(e){process.exit(1)}" 2>/dev/null)"
  if [[ -n "$CHROME_PATH_RESOLVED" && -x "$CHROME_PATH_RESOLVED" ]]; then
    export CHROME_PATH="$CHROME_PATH_RESOLVED"
    echo "CHROME_PATH=$CHROME_PATH"
  fi

  echo "running Lighthouse"
  (cd "$PERF_DIR" && node lighthouse/run.mjs "$WEB_BASE" "$LH_OUT" 2>>"$LOG_DIR/lighthouse.log") || true
  echo "running Playwright"
  (cd "$PERF_DIR" && node playwright/run.mjs "$WEB_BASE" "$PW_OUT" 2>>"$LOG_DIR/playwright.log") || true
else
  echo "skipping browser suites (web not up or SKIP_BROWSER=1)"
fi

# ---------------------------------------------------------------------------
# 4. Aggregate -> report.json / report.md (API latency + asset sizes run inside collect.py).
# ---------------------------------------------------------------------------
STRICT_FLAG=""
[[ "${STRICT:-0}" == "1" ]] && STRICT_FLAG="--strict"
API_FLAG=""
[[ "$API_UP" -eq 1 ]] && API_FLAG="--api-base $API_BASE"

echo "aggregating report"
# shellcheck disable=SC2086
PYTHONPATH="$PERF_DIR" "$VENV_PY" "$PERF_DIR/collect.py" \
  --out "$OUT_DIR" \
  --repo "$MAIN_TREE" \
  --web-base "$WEB_BASE" \
  --lighthouse "$LH_OUT" \
  --playwright "$PW_OUT" \
  $API_FLAG $STRICT_FLAG
RC=$?

echo "report at $OUT_DIR/report.md"
exit $RC
