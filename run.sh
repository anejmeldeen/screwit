#!/usr/bin/env bash
# Screw It — start or stop the whole app (backend + frontend) on macOS or Linux.
#
#   ./run.sh start    install anything missing, then launch backend + frontend
#   ./run.sh stop     stop the backend + frontend started by `start`
#
# Export your key first (never stored in the repo, per AGENTS.md decision 4):
#   export OPENROUTER_API_KEY=sk-or-...
#
# Optional, to enable multi-part SAM 3 segmentation on first start:
#   export SCREWIT_SAM3=1          # `start` will install sam3-cpu if needed
#   (SAM 3 weights are gated: run `<venv>/bin/huggingface-cli login` + accept the model license once)
#
# Config via env (defaults in parens); export to override:
#   SCREWIT_PORT (8000)  WEB_PORT (3000)  SCREWIT_DEVICE (cpu)  SCREWIT_CPU_VIEW_CAP (8)
#   SCREWIT_MODEL (anthropic/claude-sonnet-4.5)  SCREWIT_DATA_DIR (<repo>/data-live)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
WEB="$ROOT/web"
VENV="$BACKEND/.venv"
PY="$VENV/bin/python"
DATA_DIR="${SCREWIT_DATA_DIR:-$ROOT/data-live}"
HF_DIR="${HF_HOME:-$ROOT/cpu-verify/.cache/hf}"
RUN_DIR="$ROOT/.run"            # pid + log files (gitignored)
BE_PID="$RUN_DIR/backend.pid"; BE_LOG="$RUN_DIR/backend.log"
WEB_PID="$RUN_DIR/web.pid";    WEB_LOG="$RUN_DIR/web.log"

info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m OK\033[0m %s\n' "$*"; }
warn() { printf '\033[33m  !\033[0m %s\n' "$*"; }
err()  { printf '\033[31mERR\033[0m %s\n' "$*" >&2; }

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv
  elif [ -x "$HOME/.local/bin/uv" ]; then echo "$HOME/.local/bin/uv"
  else return 1; fi
}

# Install only what is missing, so `start` is safe to run repeatedly.
ensure_installed() {
  local UV
  if ! UV="$(find_uv)"; then
    info "Installing uv (Python manager)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
  fi
  if [ ! -x "$PY" ]; then
    info "Creating Python 3.12 venv…"
    "$UV" venv -p 3.12 "$VENV"
  fi
  if ! "$PY" -c "import screwit, fastapi, open3d" >/dev/null 2>&1; then
    info "Installing backend dependencies (first run; a few minutes)…"
    "$UV" pip install -p "$PY" -e "$BACKEND[geometry,cad,agent,storage,dev]"
  fi
  if [ "${SCREWIT_SAM3:-0}" = "1" ] && ! "$PY" -c "import sam3" >/dev/null 2>&1; then
    info "Installing SAM 3 CPU segmenter (sam3-cpu)…"
    "$UV" pip install -p "$PY" sam3-cpu || warn "sam3-cpu install failed; multi-part will fall back to single part"
    warn "SAM 3 weights are gated: run '$VENV/bin/huggingface-cli login' and accept the model license once."
  fi
  if command -v npm >/dev/null 2>&1; then
    [ -d "$WEB/node_modules" ] || { info "Installing web dependencies…"; ( cd "$WEB" && npm ci ); }
    [ -d "$WEB/.next" ] || { info "Building the web app…"; ( cd "$WEB" && npm run build ); }
  else
    warn "npm not found — install Node 22 to run the frontend"
  fi
  mkdir -p "$DATA_DIR" "$HF_DIR" "$RUN_DIR"
}

alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" >/dev/null 2>&1; }

start() {
  : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY=sk-or-... first (never store it in the repo)}"
  if alive "$BE_PID" || alive "$WEB_PID"; then
    warn "already running (use ./run.sh stop first). backend pid $(cat "$BE_PID" 2>/dev/null), web pid $(cat "$WEB_PID" 2>/dev/null)"
    exit 1
  fi
  ensure_installed

  info "Starting backend → http://127.0.0.1:${SCREWIT_PORT:-8000}  (device=${SCREWIT_DEVICE:-cpu}, SAM3=${SCREWIT_SAM3:-0})"
  ( cd "$ROOT" && \
    SCREWIT_MODE=live \
    SCREWIT_DEVICE="${SCREWIT_DEVICE:-cpu}" \
    SCREWIT_CPU_VIEW_CAP="${SCREWIT_CPU_VIEW_CAP:-8}" \
    SCREWIT_INFER_GS="${SCREWIT_INFER_GS:-0}" \
    SCREWIT_PORT="${SCREWIT_PORT:-8000}" \
    SCREWIT_DATA_DIR="$DATA_DIR" \
    SCREWIT_LLM="${SCREWIT_LLM:-openrouter}" \
    SCREWIT_MODEL="${SCREWIT_MODEL:-anthropic/claude-sonnet-4.5}" \
    HF_HOME="$HF_DIR" \
    SCREWIT_SAM3="${SCREWIT_SAM3:-0}" \
    ${SCREWIT_SAM3_BACKEND:+SCREWIT_SAM3_BACKEND="$SCREWIT_SAM3_BACKEND"} \
    "$PY" -m screwit.app >"$BE_LOG" 2>&1 ) &
  echo $! >"$BE_PID"

  if command -v npm >/dev/null 2>&1; then
    info "Starting frontend → http://127.0.0.1:${WEB_PORT:-3000}"
    ( cd "$WEB" && PORT="${WEB_PORT:-3000}" SCREWIT_API_ORIGIN="http://127.0.0.1:${SCREWIT_PORT:-8000}" \
      npm run start >"$WEB_LOG" 2>&1 ) &
    echo $! >"$WEB_PID"
  fi

  sleep 3
  alive "$BE_PID" && ok "backend up (pid $(cat "$BE_PID"), log $BE_LOG)" || { err "backend failed to start — see $BE_LOG"; tail -n 15 "$BE_LOG" 2>/dev/null; }
  alive "$WEB_PID" && ok "frontend up (pid $(cat "$WEB_PID"), log $WEB_LOG)" || warn "frontend not running — see $WEB_LOG"
  ok "open http://127.0.0.1:${WEB_PORT:-3000}   ·   stop with ./run.sh stop"
}

stop() {
  local any=0
  for pf in "$WEB_PID" "$BE_PID"; do
    if alive "$pf"; then
      local pid; pid="$(cat "$pf")"
      info "stopping pid $pid"
      kill "$pid" 2>/dev/null || true
      # give it a moment, then force if still alive
      sleep 1; kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
      any=1
    fi
    rm -f "$pf"
  done
  [ "$any" = 1 ] && ok "stopped" || warn "nothing was running"
}

case "${1:-}" in
  start) start ;;
  stop)  stop ;;
  *) err "usage: ./run.sh {start|stop}"; exit 1 ;;
esac
