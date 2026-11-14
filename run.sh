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
#   export SCREWIT_SAM3=1          # `start` installs sam3-cpu (from GitHub) if needed
#   (SAM 3 weights are gated: run `<venv>/bin/huggingface-cli login` + accept the model license once)
#
# Dynamic/dev edits:
#   export SCREWIT_RELOAD=1        # backend auto-reloads on code changes (uvicorn --reload)
#   start always pulls latest (fast-forward), rebuilds the web app, and kills old servers first,
#   so every `start` serves the newest code. Escape hatches: SCREWIT_NO_PULL=1, SCREWIT_SKIP_BUILD=1.
#
# Config via env (defaults in parens); export to override:
#   SCREWIT_PORT (8000)  WEB_PORT (3000)  SCREWIT_DEVICE (cpu)  SCREWIT_CPU_VIEW_CAP (8)
#   SCREWIT_INFER_GS (0)  SCREWIT_MODEL (anthropic/claude-sonnet-4.5)  SCREWIT_DATA_DIR (<repo>/data-live)
#   SCREWIT_LLM (openrouter)  SCREWIT_SAM3 (0)  SCREWIT_SAM3_BACKEND (auto)  SCREWIT_RELOAD (0)
#   Mesh cleanup (improves surface quality; usually leave at defaults):
#   SCREWIT_SMOOTH_ITERS (30, Taubin smoothing passes; 0 disables)  SCREWIT_DENOISE (1; 0 disables)
#   SCREWIT_REMOVE_TABLE (1; cuts a fused table/support slab from the mesh; 0 disables)

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

# Check SAM 3 gated-weight access and guide the user through the one-time grant. SAM 3 needs both a
# HuggingFace login AND access to the gated facebook/sam3 repo (click "Agree" on the model page).
ensure_sam3_weights() {
  # Make the HF token discoverable regardless of HF_HOME: the token lives in the DEFAULT HF cache
  # (~/.cache/huggingface/token) but we run with HF_HOME pointed elsewhere, so export HF_TOKEN.
  if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)"
  fi
  # Authoritative check: actually fetch a gated file header. Distinguishes the two failure modes.
  local status
  status="$("$PY" - <<'PY' 2>/dev/null
import sys
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
try:
    hf_hub_download("facebook/sam3", "config.json")
    print("ok")
except GatedRepoError:
    print("gated")          # authed but not authorized (403), or not logged in (401)
except Exception as e:
    print("error:" + type(e).__name__)
PY
)"
  case "$status" in
    ok) ok "SAM 3 weights accessible" ;;
    gated|*)
      warn "SAM 3 weights (facebook/sam3) are not accessible for your account yet."
      echo "  One time, to enable multi-part segmentation:"
      echo "    1. Log in:  $VENV/bin/huggingface-cli login   (paste a token from https://huggingface.co/settings/tokens)"
      echo "    2. Request access: open https://huggingface.co/facebook/sam3 and click 'Agree and access repository'."
      echo "    3. Wait for Meta to grant access (can be instant or take a while), then re-run ./run.sh start."
      if [ -t 0 ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
        printf '  Run the HuggingFace login now? [y/N] '
        read -r ans
        case "$ans" in y|Y) "$VENV/bin/huggingface-cli" login || warn "login failed" ;; esac
      fi
      warn "Until access is granted, SAM 3 falls back to single-part segmentation."
      ;;
  esac
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
  if [ "${SCREWIT_SAM3:-0}" = "1" ]; then
    if ! "$PY" -c "import sam3" >/dev/null 2>&1; then
      info "Installing SAM 3 CPU segmenter (sam3-cpu from GitHub)…"
      "$UV" pip install -p "$PY" "git+https://github.com/rhubarb-ai/sam3-cpu.git" \
        || warn "sam3-cpu install failed; multi-part will fall back to single part"
    fi
    # The CLIP BPE vocab asset is not packaged by the git install; fetch it once if missing.
    local bpe="$VENV/lib/python3.12/site-packages/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if [ ! -f "$bpe" ] && [ -d "$VENV/lib/python3.12/site-packages/sam3" ]; then
      info "Fetching SAM 3 tokenizer vocab…"
      mkdir -p "$(dirname "$bpe")"
      curl -L -fsS -o "$bpe" \
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz" \
        || warn "could not fetch the BPE vocab; SAM 3 will fail until it is present"
    fi
    ensure_sam3_weights
  fi
  if command -v npm >/dev/null 2>&1; then
    [ -d "$WEB/node_modules" ] || { info "Installing web dependencies…"; ( cd "$WEB" && npm ci ); }
    # Always (re)build so `start` serves the LATEST UI. `next start` runs a prebuilt snapshot and
    # will otherwise keep serving a stale .next after a git pull/merge. Next's cache makes a
    # no-change rebuild fast. Skip only if SCREWIT_SKIP_BUILD=1 (e.g. iterating on backend only).
    if [ "${SCREWIT_SKIP_BUILD:-0}" = "1" ] && [ -d "$WEB/.next" ]; then
      warn "SCREWIT_SKIP_BUILD=1 — serving the existing web build, not rebuilding"
    else
      info "Building the web app (always, to serve the latest UI)…"
      ( cd "$WEB" && npm run build )
    fi
  else
    warn "npm not found — install Node 22 to run the frontend"
  fi
  mkdir -p "$DATA_DIR" "$HF_DIR" "$RUN_DIR"
}

alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" >/dev/null 2>&1; }

# Pull the latest code so `start` always serves the newest UI/backend. Safe: fetches, then only
# fast-forwards; never merges/rebases/forces, and skips entirely if the tree is dirty (so local work
# is never clobbered) or if offline. Disable with SCREWIT_NO_PULL=1.
update_repo() {
  [ "${SCREWIT_NO_PULL:-0}" = "1" ] && { warn "SCREWIT_NO_PULL=1 — skipping git update"; return; }
  command -v git >/dev/null 2>&1 || return
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return
  local branch; branch="$(git -C "$ROOT" symbolic-ref --short -q HEAD || echo "")"
  [ -n "$branch" ] || { warn "detached HEAD — skipping git update"; return; }
  if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    warn "uncommitted changes — skipping git update (serving your local code)"
    return
  fi
  info "Updating from origin/$branch (fast-forward only)…"
  if git -C "$ROOT" fetch --quiet origin "$branch" 2>/dev/null; then
    if git -C "$ROOT" merge --ff-only "origin/$branch" >/dev/null 2>&1; then
      ok "updated to $(git -C "$ROOT" rev-parse --short HEAD)"
    else
      warn "local branch diverged from origin/$branch — not fast-forwarding; serving local code"
    fi
  else
    warn "could not reach origin (offline?) — serving local code"
  fi
}

start() {
  : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY=sk-or-... first (never store it in the repo)}"
  # Always clear any previous run first (pid files AND whatever holds the ports), so `start` never
  # collides with a stale server (EADDRINUSE) and always ends up serving the fresh build.
  stop_quiet
  update_repo
  ensure_installed

  info "Starting backend → http://127.0.0.1:${SCREWIT_PORT:-8000}  (device=${SCREWIT_DEVICE:-cpu}, SAM3=${SCREWIT_SAM3:-0})"
  (
    cd "$ROOT"
    export SCREWIT_MODE=live
    export SCREWIT_DEVICE="${SCREWIT_DEVICE:-cpu}"
    export SCREWIT_CPU_VIEW_CAP="${SCREWIT_CPU_VIEW_CAP:-8}"
    export SCREWIT_INFER_GS="${SCREWIT_INFER_GS:-0}"
    export SCREWIT_PORT="${SCREWIT_PORT:-8000}"
    export SCREWIT_DATA_DIR="$DATA_DIR"
    export SCREWIT_LLM="${SCREWIT_LLM:-openrouter}"
    export SCREWIT_MODEL="${SCREWIT_MODEL:-anthropic/claude-sonnet-4.5}"
    export HF_HOME="$HF_DIR"
    export SCREWIT_SAM3="${SCREWIT_SAM3:-0}"
    export SCREWIT_RELOAD="${SCREWIT_RELOAD:-0}"
    [ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
    [ -n "${SCREWIT_SAM3_BACKEND:-}" ] && export SCREWIT_SAM3_BACKEND
    exec "$PY" -m screwit.app
  ) >"$BE_LOG" 2>&1 &
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

stop_quiet() {
  # Kill processes from the pid files, then anything still holding our ports (orphans the pid file
  # missed). Quiet: no "nothing was running" noise; used by both `start` and `stop`.
  local pid
  for pf in "$WEB_PID" "$BE_PID"; do
    if alive "$pf"; then
      pid="$(cat "$pf")"
      kill "$pid" 2>/dev/null || true
      sleep 1; kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
  done
  kill_port "${WEB_PORT:-3000}"
  kill_port "${SCREWIT_PORT:-8000}"
}

kill_port() {
  command -v lsof >/dev/null 2>&1 || return 0
  local p pids; pids="$(lsof -ti "tcp:$1" 2>/dev/null || true)"
  [ -n "$pids" ] || return 0
  for p in $pids; do kill "$p" 2>/dev/null || true; done
  sleep 1
  pids="$(lsof -ti "tcp:$1" 2>/dev/null || true)"
  for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
}

stop() {
  local running=0
  alive "$WEB_PID" && running=1
  alive "$BE_PID" && running=1
  if command -v lsof >/dev/null 2>&1; then
    [ -n "$(lsof -ti "tcp:${WEB_PORT:-3000}" 2>/dev/null)" ] && running=1
    [ -n "$(lsof -ti "tcp:${SCREWIT_PORT:-8000}" 2>/dev/null)" ] && running=1
  fi
  if [ "$running" = 1 ]; then
    info "stopping backend + frontend"
    stop_quiet
    ok "stopped"
  else
    warn "nothing was running"
  fi
}

case "${1:-}" in
  start) start ;;
  stop)  stop ;;
  *) err "usage: ./run.sh {start|stop}"; exit 1 ;;
esac
