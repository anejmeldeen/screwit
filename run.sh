#!/usr/bin/env bash
# Screw It: start or stop the whole app (backend + frontend) on macOS or Linux.
#
#   ./run.sh start    install or update anything missing, then launch backend + frontend
#   ./run.sh stop     stop the backend + frontend started by `start`
#
# Export your key first (it is never stored in the repo, per AGENTS.md decision 4):
#   export OPENROUTER_API_KEY=sk-or-...
# (Not needed with SCREWIT_LLM=fake, which runs without any LLM.)
#
# What `start` does, every time, skipping whatever is already in place:
#   1. checks the machine (OS, git, curl, Node >= 20, free disk) and says exactly what is missing
#   2. stops the previous run (pid files and anything still holding the ports)
#   3. pulls the latest code (fast-forward only; skipped if you have local changes)
#   4. installs uv and a Python 3.12 virtualenv, then the backend and its extras; reinstalls when
#      backend/pyproject.toml changed since the last install
#   5. installs the CPU model stack: torch, Depth Anything 3 (pinned commit, cloned into
#      cpu-verify/src) and its runtime dependencies
#   6. downloads the Depth Anything 3 weights once (about 6.5 GB) so the first scan does not stall
#   7. installs gltfpack (phone-sized meshes) and the web dependencies; reinstalls when a lockfile
#      changed; always rebuilds the web app so the newest UI is served
#   8. verifies the whole stack imports, prints the versions, then starts both servers
#
# Dev and escape hatches:
#   SCREWIT_RELOAD=1      backend auto-reloads on code changes (uvicorn --reload)
#   SCREWIT_NO_PULL=1     do not git pull       SCREWIT_SKIP_BUILD=1    reuse the existing web build
#   SCREWIT_SKIP_MODELS=1 skip the model stack and weights (mock mode or tests only)
#
# Config via env (defaults in parens):
#   SCREWIT_PORT (8000)  WEB_PORT (3000)  SCREWIT_CPU_VIEW_CAP (8; also capped to what RAM can hold)
#   SCREWIT_DEVICE (auto: an NVIDIA GPU, else the Apple GPU (about 10x faster than the CPU), else cpu)
#   SCREWIT_INFER_GS (0)  SCREWIT_MODEL (anthropic/claude-sonnet-4.5)  SCREWIT_DATA_DIR (<repo>/data-live)
#   SCREWIT_LLM (openrouter; also bedrock, anthropic, fake)  SCREWIT_LLM_CACHE (1)  SCREWIT_RELOAD (0)
#   Scan quality (usually leave at defaults):
#   SCREWIT_DEPTH_CLEAN (1)  SCREWIT_MIN_VIEWS (1 = off; 2 peels single-view rim flaps but eats lower walls)
#   SCREWIT_SMOOTH_ITERS (80)  SCREWIT_DENOISE (1)  SCREWIT_REMOVE_TABLE (1)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
WEB="$ROOT/web"
TOOLCHAIN="$BACKEND/screwit/optimize/toolchain"
VENV="$BACKEND/.venv"
PY="$VENV/bin/python"
DATA_DIR="${SCREWIT_DATA_DIR:-$ROOT/data-live}"
HF_DIR="${HF_HOME:-$ROOT/cpu-verify/.cache/hf}"
SRC_DIR="$ROOT/cpu-verify/src"            # gitignored; third-party model sources
RUN_DIR="$ROOT/.run"                      # pid, log and install-stamp files (gitignored)
BE_PID="$RUN_DIR/backend.pid"; BE_LOG="$RUN_DIR/backend.log"
WEB_PID="$RUN_DIR/web.pid";    WEB_LOG="$RUN_DIR/web.log"

# Pinned model stack (verified on the CPU path; see cpu-verify/README.md).
TORCH_VERSION="2.5.1"; TORCHVISION_VERSION="0.20.1"
DA3_REPO="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_WEIGHTS="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_DEPS=(einops omegaconf addict plyfile pillow_heif safetensors "moviepy==1.0.3" e3nn evo pycolmap
          jaxtyping imageio psutil "huggingface_hub>=0.30,<1.0")

info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m OK\033[0m %s\n' "$*"; }
warn() { printf '\033[33m  !\033[0m %s\n' "$*"; }
err()  { printf '\033[31mERR\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

hash_of() {  # sha256 of the given files (portable: Linux sha256sum, macOS shasum)
  if command -v sha256sum >/dev/null 2>&1; then cat "$@" | sha256sum | cut -d' ' -f1
  else cat "$@" | shasum -a 256 | cut -d' ' -f1; fi
}
stamp_ok()  { [ -f "$RUN_DIR/$1.stamp" ] && [ "$(cat "$RUN_DIR/$1.stamp")" = "$2" ]; }
stamp_set() { mkdir -p "$RUN_DIR"; printf '%s' "$2" >"$RUN_DIR/$1.stamp"; }

find_uv() {
  if command -v uv >/dev/null 2>&1; then command -v uv
  elif [ -x "$HOME/.local/bin/uv" ]; then echo "$HOME/.local/bin/uv"
  elif [ -x "$HOME/.cargo/bin/uv" ]; then echo "$HOME/.cargo/bin/uv"
  else return 1; fi
}

# ------------------------------------------------------------------------------------- preflight
preflight() {
  local os; os="$(uname -s)"
  case "$os" in
    Darwin|Linux) ok "$os $(uname -m)" ;;
    *) die "unsupported OS $os (macOS or Linux only)" ;;
  esac
  local missing=0
  command -v git  >/dev/null 2>&1 || { err "git is missing (macOS: xcode-select --install; Linux: apt install git)"; missing=1; }
  command -v curl >/dev/null 2>&1 || { err "curl is missing (Linux: apt install curl)"; missing=1; }
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    local major; major="$(node -v | sed 's/^v//' | cut -d. -f1)"
    if [ "$major" -lt 20 ]; then err "Node $(node -v) is too old; install Node 22 (https://nodejs.org or nvm)"; missing=1
    else ok "Node $(node -v), npm $(npm -v)"; fi
  else
    err "Node.js/npm are missing; install Node 22 (macOS: brew install node@22; Linux: https://nodejs.org)"
    missing=1
  fi
  [ "$missing" = 0 ] || die "install the tools above, then run ./run.sh start again"
  # Disk: the weights alone are about 6.5 GB; torch and the venv add about 3 GB.
  local need_gb=2
  [ -d "$HF_DIR/hub/models--${DA3_WEIGHTS//\//--}" ] || need_gb=12
  local free_kb; free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  if [ -n "$free_kb" ] && [ "$free_kb" -lt $((need_gb * 1024 * 1024)) ]; then
    die "only $((free_kb / 1024 / 1024)) GB free under $ROOT; need about ${need_gb} GB"
  fi
}

# ---------------------------------------------------------------------------------- python side
ensure_python() {
  local UV
  if ! UV="$(find_uv)"; then
    info "Installing uv (Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$(find_uv)" || die "uv install failed; see https://docs.astral.sh/uv/getting-started/installation/"
  fi
  if [ ! -x "$PY" ]; then
    info "Creating the Python 3.12 virtualenv (uv downloads Python 3.12 if needed)"
    "$UV" venv -p 3.12 "$VENV"
  fi
  local h; h="$(hash_of "$BACKEND/pyproject.toml")"
  if ! stamp_ok backend "$h" || ! "$PY" -c "import screwit, fastapi, uvicorn, open3d, cv2, trimesh, scipy, anthropic, imageio_ffmpeg, cadquery" >/dev/null 2>&1; then
    info "Installing the backend and its extras (a few minutes the first time)"
    "$UV" pip install -p "$PY" -e "$BACKEND[geometry,cad,agent,storage,dev]"
    stamp_set backend "$h"
  fi
  ok "backend dependencies"

  [ "${SCREWIT_SKIP_MODELS:-0}" = "1" ] && { warn "SCREWIT_SKIP_MODELS=1: skipping the model stack and weights"; return; }

  # torch (CPU build on Linux; the default macOS wheels already run on the CPU).
  if ! "$PY" -c "import torch, torchvision" >/dev/null 2>&1; then
    info "Installing torch $TORCH_VERSION (CPU)"
    if [ "$(uname -s)" = "Linux" ]; then
      "$UV" pip install -p "$PY" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url https://download.pytorch.org/whl/cpu
    else
      "$UV" pip install -p "$PY" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    fi
  fi
  ok "torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"

  # Depth Anything 3 from its repo at the pinned commit (it is not on PyPI).
  local da3="$SRC_DIR/Depth-Anything-3"
  if [ ! -d "$da3/.git" ]; then
    info "Cloning Depth Anything 3"
    mkdir -p "$SRC_DIR"
    git clone --quiet "$DA3_REPO" "$da3"
  fi
  if [ "$(git -C "$da3" rev-parse HEAD 2>/dev/null)" != "$DA3_COMMIT" ]; then
    git -C "$da3" fetch --quiet origin "$DA3_COMMIT" 2>/dev/null || git -C "$da3" fetch --quiet origin
    git -C "$da3" -c advice.detachedHead=false checkout --quiet "$DA3_COMMIT"
  fi
  if ! "$PY" -c "import depth_anything_3.api" >/dev/null 2>&1; then
    info "Installing Depth Anything 3 and its runtime dependencies"
    # --no-deps: its own requirements pin a CUDA xformers and a second OpenCV; the code runs without.
    "$UV" pip install -p "$PY" -e "$da3" --no-deps
    "$UV" pip install -p "$PY" "${DA3_DEPS[@]}"
    "$PY" -c "import depth_anything_3.api" >/dev/null 2>&1 || die "Depth Anything 3 does not import; see $da3"
  fi
  ok "Depth Anything 3 @ ${DA3_COMMIT:0:7}"

  # Weights, once, into the HF cache the backend uses (public repo: no login needed).
  mkdir -p "$HF_DIR"
  if [ ! -d "$HF_DIR/hub/models--${DA3_WEIGHTS//\//--}/snapshots" ]; then
    info "Downloading the Depth Anything 3 weights (about 6.5 GB, once)"
    HF_HOME="$HF_DIR" "$PY" -c "from huggingface_hub import snapshot_download as d; d('$DA3_WEIGHTS')" \
      || die "weight download failed; re-run ./run.sh start to resume"
  fi
  ok "weights: $DA3_WEIGHTS"
}

# ----------------------------------------------------------------------------------- node side
ensure_node() {
  # gltfpack (meshopt compression for the phone-sized mesh); pinned in the toolchain package.
  if [ ! -x "$TOOLCHAIN/node_modules/.bin/gltfpack" ] || ! stamp_ok gltfpack "$(hash_of "$TOOLCHAIN/package-lock.json")"; then
    info "Installing gltfpack"
    ( cd "$TOOLCHAIN" && npm ci --silent ) && stamp_set gltfpack "$(hash_of "$TOOLCHAIN/package-lock.json")" \
      || warn "gltfpack install failed; scans still work, without phone-sized meshes"
  fi
  ok "gltfpack"

  local h; h="$(hash_of "$WEB/package-lock.json")"
  if [ ! -d "$WEB/node_modules" ] || ! stamp_ok web "$h"; then
    info "Installing web dependencies"
    ( cd "$WEB" && npm ci )
    stamp_set web "$h"
  fi
  ok "web dependencies"
  # Always rebuild so `start` serves the latest UI: `next start` serves a prebuilt snapshot and
  # would otherwise keep a stale .next after a pull. Next's cache makes a no-change build fast.
  if [ "${SCREWIT_SKIP_BUILD:-0}" = "1" ] && [ -d "$WEB/.next" ]; then
    warn "SCREWIT_SKIP_BUILD=1: serving the existing web build"
  else
    info "Building the web app"
    ( cd "$WEB" && npm run build >"$RUN_DIR/web-build.log" 2>&1 ) \
      || { tail -n 30 "$RUN_DIR/web-build.log"; die "web build failed (full log: $RUN_DIR/web-build.log)"; }
    ok "web build"
  fi
}

verify_stack() {
  "$PY" - <<'PY' || die "the backend stack does not import cleanly (see above)"
import importlib, importlib.metadata as md, os
mods = [("screwit.recon.pipeline", None), ("screwit.perception.service", None),
        ("open3d", "open3d"), ("cv2", None), ("trimesh", "trimesh"), ("anthropic", "anthropic")]
if os.environ.get("SCREWIT_SKIP_MODELS", "0") != "1":
    mods += [("torch", "torch"), ("depth_anything_3.api", None)]
for mod, dist in mods:
    importlib.import_module(mod)
    ver = md.version(dist) if dist else getattr(importlib.import_module(mod.split(".")[0]), "__version__", "")
    print(f"     {mod:22s} {ver}")
PY
  ok "stack verified"
}

ensure_installed() {
  mkdir -p "$DATA_DIR" "$HF_DIR" "$RUN_DIR"
  ensure_python
  ensure_node
  verify_stack
}

alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" >/dev/null 2>&1; }

# Pull the latest code so `start` serves the newest UI and backend. Fetches, then only
# fast-forwards; never merges, rebases or forces, and skips if the tree has local changes (so local
# work is never touched) or if offline. Disable with SCREWIT_NO_PULL=1.
update_repo() {
  [ "${SCREWIT_NO_PULL:-0}" = "1" ] && { warn "SCREWIT_NO_PULL=1: skipping git update"; return; }
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
  local branch; branch="$(git -C "$ROOT" symbolic-ref --short -q HEAD || echo "")"
  [ -n "$branch" ] || { warn "detached HEAD: skipping git update"; return; }
  if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    warn "uncommitted changes: skipping git update (serving your local code)"
    return
  fi
  info "Updating from origin/$branch (fast-forward only)"
  if git -C "$ROOT" fetch --quiet origin "$branch" 2>/dev/null; then
    if git -C "$ROOT" merge --ff-only "origin/$branch" >/dev/null 2>&1; then
      ok "at $(git -C "$ROOT" rev-parse --short HEAD)"
    else
      warn "local branch diverged from origin/$branch: not fast-forwarding; serving local code"
    fi
  else
    warn "could not reach origin (offline?): serving local code"
  fi
}

start() {
  if [ "${SCREWIT_LLM:-openrouter}" = "openrouter" ]; then
    : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY=sk-or-... first (never store it in the repo), or set SCREWIT_LLM=fake}"
  fi
  preflight
  # Always clear any previous run first, so `start` never collides with a stale server.
  stop_quiet
  update_repo
  ensure_installed

  info "Starting backend: http://127.0.0.1:${SCREWIT_PORT:-8000}  (device=${SCREWIT_DEVICE:-auto}, llm=${SCREWIT_LLM:-openrouter})"
  (
    cd "$ROOT"
    export SCREWIT_MODE=live
    export SCREWIT_DEVICE="${SCREWIT_DEVICE:-auto}"
    # Depth Anything 3 uses one op the Apple GPU lacks (bicubic upsampling); run it on the CPU
    # instead of failing. Must be set before torch loads, so it lives here, not in the code.
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    # Reuse Claude's answers for identical requests (a re-scan of the same video): faster, and the
    # same parts every time. Stored under the data dir in llm-cache/.
    export SCREWIT_LLM_CACHE="${SCREWIT_LLM_CACHE:-1}"
    export SCREWIT_CPU_VIEW_CAP="${SCREWIT_CPU_VIEW_CAP:-8}"
    export SCREWIT_INFER_GS="${SCREWIT_INFER_GS:-0}"
    export SCREWIT_PORT="${SCREWIT_PORT:-8000}"
    export SCREWIT_DATA_DIR="$DATA_DIR"
    export SCREWIT_LLM="${SCREWIT_LLM:-openrouter}"
    export SCREWIT_MODEL="${SCREWIT_MODEL:-anthropic/claude-sonnet-4.5}"
    export HF_HOME="$HF_DIR"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights are already local; never phone home
    export SCREWIT_RELOAD="${SCREWIT_RELOAD:-0}"
    exec "$PY" -m screwit.app
  ) >"$BE_LOG" 2>&1 &
  echo $! >"$BE_PID"

  info "Starting frontend: http://127.0.0.1:${WEB_PORT:-3000}"
  ( cd "$WEB" && PORT="${WEB_PORT:-3000}" SCREWIT_API_ORIGIN="http://127.0.0.1:${SCREWIT_PORT:-8000}" \
    npm run start >"$WEB_LOG" 2>&1 ) &
  echo $! >"$WEB_PID"

  # Wait for the backend to answer (model imports take a few seconds), up to 60 s.
  local i
  for i in $(seq 1 60); do
    curl -s -o /dev/null "http://127.0.0.1:${SCREWIT_PORT:-8000}/api/demo-scans" && break
    alive "$BE_PID" || break
    sleep 1
  done
  if curl -s -o /dev/null "http://127.0.0.1:${SCREWIT_PORT:-8000}/api/demo-scans"; then
    ok "backend up (pid $(cat "$BE_PID"), log $BE_LOG)"
  else
    err "backend failed to start; last lines of $BE_LOG:"; tail -n 20 "$BE_LOG" 2>/dev/null
  fi
  alive "$WEB_PID" && ok "frontend up (pid $(cat "$WEB_PID"), log $WEB_LOG)" || warn "frontend not running; see $WEB_LOG"
  ok "open http://127.0.0.1:${WEB_PORT:-3000}   (stop with ./run.sh stop)"
}

stop_quiet() {
  # Kill the processes from the pid files, then anything still holding our ports.
  local pid pf
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
