#!/usr/bin/env bash
# Idempotent GPU-host setup for the Screw It verification kit.
# Run on the GPU host (Deep Learning Base GPU AMI). Safe to re-run.
#
# Sources for every install step (read 2026-11-14):
#   Depth Anything 3  https://github.com/ByteDance-Seed/Depth-Anything-3#-installation   (xformers, torch>=2, pip install -e ., gsplat pinned commit with --no-build-isolation)
#   MapAnything       https://github.com/facebookresearch/map-anything#installation        (python 3.12, pip install -e ., torch not pinned by the project)
#   SAM 2             https://github.com/facebookresearch/sam2#installation               (pip install -e ., checkpoint URLs in its README)
#   Grounded SAM 2    https://github.com/IDEA-Research/Grounded-SAM-2                      (HF path: grounded_sam2_hf_model_demo.py loads IDEA-Research/grounding-dino-tiny via transformers)
set -euo pipefail

# ---- pins (change deliberately, then re-run) --------------------------------
UV_VERSION="0.13.8"                                   # PyPI
PYTHON_VERSION="3.12"
TORCH_VERSION="2.13.0"                                # download.pytorch.org/whl/cu126 lists torch 2.13.0+cu126 cp312
TORCHVISION_VERSION="0.28.0"                          # pairs with torch 2.13 (same index)
CUDA_TAG="${CUDA_TAG:-cu126}"                         # wheel flavour; must be <= the driver's CUDA version
XFORMERS_VERSION="0.0.36"                             # PyPI
TRANSFORMERS_VERSION="4.57.6"                         # 4.x: post_process_grounded_object_detection(threshold=...) as in the Grounded-SAM-2 demo
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"        # ByteDance-Seed/Depth-Anything-3 main on 2026-11-14
GSPLAT_COMMIT="0b4dddf04cb687367602c01196913cde6a743d70"     # named in the DA3 README and pyproject (gs extra)
MAPANYTHING_COMMIT="3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9" # facebookresearch/map-anything main on 2026-11-14
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"        # facebookresearch/sam2 main on 2026-11-14
SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"  # SAM 2 README, SAM 2.1 large
SAM2_CKPT_NAME="sam2.1_hiera_large.pt"
HF_MODELS=("depth-anything/DA3NESTED-GIANT-LARGE-1.1" "facebook/map-anything-apache" "IDEA-Research/grounding-dino-tiny")

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${SCREWIT_GPU_CACHE:-$HOME/screwit-gpu-cache}"
SRC="$CACHE/src"
VENV="$CACHE/venv"
WEIGHTS="$CACHE/weights"
export HF_HOME="$CACHE/hf"

die() { echo "setup.sh: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. This must run on a CUDA GPU host (Deep Learning Base GPU AMI on g5.xlarge). Nothing was installed."
nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi found but reports no GPU or the driver is not loaded."
command -v git >/dev/null 2>&1 || die "git is required."
command -v curl >/dev/null 2>&1 || die "curl is required."

mkdir -p "$SRC" "$WEIGHTS" "$HF_HOME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"

# ---- uv ---------------------------------------------------------------------
UV="$CACHE/uv-bin/uv"
if [ ! -x "$UV" ] || [ "$("$UV" --version | awk '{print $2}')" != "$UV_VERSION" ]; then
  python3 -m venv "$CACHE/uv-bin-venv"
  "$CACHE/uv-bin-venv/bin/pip" install --quiet "uv==$UV_VERSION"
  mkdir -p "$CACHE/uv-bin"
  ln -sf "$CACHE/uv-bin-venv/bin/uv" "$UV"
fi

# ---- python 3.12 venv ---------------------------------------------------------
[ -x "$VENV/bin/python" ] || "$UV" venv --python "$PYTHON_VERSION" "$VENV"
PY="$VENV/bin/python"
PIP=("$UV" pip install --python "$PY")

# ---- pinned source checkouts --------------------------------------------------
checkout() { # name url commit
  local dir="$SRC/$1"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$3" ]; then return 0; fi
  rm -rf "$dir" && mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$2"
  git -C "$dir" fetch -q --depth 1 origin "$3"
  git -C "$dir" checkout -q FETCH_HEAD
  [ "$(git -C "$dir" rev-parse HEAD)" = "$3" ] || die "checkout of $1 did not land on $3"
}
checkout da3 https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_COMMIT"
checkout map-anything https://github.com/facebookresearch/map-anything.git "$MAPANYTHING_COMMIT"
checkout sam2 https://github.com/facebookresearch/sam2.git "$SAM2_COMMIT"

# ---- torch for this CUDA flavour ---------------------------------------------
"${PIP[@]}" "torch==${TORCH_VERSION}+${CUDA_TAG}" "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---- DA3 (README order: xformers, then -e ., then gsplat with --no-build-isolation) ----
"${PIP[@]}" "xformers==${XFORMERS_VERSION}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
"${PIP[@]}" -e "$SRC/da3"

# gsplat compiles CUDA code and needs nvcc matching the torch CUDA flavour.
want="${CUDA_TAG#cu}"; want="${want:0:2}.${want:2}"
if [ -z "${CUDA_HOME:-}" ]; then
  for c in "/usr/local/cuda-$want" /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && CUDA_HOME="$c" && break; done
fi
[ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ] || die "nvcc for CUDA $want not found. Set CUDA_HOME to a CUDA $want toolkit (the DLAMI ships /usr/local/cuda-*), or set CUDA_TAG to a flavour that matches an installed toolkit."
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # 8.6 = A10G
"${PIP[@]}" --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_COMMIT}"

# ---- MapAnything, SAM 2, Grounding DINO (HF) -------------------------------------
"${PIP[@]}" -e "$SRC/map-anything"
"${PIP[@]}" -e "$SRC/sam2"
"${PIP[@]}" "transformers==${TRANSFORMERS_VERSION}" "imageio-ffmpeg" "pillow" "huggingface_hub"

# ---- weights -----------------------------------------------------------------------
if [ ! -s "$WEIGHTS/$SAM2_CKPT_NAME" ]; then
  curl -fL --retry 3 -o "$WEIGHTS/$SAM2_CKPT_NAME.part" "$SAM2_CKPT_URL"
  mv "$WEIGHTS/$SAM2_CKPT_NAME.part" "$WEIGHTS/$SAM2_CKPT_NAME"
fi
for m in "${HF_MODELS[@]}"; do
  "$PY" - "$m" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(repo)
print(f"weights ready: {repo} -> {path}")
PYEOF
done

# ---- record exactly what got installed ---------------------------------------------
mkdir -p "$KIT_DIR/out"
"$UV" pip freeze --python "$PY" > "$KIT_DIR/out/pip-freeze.txt"
sha256sum "$WEIGHTS/$SAM2_CKPT_NAME" > "$KIT_DIR/out/weights.sha256"
{
  echo "da3=$DA3_COMMIT gsplat=$GSPLAT_COMMIT map-anything=$MAPANYTHING_COMMIT sam2=$SAM2_COMMIT"
  echo "torch=$TORCH_VERSION+$CUDA_TAG torchvision=$TORCHVISION_VERSION xformers=$XFORMERS_VERSION transformers=$TRANSFORMERS_VERSION"
} > "$KIT_DIR/out/pins.txt"

echo
echo "Setup complete. Run:"
echo "  SCREWIT_GPU_CACHE=$CACHE HF_HOME=$HF_HOME $PY $KIT_DIR/smoke.py <video.mp4>"
