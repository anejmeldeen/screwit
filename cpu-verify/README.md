# cpu-verify: real models on the CPU (X4)

Runs the real Depth Anything 3, MapAnything, SAM 2 and Grounding DINO weights on a CPU-only host (32 cores,
123 GB RAM, Xeon 8488C, torch 2.5.1+cpu) to check what SPEC.md and the adapters assume against what the models
actually return. Results: `FINDINGS.md`. Raw machine-readable results: `out/*.json`.

## Reproduce

All paths are relative to this directory. Own virtualenv and caches; nothing touches the shared backend venv.

```bash
export UV_CACHE_DIR=$PWD/.cache/uv
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
mkdir src && cd src
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git   # 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
git clone --depth 1 https://github.com/facebookresearch/map-anything.git      # 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
git clone --depth 1 https://github.com/facebookresearch/sam2.git              # 2b90b9f5ceec907a1c18123530e92e794ad901a4
git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2.git       # b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
git clone https://github.com/nerfstudio-project/gsplat.git && git -C gsplat checkout 0b4dddf04cb687367602c01196913cde6a743d70
cd ..
U="uv pip install -p .venv/bin/python"
# Depth Anything 3 (README "Installation"): xformers is skipped (it pins a CUDA torch); the code runs without it.
$U -e src/Depth-Anything-3 --no-deps
$U trimesh einops huggingface_hub imageio "numpy<2" opencv-python open3d requests pillow omegaconf plyfile pillow_heif \
   safetensors "moviepy==1.0.3" e3nn imageio-ffmpeg psutil fastapi uvicorn typer evo pycolmap addict setuptools wheel ninja jaxtyping rich
# gsplat (DA3 README "gaussian head"): the CUDA build fails without CUDA_HOME, so build the python-only package.
BUILD_NO_CUDA=1 $U --no-build-isolation --no-deps src/gsplat
# MapAnything (README "Installation")
$U -e src/map-anything --no-deps
$U hydra-core natsort orjson python-box "uniception==0.1.7" tqdm timm "rerun-sdk~=0.24.1" tensorboard
# SAM 2 (README "Installation"): SAM2_BUILD_CUDA=0 skips the optional CUDA kernel
SAM2_BUILD_CUDA=0 $U -e src/sam2 --no-deps --no-build-isolation
$U iopath
# Grounding DINO: HF port (transformers must be < 5; 5.x needs torch >= 2.6) and the Grounded-SAM-2 repo package
$U "transformers==4.49.0" "tokenizers<0.22"
$U --no-build-isolation --no-deps -e src/Grounded-SAM-2/grounding_dino
$U supervision pycocotools yapf
# weights (HF downloads happen on first use into .cache/hf)
mkdir -p .cache/ckpt
curl -L -o .cache/ckpt/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
curl -L -o .cache/ckpt/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

`requirements.lock.txt` is the full `uv pip freeze` of the venv that produced the results.

Run (each script writes `out/*.json`; `scripts/common.py` points HF/torch caches at `.cache/`):

```bash
cd scripts
../.venv/bin/python prep_data.py                                              # real image sets (see below)
../.venv/bin/python run_da3.py depth-anything/DA3-LARGE 8
../.venv/bin/python run_da3.py depth-anything/DA3-LARGE 2 --hd                # 1920x1080 grid arithmetic
../.venv/bin/python run_da3.py depth-anything/DA3NESTED-GIANT-LARGE-1.1 8
../.venv/bin/python run_da3.py depth-anything/DA3NESTED-GIANT-LARGE-1.1 8 --gs   # Gaussian head + gs_ply export
../.venv/bin/python analyze_da3_gs.py; ../.venv/bin/python analyze_da3_gs2.py   # PLY layout, scale_factor quirk
../.venv/bin/python run_mapanything.py
../.venv/bin/python run_gdino_sam2.py                                         # HF Grounding DINO + SAM 2 image/video/auto
../.venv/bin/python run_gdino_repo.py                                         # Grounded-SAM-2 repo's own grounding_dino
../.venv/bin/python check_gdino_labels.py
```

## Test images

The DA3 repo ships only two example images (`assets/examples/SOH/*.png`), too few for pose and reprojection
checks, so all models use 8 real frames (every 25th frame, 960x540 JPEG, a moving handheld camera in a bedroom) of the
clip in the SAM 2 repo (`notebooks/videos/bedroom`). Camera baselines are small (a few percent of scene depth), so
the reprojection check discriminates conventions but is not a precision test. `data/hd/` upsamples two of them to
1920x1080 only to check the model-grid arithmetic. `data/` is git-ignored; `prep_data.py` rebuilds it.

## Pinned versions

torch 2.5.1+cpu, torchvision 0.20.1+cpu, numpy < 2, transformers 4.49.0, Python 3.12.16; commits above;
depth-anything-3 0.0.0 (commit above), mapanything 1.1.4, sam-2 1.0, gsplat 1.5.2 (python-only build), groundingdino 0.1.0.
Weights: `depth-anything/DA3-LARGE`, `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, `facebook/map-anything-apache`,
`sam2.1_hiera_base_plus.pt`, `IDEA-Research/grounding-dino-tiny`, `groundingdino_swint_ogc.pth`.
`DA3-LARGE` is a Hugging Face alias; the revision hashes are in `.cache/hf` (not committed).
