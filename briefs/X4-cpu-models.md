# X4 real models on the CPU (worktree screw-it-wt/cpu, branch ws/cpu)

Read: briefs/common.md, AGENTS.md, SPEC.md 1.5, 3.1 (camera contract, gaussians, DA3 nested scale_factor), 6.1 steps 1-6, 6.2 steps 1-2.

You own only cpu-verify/ (top-level). Create nothing elsewhere. Use a SEPARATE virtualenv at cpu-verify/.venv in your worktree (torch is large and must not go into the shared backend venv). Put Hugging Face and torch caches under cpu-verify/.cache in your worktree (gitignore it); do not use $KIROCREW_SCRATCH for weights.

Purpose: this host has no GPU, but PyTorch models run on the CPU, only slower. Prove, with the REAL weights and REAL images, that each model loads, runs, and produces outputs in the shapes and conventions our adapters assume. This replaces guesses with observed facts. No training, no fine-tuning, no per-scene optimization.

Install CPU torch from https://download.pytorch.org/whl/cpu, then each project from its official repo/README with pinned commits (cite each README). If a project hard-requires CUDA at import or build time, try its documented CPU path or env flag; if there is none, record the exact error and move on; do not patch upstream source.

Run, timing each (wall time, peak RSS) and saving outputs to cpu-verify/out/:
1. Depth Anything 3. Real images: the example images shipped in the DA3 repo (e.g. assets/examples/*). Start with da3-large on 4-8 images, then DA3NESTED-GIANT-LARGE-1.1 on 4-8 images. Record: prediction field names and shapes (depth, conf, extrinsics, intrinsics, processed_images), extrinsics convention (check world-to-camera: camera centres C = -R^T t; reproject a depth pixel of view A into view B and confirm it lands on the same image content), the model-grid size actually produced (does 1920x1080 input give 504x280?), prediction.scale_factor and is_metric for the nested model, and the nested-model quirk SPEC 3.1 describes (depth and extrinsic translations scaled, gaussians not). Then try infer_gs=True with export_format="mini_npz-gs_ply" and export_dir: gsplat may need CUDA; record whether the Gaussian head runs on CPU, where the gaussians live (prediction.gaussians or prediction.aux), the PLY field names and quaternion order in the written file.
2. MapAnything (facebook/map-anything-apache) on the same images with memory_efficient_inference=True: field names, camera_poses convention (camera-to-world, OpenCV) checked the same way, metric_scaling_factor.
3. SAM 2 (sam2.1-hiera-base-plus or small): image predictor with a box prompt and a negative point (label 0); video predictor over 8 frames with propagation; SAM2AutomaticMaskGenerator on one image. Record API names actually used.
4. Grounding DINO for text-to-box: via the Grounded-SAM-2 repo if its CPU path works, else the Hugging Face transformers port (IDEA-Research/grounding-dino-tiny) as the documented alternative; record which works on CPU.
5. If the repo modules exist when you finish a model (check /local/home/brancli/.kiro/crew/workspace/screw-it-wt/recon/backend/screwit/recon/geometry.py and .../perception/backend/screwit/perception/), read their adapter code and list every mismatch between what they assume and what you observed (names, shapes, conventions). Do not edit their code.

Write cpu-verify/README.md (how to reproduce, every pinned version/commit) and cpu-verify/FINDINGS.md: a table per model of what was assumed in SPEC.md/AGENTS.md versus observed, with pass/fail, plus timings, and an estimate of whether CPU inference is usable for the hackathon (seconds per view) as a fallback when no GPU is available. Commit the scripts, README and FINDINGS; do not commit weights or large outputs (keep committed files under 5 MB).

Stop when every model has either run with findings recorded or has a recorded, specific blocker. Report the FINDINGS summary.
