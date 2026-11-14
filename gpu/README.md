# GPU verification kit (X1)

The dev machine has no GPU, so the real model calls are untested there. This kit runs on one CUDA host, proves Depth Anything 3 (DA3), MapAnything, SAM 2 and Grounding DINO work on a real phone video, and writes numbers to compare with the CPU path.

Nothing in this kit launches cloud resources by itself. A human runs the steps below in a **non-production** AWS account.

Files: `setup.sh` (host setup), `smoke.py` (the measurements), `compare.py` (scan vs caliper), `tests/` (CPU-only tests).

## 0. Licence gate

`DA3NESTED-GIANT-LARGE-1.1` is CC BY-NC 4.0 (DA3 README model table). Confirm with the organisers before relying on it (SPEC 1.5). MapAnything's `facebook/map-anything-apache` is the Apache-2.0 alternative.

## 1. Launch the host (about 15 minutes)

Cost: I could not read a price off an AWS page (the g5 product page and calculator render prices dynamically). Third-party trackers read on 2026-11-14 list g5.xlarge Linux on-demand in us-east-1 at about **$1.006/hour** ([usage.ai](https://instances.usage.ai/aws/ec2/g5.xlarge), [DoiT](https://www.doit.com/compute/spot/us-east-1/g5.xlarge)). Verify on the [EC2 on-demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) before launching. A full run is about 1 to 2 hours including setup, so budget a few dollars plus EBS.

g5.xlarge is 1x NVIDIA A10G, 24 GB GPU memory, 4 vCPU, 16 GiB RAM ([AWS G5 page](https://aws.amazon.com/ec2/instance-types/g5/)). 16 GiB host RAM is tight for the 1.4B-parameter DA3 model load; if loading is killed by the OOM killer use g5.2xlarge (32 GiB, same GPU).

AMI: **Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)**, listed as "Base GPU AMI (Amazon Linux 2023)" in the [DLAMI release notes](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html). Its latest ID is published in SSM Parameter Store under the name format described in [Finding the ID of a DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html). The Ubuntu 24.04 Base GPU AMI from the same list also works; `setup.sh` is distro-neutral (it needs `git`, `curl`, `python3 -m venv`).

Set `REGION`, `SUBNET_ID` (a private subnet with **outbound** internet through a NAT gateway, needed for GitHub, PyPI, PyTorch and Hugging Face downloads) and `BUCKET` (a private bucket you own).

```bash
export AWS_REGION=us-east-1 SUBNET_ID=subnet-xxxx BUCKET=my-screwit-gpu-kit
AMI=$(aws ssm get-parameter --region $AWS_REGION \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id \
  --query Parameter.Value --output text)

# role with SSM only, plus read/write on one bucket prefix
aws iam create-role --role-name screwit-gpu-kit --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name screwit-gpu-kit --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name screwit-gpu-kit --policy-name kit-bucket --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/gpu-kit/*\"}]}"
aws iam create-instance-profile --instance-profile-name screwit-gpu-kit
aws iam add-role-to-instance-profile --instance-profile-name screwit-gpu-kit --role-name screwit-gpu-kit

# security group with NO inbound rules (SSM Session Manager needs none)
VPC_ID=$(aws ec2 describe-subnets --subnet-ids $SUBNET_ID --query 'Subnets[0].VpcId' --output text)
SG=$(aws ec2 create-security-group --group-name screwit-gpu-kit --description "no inbound" \
  --vpc-id $VPC_ID --query GroupId --output text)

aws ec2 run-instances --image-id $AMI --instance-type g5.xlarge \
  --subnet-id $SUBNET_ID --security-group-ids $SG --no-associate-public-ip-address \
  --iam-instance-profile Name=screwit-gpu-kit \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1 \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=screwit-gpu-kit}]'
```

Checks before continuing: the security group has no ingress rules; the instance has no public IP; `HttpTokens` is `required` (IMDSv2); the root volume is encrypted. A brand-new instance profile can take a minute to propagate. Confirm the instance appears under Systems Manager > Fleet Manager before connecting. The 200 GB root volume is a generous guess for torch, three model families and extracted frames; actual use has not been measured.

Confirm the AMI name you resolved: `aws ec2 describe-images --image-ids $AMI --query 'Images[0].Name'`.

## 2. Put the repo and the video on the host

Only `gpu/` is needed. From the repo root on your machine:

```bash
tar czf /tmp/gpu-kit.tgz gpu --exclude=gpu/out
aws s3 cp /tmp/gpu-kit.tgz s3://$BUCKET/gpu-kit/gpu-kit.tgz
aws s3 cp my-phone-video.mp4 s3://$BUCKET/gpu-kit/video.mp4
```

Connect with Session Manager (needs the Session Manager plugin; no SSH, no open port):

```bash
aws ssm start-session --target i-xxxxxxxx
```

On the host:

```bash
sudo -iu ec2-user
mkdir -p ~/screwit && cd ~/screwit
aws s3 cp s3://BUCKET/gpu-kit/gpu-kit.tgz . && tar xzf gpu-kit.tgz
aws s3 cp s3://BUCKET/gpu-kit/video.mp4 .
```

(Replace `BUCKET` with the real name; the host shell does not have your exports.)

## 3. Run

```bash
cd ~/screwit
bash gpu/setup.sh 2>&1 | tee setup.log      # 15 to 40 minutes, mostly downloads and the gsplat build
~/screwit-gpu-cache/venv/bin/python gpu/smoke.py video.mp4 --run first
```

`setup.sh` fails at once if `nvidia-smi` is missing, and fails with a message if no CUDA toolkit matching the torch wheel (default `cu126`, override with `CUDA_TAG=cu128`) is installed for the gsplat build. It records the installed set in `gpu/out/pip-freeze.txt`, `gpu/out/pins.txt` and `gpu/out/weights.sha256`.

Useful options: `--views 32,48,64` (default), `--stages da3,mapanything,sam2,gdino`, `--prompt "pipe."`, `--board-spec board.json` (recorded in the report only), `--vram-threshold-mb 1024`.

Outputs go to `gpu/out/<run>/`: `report.json`, `report.md`, per-run `da3_<n>/outputs.npz` (+ `export/` with `gs_ply`), `mapanything_<n>/outputs.npz`, the extracted frames.

What the run measures, and the numbers to read in `report.md`:

| Stage | Settings | Recorded |
|---|---|---|
| DA3 `DA3NESTED-GIANT-LARGE-1.1` at 32/48/64 views | `infer_gs=True`, `ref_view_strategy="middle"`, `export_format="mini_npz-gs_ply"`, `gs_views_interval: 4` | peak `torch.cuda.max_memory_allocated`, wall time, output shapes, `scale_factor`, whether gaussians are on `prediction.gaussians` or under `aux`, extrinsics orbit check, OOM status |
| Free DA3 | drop gaussians/aux, delete model, `gc`, `empty_cache` | allocated before/after; the run errors if above `--vram-threshold-mb` |
| MapAnything `facebook/map-anything-apache` at the same view counts | `memory_efficient_inference=True` | same, plus `metric_scaling_factor`; poses are cam-to-world and inverted before the orbit check |
| SAM 2 video predictor | centre box prompt on frame 0, propagated | memory, time, mask area fraction |
| Grounding DINO (`IDEA-Research/grounding-dino-tiny`) | text prompt on the middle frame | memory, time, detections |

Extrinsics check: with world-to-camera `[R|t]` the camera centre is `-R^T t`. The report gives orbit metrics (radius spread, plane thickness, angular span, and how close the optical axes converge) for that reading and for the opposite reading (matrices are camera-to-world). `verdict: world_to_camera_consistent` means only the first reading looks like a phone orbit.

Compare with caliper readings (names are yours; values in mm):

```bash
echo '{"od_mm": 31.8, "length_mm": 100.2}' > caliper.json
echo '{"od_mm": 33.1, "length_mm": 98.9}'  > scan.json     # from the repo's measurement code or by hand
~/screwit-gpu-cache/venv/bin/python gpu/compare.py gpu/out/first/report.json --caliper caliper.json --scan-mm scan.json
```

## 4. Copy results back and terminate

```bash
# on the host
cd ~/screwit && tar czf results.tgz gpu/out setup.log
aws s3 cp results.tgz s3://BUCKET/gpu-kit/results.tgz
# on your machine
aws s3 cp s3://$BUCKET/gpu-kit/results.tgz . && tar xzf results.tgz
aws ec2 terminate-instances --instance-ids i-xxxxxxxx
```

Then delete the bucket objects, the instance profile, the role and the security group if they were created only for this. Terminating is what stops the hourly charge; stopping leaves EBS billing.

## What the kit does not do

- It does not call `screwit.recon.frames`, `screwit.recon.scale` or `screwit.perception`: they were empty or had no confirmed entry point when this was written. `smoke.py` records whether `screwit.recon.frames` imports and uses its own ffmpeg extraction. `compare.py` lists whatever public callables those modules expose and computes scan-vs-caliper error from `--scan-mm`. Wire the real calls in once the entry points exist.
- Grounded SAM 2 is exercised as its Grounding DINO half (the Hugging Face path used by the Grounded-SAM-2 `grounded_sam2_hf_model_demo.py`) plus the SAM 2 video predictor. The Grounded-SAM-2 repo itself is not installed, which avoids compiling Grounding DINO's custom CUDA op.
- Nothing here has run on a GPU. See the unconfirmed list in the build report.

## Upstream APIs and the docs that confirm them

| API | Source |
|---|---|
| `DepthAnything3.from_pretrained(id)`, `model.inference(images, ...)`, `prediction.depth/conf/extrinsics/intrinsics/processed_images` | [DA3 README Quick Start](https://github.com/ByteDance-Seed/Depth-Anything-3#-basic-usage) |
| `infer_gs`, `ref_view_strategy` (`"middle"`), `export_dir`, `export_format` (`mini_npz`, `gs_ply`), `export_kwargs={"gs_ply": {"gs_views_interval": n}}` | [DA3 docs/API.md](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md) |
| `Prediction.gaussians`, `.aux`, `.scale_factor`, `.is_metric`; `Gaussians.means/scales/rotations/harmonics/opacities` | [DA3 src/depth_anything_3/specs.py](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/specs.py) (source; the README does not document them) |
| gsplat pinned commit `0b4dddf0...`, `--no-build-isolation` | DA3 README Installation and `pyproject.toml` |
| `MapAnything.from_pretrained`, `load_images`, `model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=True)`; outputs `camera_poses` (cam2world 4x4), `pts3d`, `depth_z`, `conf`, `intrinsics`, `metric_scaling_factor` | [MapAnything README](https://github.com/facebookresearch/map-anything#image-only-inference) |
| `build_sam2_video_predictor(cfg, ckpt)`, `init_state(video_path)`, `add_new_points_or_box(state, frame_idx, obj_id, box=...)`, `propagate_in_video(state)`; checkpoint URL | [SAM 2 README](https://github.com/facebookresearch/sam2#video-prediction) and `sam2/sam2_video_predictor.py` source |
| `AutoProcessor`, `AutoModelForZeroShotObjectDetection`, `post_process_grounded_object_detection(..., threshold, text_threshold, target_sizes)`; lowercase text ending in a dot | [Grounded-SAM-2 `grounded_sam2_hf_model_demo.py`](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_hf_model_demo.py) |
| DLAMI name and SSM parameter | [DLAMI release notes](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html), [Finding the ID of a DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html) |

## Tests that run without a GPU

```bash
M=/local/home/brancli/.kiro/crew/workspace/screw-it   # main tree with the shared venv
$M/backend/.venv/bin/python -m pytest gpu/tests -q
$M/backend/.venv/bin/ruff check gpu --config backend/pyproject.toml
$M/backend/.venv/bin/shellcheck gpu/setup.sh
$M/backend/.venv/bin/python gpu/smoke.py v.mp4 --dry-run
```
