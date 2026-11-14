"""Depth Anything 3 on CPU. usage: run_da3.py <model-id> <n_views> [--gs] [--hd]"""
import sys, glob, json, argparse, traceback
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("n", type=int)
ap.add_argument("--gs", action="store_true"); ap.add_argument("--hd", action="store_true")
a = ap.parse_args()
import numpy as np, torch
torch.set_num_threads(32)
from depth_anything_3.api import DepthAnything3
from geom_check import convention_report, camera_centres
tag = a.model.split("/")[-1] + ("_gs" if a.gs else "") + ("_hd" if a.hd else "")
log = {"model": a.model, "n_views": a.n, "torch": torch.__version__, "cuda": torch.cuda.is_available()}
d = ROOT / "data" / ("hd" if a.hd else "scene")
imgs = sorted(glob.glob(str(d / "*.jpg")))[: a.n]
try:
    with Timer("load", log):
        model = DepthAnything3.from_pretrained(a.model).to(device=torch.device("cpu"))
    kw = {}
    if a.gs:
        edir = OUT / f"{tag}_export"
        kw = dict(infer_gs=True, ref_view_strategy="middle")
    with Timer("inference", log):
        pred = model.inference(imgs, **kw)
    if a.gs:
        try:
            with Timer("export_call_full_inference", log):
                model.inference(imgs, export_dir=str(edir), export_format="mini_npz-gs_ply",
                                export_kwargs={"gs_ply": {"gs_views_interval": 4}}, **kw)
            log["export_call"] = "ok"
        except Exception as e:
            log["export_call"] = "error: " + repr(e)[:500]
            log["export_trace"] = traceback.format_exc()[-2500:]
    log["fields"] = {k: (list(v.shape) if hasattr(v, "shape") else repr(v)) for k, v in vars(pred).items()
                     if k not in ("aux", "gaussians")}
    log["is_metric"] = pred.is_metric; log["scale_factor"] = pred.scale_factor
    log["aux_keys"] = None if pred.aux is None else list(pred.aux.keys())
    log["gaussians_attr"] = None if pred.gaussians is None else {k: list(v.shape) for k, v in vars(pred.gaussians).items()}
    log["extrinsics_dtype_shape"] = [str(pred.extrinsics.dtype), list(pred.extrinsics.shape)]
    log["depth_range"] = [float(np.nanmin(pred.depth)), float(np.nanmax(pred.depth))]
    log["conf_range"] = [float(pred.conf.min()), float(pred.conf.max())]
    log["intrinsics0"] = pred.intrinsics[0].tolist()
    log["extrinsics0"] = np.asarray(pred.extrinsics[0]).tolist()
    log["camera_centres"] = camera_centres(pred.extrinsics)
    log["input_size_wh"] = None
    from PIL import Image
    log["input_size_wh"] = list(Image.open(imgs[0]).size)
    log["model_grid_hw"] = list(pred.depth.shape[1:])
    log["convention_photo_error"] = convention_report(pred.processed_images, pred.depth, pred.conf, pred.extrinsics, pred.intrinsics)
    np.savez_compressed(OUT / f"{tag}_pred.npz", depth=pred.depth, conf=pred.conf, ext=pred.extrinsics, K=pred.intrinsics)
    if a.gs:
        g = pred.gaussians
        if g is not None:
            log["gaussian_means_range"] = [float(g.means.min()), float(g.means.max())]
            log["gaussian_scales_range"] = [float(g.scales.min()), float(g.scales.max())]
            log["gaussian_quat_norm_mean"] = float(g.rotations.norm(dim=-1).mean())
            log["gaussian_opacity_range"] = [float(g.opacities.min()), float(g.opacities.max())]
            log["gaussian_harmonics_shape"] = list(g.harmonics.shape)
        log["export_files"] = sorted(str(p.relative_to(OUT)) for p in edir.rglob("*"))
        np.save(OUT / f"{tag}_depth_range.npy", np.array(log["depth_range"]))
    log["status"] = "ok"
except Exception as e:
    log["status"] = "error"; log["error"] = repr(e); log["trace"] = traceback.format_exc()[-3000:]
log["total_peak_rss_gb"] = round(peak_rss_gb(), 2)
save(f"da3_{tag}.json", log)
print(json.dumps({k: v for k, v in log.items() if k not in ("trace",)}, indent=1, default=str)[:4000])
