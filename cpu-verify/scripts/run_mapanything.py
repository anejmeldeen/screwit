import glob, json, traceback, numpy as np, torch
from common import *
from geom_check import convention_report, camera_centres
torch.set_num_threads(32)
log = {"model": "facebook/map-anything-apache", "torch": torch.__version__}
try:
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    imgs = sorted(glob.glob(str(ROOT / "data/scene/*.jpg")))[:8]
    with Timer("load", log):
        model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cpu")
    views = load_images(imgs)
    log["view_keys"] = sorted(views[0].keys())
    log["view0_img_shape"] = list(views[0]["img"].shape)
    with Timer("infer", log):
        preds = model.infer(views, memory_efficient_inference=True, use_amp=False, apply_mask=True,
                            mask_edges=True, apply_confidence_mask=False)
    p0 = preds[0]
    log["n_preds"] = len(preds)
    log["pred_fields"] = {k: (list(v.shape) if hasattr(v, "shape") else repr(v)) for k, v in p0.items()}
    c2w = np.stack([p["camera_poses"][0].cpu().numpy() for p in preds])
    K = np.stack([p["intrinsics"][0].cpu().numpy() for p in preds])
    depth = np.stack([p["depth_z"][0, ..., 0].cpu().numpy() for p in preds])
    conf = np.stack([p["conf"][0].cpu().numpy() for p in preds])
    img = np.stack([(p["img_no_norm"][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8) for p in preds])
    log["metric_scaling_factor"] = [float(np.asarray(p["metric_scaling_factor"].cpu()).ravel()[0]) for p in preds]
    log["depth_range"] = [float(np.nanmin(depth)), float(np.nanmax(depth))]
    log["intrinsics0"] = K[0].tolist(); log["camera_poses0"] = c2w[0].tolist()
    w2c = np.linalg.inv(c2w)
    log["camera_centres_from_c2w_translation"] = c2w[:, :3, 3].tolist()
    log["convention_photo_error_if_poses_inverted(w2c=inv(camera_poses))"] = convention_report(img, depth, conf, w2c, K)
    log["convention_photo_error_if_poses_used_directly_as_w2c"] = convention_report(img, depth, conf, c2w, K)
    log["status"] = "ok"
except Exception as e:
    log["status"] = "error"; log["error"] = repr(e); log["trace"] = traceback.format_exc()[-3000:]
log["total_peak_rss_gb"] = round(peak_rss_gb(), 2)
save("mapanything.json", log)
print(json.dumps(log, indent=1, default=str)[:5000])
