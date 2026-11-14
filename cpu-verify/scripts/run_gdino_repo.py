"""Grounding DINO via the Grounded-SAM-2 repo's own grounding_dino package (grounded_sam2_local_demo.py path), CPU."""
import json, sys, traceback, torch
from common import *
sys.path.append(str(ROOT / "src/Grounded-SAM-2"))
log = {}
try:
    from groundingdino.util.inference import load_model, load_image, predict
    root = ROOT / "src/Grounded-SAM-2/grounding_dino"
    with Timer("load", log):
        m = load_model(str(root / "groundingdino/config/GroundingDINO_SwinT_OGC.py"), str(ROOT / ".cache/ckpt/groundingdino_swint_ogc.pth"), device="cpu")
    src, img = load_image(str(ROOT / "data/sam_frames/00000.jpg"))
    with Timer("infer", log):
        boxes, conf, phrases = predict(m, img, "bed. pillow.", 0.35, 0.25, device="cpu")
    log["boxes_cxcywh_normalized"] = boxes.tolist(); log["conf"] = conf.tolist(); log["phrases"] = phrases
    log["status"] = "ok"
except Exception as e:
    log["status"] = "error"; log["error"] = repr(e); log["trace"] = traceback.format_exc()[-3000:]
log["total_peak_rss_gb"] = round(peak_rss_gb(), 2)
save("gdino_repo.json", log); print(json.dumps(log, indent=1)[:3000])
