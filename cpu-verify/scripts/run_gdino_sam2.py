"""Grounding DINO (HF transformers port) text->box, then SAM 2.1 image/video/auto-mask on CPU."""
import glob, json, traceback, numpy as np, torch
from PIL import Image
from common import *
torch.set_num_threads(32)
log = {"torch": torch.__version__}
frames = sorted(glob.glob(str(ROOT / "data/sam_frames/*.jpg")))
img0 = Image.open(frames[0]).convert("RGB")
box = None
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import transformers; log["transformers"] = transformers.__version__
    with Timer("gdino_load", log):
        proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
    with Timer("gdino_infer", log):
        inp = proc(images=img0, text="a bed. a pillow.", return_tensors="pt")
        with torch.no_grad():
            o = gd(**inp)
        res = proc.post_process_grounded_object_detection(o, inp.input_ids, box_threshold=0.3, text_threshold=0.25,
                                                           target_sizes=[img0.size[::-1]])[0]
    log["gdino_keys"] = list(res.keys())
    log["gdino"] = [{"label": l, "score": float(s), "box_xyxy": [float(x) for x in b]}
                    for l, s, b in zip(res.get("text_labels", res.get("labels")), res["scores"], res["boxes"])]
    best = max(log["gdino"], key=lambda r: r["score"]) if log["gdino"] else None
    if best: box = np.array(best["box_xyxy"])
    log["gdino_status"] = "ok"
except Exception as e:
    log["gdino_status"] = "error"; log["gdino_error"] = repr(e); log["gdino_trace"] = traceback.format_exc()[-2000:]
if box is None:
    box = np.array([100, 150, 800, 520.0]); log["box_fallback"] = True
log["box_used"] = box.tolist()
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    ck = str(ROOT / ".cache/ckpt/sam2.1_hiera_base_plus.pt"); cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    with Timer("sam2_image_load", log):
        m = build_sam2(cfg, ck, device="cpu")
        pr = SAM2ImagePredictor(m)
    with Timer("sam2_set_image", log):
        pr.set_image(np.array(img0))
    neg = np.array([[img0.size[0] - 30, img0.size[1] - 30]])
    with Timer("sam2_predict_box_negpoint", log):
        masks, scores, logits = pr.predict(point_coords=neg, point_labels=np.array([0]), box=box[None], multimask_output=False)
    log["image_predict"] = {"masks": [list(masks.shape), str(masks.dtype)], "scores": scores.tolist(), "logits": list(logits.shape),
                            "mask_area_frac": float(masks[0].mean()), "neg_point_in_mask": bool(masks[0][neg[0][1], neg[0][0]])}
    Image.fromarray((masks[0] * 255).astype(np.uint8)).save(OUT / "sam2_image_mask.png")
    with Timer("sam2_video_load", log):
        vp = build_sam2_video_predictor(cfg, ck, device="cpu")
    with Timer("sam2_video_init_state", log):
        st = vp.init_state(video_path=str(ROOT / "data/sam_frames"))
    with Timer("sam2_video_add_prompt", log):
        fi, oids, ml = vp.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=box, points=neg, labels=np.array([0], np.int32))
    log["video_add_prompt_return"] = [fi, list(oids), list(ml.shape)]
    areas = {}
    with Timer("sam2_video_propagate_8", log):
        for f, oids, ml in vp.propagate_in_video(st):
            areas[f] = float((ml[0] > 0).float().mean())
    log["video_mask_area_frac_per_frame"] = areas
    with Timer("sam2_auto_masks", log):
        gen = SAM2AutomaticMaskGenerator(m, points_per_side=16)
        am = gen.generate(np.array(img0))
    log["auto_masks"] = {"n": len(am), "keys": list(am[0].keys()) if am else None}
    log["sam2_status"] = "ok"
except Exception as e:
    log["sam2_status"] = "error"; log["sam2_error"] = repr(e); log["sam2_trace"] = traceback.format_exc()[-3000:]
log["total_peak_rss_gb"] = round(peak_rss_gb(), 2)
save("gdino_sam2.json", log)
print(json.dumps(log, indent=1, default=str)[:6000])
