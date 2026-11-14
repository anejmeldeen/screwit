"""SAM 3 text-prompted segmentation adapter (P3), GPU-gated, defensive.

SAM 3 adds native "promptable concept segmentation" (PCS): you give it a NOUN PHRASE ("ear cup",
"headband") and it returns masks for that concept directly -- no separate Grounding DINO box step.
That makes it the natural successor to the Grounding-DINO + SAM 2 pipeline in ``part_segmenter.py``:
Claude's part names (or a chat message) become text prompts straight into SAM 3.

Why this is a separate, opt-in path rather than the default:
- **Weights are gated.** SAM 3 checkpoints are a Meta Hugging Face gated download under the SAM
  License (research / non-commercial terms); a user must accept the license and authenticate.
- **GPU-recommended, not strictly required.** Meta's stock ``sam3`` package hard-fails on CPU
  (CUDA-only Triton NMS in ``perflib``, unconditional ``.cuda()``, a ``device.type == "cuda"``
  assert). The model itself is not intrinsically GPU-bound -- the third-party ``sam3-cpu`` wrapper
  runs it on CPU/Apple Silicon -- but on CPU it is much slower. So we DEFAULT OFF and only engage
  when explicitly enabled, so a CPU dev box is never silently dragged into a slow/erroring path.

Backends, auto-detected at call time (first that imports wins), all behind this boundary:
1. Hugging Face ``transformers`` ``Sam3VideoModel`` / ``Sam3Model`` (Meta-maintained, clean device
   args, no import-name clash) -- preferred.
2. ``sam3-cpu`` (rhubarb-ai): ``from sam3 import Sam3`` with ``process_video_with_prompts`` -- the
   CPU-first option; note it shadows the import name of Meta's official ``sam3`` package.
3. Meta official ``sam3``: ``build_sam3_video_predictor`` / ``handle_request`` -- GPU path.

Enable with ``SCREWIT_SAM3=1``. Config:
- SCREWIT_SAM3_MODEL     : HF model id or local path (default ``facebook/sam3``).
- SCREWIT_SAM3_BACKEND   : force ``hf`` | ``sam3cpu`` | ``meta`` (else auto-detect).
- SCREWIT_DEVICE         : cpu/cuda/mps (via adapters.resolve_device()).

Every method is best-effort and feature-detected: method/argument names across the three backends
are not all publicly pinned, so we probe for them and return None (never raise) when a backend is
absent, the weights are not authenticated, or the API shape does not match. The caller then keeps
the SAM 2 path or the single-part fallback.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SAM3_MODEL = "facebook/sam3"


def enabled() -> bool:
    """SAM 3 is opt-in (gated weights + GPU-recommended); off unless SCREWIT_SAM3 is truthy."""
    return os.environ.get("SCREWIT_SAM3", "0").strip().lower() in ("1", "true", "on", "yes")


def available() -> tuple[bool, str]:
    """(usable, backend) where backend is 'hf' | 'sam3cpu' | 'meta' | ''. usable is False when SAM 3
    is disabled or no backend imports. Import errors are swallowed -- absence is normal."""
    if not enabled():
        return False, ""
    forced = os.environ.get("SCREWIT_SAM3_BACKEND", "").strip().lower()
    order = [forced] if forced else ["hf", "sam3cpu", "meta"]
    for backend in order:
        if backend == "hf" and _can_import("transformers"):
            # Only claim HF if it actually exposes a SAM 3 class (older transformers will not).
            try:
                import transformers

                if any(hasattr(transformers, n) for n in ("Sam3VideoModel", "Sam3Model")):
                    return True, "hf"
            except Exception:  # noqa: BLE001
                pass
        elif backend == "sam3cpu" and _can_import("sam3"):
            try:
                import sam3

                if hasattr(sam3, "Sam3"):
                    return True, "sam3cpu"
            except Exception:  # noqa: BLE001
                pass
        elif backend == "meta" and _can_import("sam3"):
            try:
                import sam3

                if hasattr(sam3, "build_sam3_video_predictor"):
                    return True, "meta"
            except Exception:  # noqa: BLE001
                pass
    return False, ""


def _can_import(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001
        return False


class Sam3TextSegmenter:
    """Text-prompted video segmentation via SAM 3, across whichever backend is installed.

    ``segment(frames, phrase)`` returns ``{frame_idx: HxW bool mask}`` for the single best instance
    of ``phrase``, mirroring ``Sam2VideoAdapter.segment`` so it can drop into the same caller. Any
    failure returns ``{}`` (never raises)."""

    def __init__(self, model: str | None = None, device: str | None = None,
                 backend: str | None = None) -> None:
        self.model_id = model or os.environ.get("SCREWIT_SAM3_MODEL", DEFAULT_SAM3_MODEL)
        from screwit.perception.adapters import resolve_device

        self.device = device or resolve_device()
        self._backend = backend
        self._impl = None

    def _resolve_backend(self) -> str | None:
        if self._backend:
            return self._backend
        ok, backend = available()
        self._backend = backend if ok else None
        return self._backend

    def segment(self, frames: list[np.ndarray], phrase: str) -> dict[int, np.ndarray]:
        backend = self._resolve_backend()
        if backend is None or not phrase:
            return {}
        try:
            if backend == "hf":
                return self._segment_hf(frames, phrase)
            if backend == "sam3cpu":
                return self._segment_sam3cpu(frames, phrase)
            if backend == "meta":
                return self._segment_meta(frames, phrase)
        except Exception as e:  # noqa: BLE001 - any backend/API/auth failure -> empty, caller falls back
            log.warning("SAM 3 (%s) failed for %r: %s", backend, phrase, e)
        return {}

    # -- backends ----------------------------------------------------------

    def _load_hf(self):
        if self._impl is None:
            import transformers

            cls = getattr(transformers, "Sam3VideoModel", None) or getattr(transformers, "Sam3Model")
            proc_cls = getattr(transformers, "Sam3Processor", None) or getattr(transformers, "AutoProcessor")
            model = cls.from_pretrained(self.model_id).to(self.device)
            processor = proc_cls.from_pretrained(self.model_id)
            self._impl = (model, processor)
        return self._impl

    def _segment_hf(self, frames, phrase) -> dict[int, np.ndarray]:
        """HF transformers path. Method/arg names are probed because the SAM 3 API is new and may
        shift between transformers versions; we try the documented shape and bail cleanly if it
        does not match."""
        import torch
        from PIL import Image

        model, processor = self._load_hf()
        out: dict[int, np.ndarray] = {}
        # Per-frame image segmentation with a text prompt, assembled into a per-frame dict. (A true
        # video predictor would propagate; per-frame keeps the adapter simple and backend-agnostic.)
        for i, img in enumerate(frames):
            pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
            inputs = processor(images=pil, text=phrase, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            masks = processor.post_process_instance_segmentation(
                outputs, target_sizes=[pil.size[::-1]])
            m = _best_mask_from_hf(masks, pil.size[::-1])
            if m is not None:
                out[i] = m
        return out

    def _segment_sam3cpu(self, frames, phrase) -> dict[int, np.ndarray]:
        """rhubarb-ai sam3-cpu path: Sam3().process_image_with_prompts over the frame images.

        Uses the IMAGE API with a list of per-frame paths (the video API wants a real video file).
        Each frame is segmented independently by the text prompt; masks are keyed by frame index."""
        import tempfile

        from PIL import Image

        if self._impl is None:
            from sam3 import Sam3

            self._impl = Sam3()
        sam3 = self._impl
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, img in enumerate(frames):
                p = os.path.join(d, f"{i:05d}.jpg")
                Image.fromarray(np.asarray(img, dtype=np.uint8)).save(p)
                paths.append(p)
            result = sam3.process_image_with_prompts(image_path=paths, prompts=[phrase],
                                                     output_dir=d, device=self.device)
            return _per_frame_from_sam3cpu(result, len(frames))

    def _segment_meta(self, frames, phrase) -> dict[int, np.ndarray]:
        """Meta official sam3 path (GPU). build_sam3_video_predictor + a text request. Names are
        probed defensively; this path is expected only on a CUDA host with gated weights."""
        import tempfile

        import torch
        from PIL import Image

        if self._impl is None:
            import sam3

            self._impl = sam3.build_sam3_video_predictor(self.model_id, device=self.device)
        predictor = self._impl
        with tempfile.TemporaryDirectory() as d:
            for i, img in enumerate(frames):
                Image.fromarray(np.asarray(img, dtype=np.uint8)).save(os.path.join(d, f"{i:05d}.jpg"))
            with torch.inference_mode():
                state = predictor.init_state(d)
                # Text/concept prompt entry point; name probed across likely variants.
                add = (getattr(predictor, "add_text_prompt", None)
                       or getattr(predictor, "add_new_text", None)
                       or getattr(predictor, "set_text_prompt", None))
                if add is None:
                    return {}
                add(state, text=phrase)
                out: dict[int, np.ndarray] = {}
                for fidx, _obj_ids, logits in predictor.propagate_in_video(state):
                    out[fidx] = (logits[0] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                return out


def _best_mask_from_hf(masks, hw) -> np.ndarray | None:
    """Pull one HxW bool mask (the top-scoring instance) from a probed HF post-processed result."""
    if not masks:
        return None
    first = masks[0] if isinstance(masks, (list, tuple)) else masks
    seg = first.get("segmentation") if isinstance(first, dict) else getattr(first, "masks", None)
    if seg is None:
        return None
    arr = seg.detach().cpu().numpy() if hasattr(seg, "detach") else np.asarray(seg)
    if arr.ndim == 3:  # (instances, H, W) -> take the first/best
        arr = arr[0]
    if arr.shape != tuple(hw):
        return None
    return arr.astype(bool)


def _per_frame_from_sam3cpu(result, n_frames: int) -> dict[int, np.ndarray]:
    """Map a sam3-cpu process_image_with_prompts result to {frame_idx: bool mask}.

    Shape (verified against the installed sam3-cpu): result["images"] is a list, one per input frame
    in order; each has ["prompts"][<phrase>] with "num_objects", "object_ids", and "masks_dir" -- a
    directory of per-object masks written as object_<id>_mask.npy (bool) + .png. We OR all objects
    for the prompt into one mask per frame."""
    import os

    if not isinstance(result, dict):
        return {}
    images = result.get("images")
    if not isinstance(images, list):
        return {}
    out: dict[int, np.ndarray] = {}
    for idx, img in enumerate(images):
        if idx >= n_frames or not isinstance(img, dict):
            continue
        prompts = img.get("prompts") or {}
        if not isinstance(prompts, dict):
            continue
        combined = None
        for pr in prompts.values():
            masks_dir = pr.get("masks_dir") if isinstance(pr, dict) else None
            if not masks_dir or not os.path.isdir(masks_dir):
                continue
            for fn in os.listdir(masks_dir):
                if not fn.endswith(".npy"):
                    continue
                try:
                    m = np.load(os.path.join(masks_dir, fn)).astype(bool)
                except Exception:  # noqa: BLE001
                    continue
                if m.ndim != 2:
                    continue
                combined = m if combined is None else (combined | m)
        if combined is not None and combined.any():
            out[idx] = combined
    return out
