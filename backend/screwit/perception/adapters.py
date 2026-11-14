"""Model adapters for perception, with CPU fakes (SPEC: AGENTS.md toolchain, 6.2).

The real adapters lazy-import torch and the model packages, so importing this module never pulls
in torch; their real calls are the ones a GPU worker runs and carry ``@pytest.mark.gpu`` in the
tests. Every adapter also has a deterministic CPU fake driven by known synthetic labels, so the
whole pipeline is testable end to end on this machine.

API names verified against the official projects and against a real CPU run (cpu-verify/FINDINGS.md
section 5, items 13-16):
- SAM 2 (https://github.com/facebookresearch/sam2): ``build_sam2_video_predictor`` / ``build_sam2``
  default to ``device="cuda"``, so a device is passed explicitly (``SCREWIT_DEVICE``);
  ``predictor.init_state(video_dir)`` needs a directory of integer-named JPEGs, so frames are
  written as ``00000.jpg`` into a temp dir; ``add_new_points_or_box`` takes ``points``/``labels``
  (positive 1, negative 0, as an int32 array) and ``box``; ``propagate_in_video`` yields logits.
  ``SAM2AutomaticMaskGenerator`` uses 16 points/side on CPU, 32 on GPU.
- Grounded SAM 2 (https://github.com/IDEA-Research/Grounded-SAM-2): Grounding DINO (HF
  ``IDEA-Research/grounding-dino-tiny``) turns a period-separated lowercase prompt into boxes; the
  phrase strings are read from ``results["text_labels"]`` (``results["labels"]`` became ints in
  transformers >= 4.51). Pin ``transformers>=4.49,<5`` with torch 2.5.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from screwit.interfaces import FrameSet


def resolve_device() -> str:
    """Torch device for the real adapters (finding 13).

    ``SCREWIT_DEVICE`` wins when set; otherwise cuda when a GPU is present, else cpu. The builders
    default to ``device="cuda"`` and crash on a CPU-only host, so every builder call passes this.
    """
    env = os.environ.get("SCREWIT_DEVICE")
    if env:
        return env
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class DinoBox:
    """One Grounding DINO detection in model-grid pixels (xyxy)."""

    phrase: str
    xyxy: tuple[float, float, float, float]
    score: float


# --------------------------------------------------------------------------- real adapters

class Sam2VideoAdapter:
    """SAM 2 video predictor (real GPU/CPU path)."""

    def __init__(self, checkpoint: str, model_cfg: str, device: str | None = None):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.device = device or resolve_device()
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from sam2.build_sam import build_sam2_video_predictor

            # finding 13: the builder defaults to device="cuda"; pass the resolved device.
            self._predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint,
                                                         device=self.device)
        return self._predictor

    @staticmethod
    def _write_jpeg_dir(frames: list[np.ndarray], dest: str) -> None:
        """Write frames as integer-named JPEGs (00000.jpg, ...), which init_state requires (14)."""
        from PIL import Image

        for i, img in enumerate(frames):
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(os.path.join(dest, f"{i:05d}.jpg"))

    def segment(self, frames: list[np.ndarray], *, points: np.ndarray | None = None,
                labels: np.ndarray | list[int] | None = None,
                box: np.ndarray | None = None, ann_frame_idx: int = 0,
                obj_id: int = 1) -> dict[int, np.ndarray]:
        """Propagate a mask across ``frames`` (model-grid uint8 images).

        The video directory must hold integer-named JPEGs, so the frames are written to a temp dir
        first (finding 14). ``labels`` (positive 1, negative 0) is passed as an int32 array.
        """
        import torch

        predictor = self._load()
        lbl = None if labels is None else np.asarray(labels, dtype=np.int32)
        with tempfile.TemporaryDirectory() as video_dir:
            self._write_jpeg_dir(frames, video_dir)
            with torch.inference_mode():
                state = predictor.init_state(video_dir)
                predictor.add_new_points_or_box(state, frame_idx=ann_frame_idx, obj_id=obj_id,
                                                points=points, labels=lbl, box=box)
                out: dict[int, np.ndarray] = {}
                for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                    out[frame_idx] = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                return out


class GroundingDinoAdapter:
    """Grounding DINO text->boxes (real GPU/CPU path, HF implementation)."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: str | None = None):
        self.model_id = model_id
        self.device = device or resolve_device()
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(
                self.device)
        return self._model, self._processor

    def detect(self, image: np.ndarray, phrases: list[str], *, box_threshold: float = 0.35,
               text_threshold: float = 0.25) -> list[DinoBox]:
        import torch
        from PIL import Image

        model, processor = self._load()
        text = ". ".join(p.strip().lower() for p in phrases) + "."
        pil = Image.fromarray(image)
        inputs = processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[pil.size[::-1]])[0]
        # finding 15: phrase strings live in results["text_labels"]; results["labels"] became ints
        # in transformers >= 4.51. Pin transformers>=4.49,<5 (torch 2.5 refuses transformers 5.x).
        phrase_labels = results["text_labels"]
        boxes = []
        for box, score, label in zip(results["boxes"], results["scores"], phrase_labels,
                                     strict=False):
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            boxes.append(DinoBox(str(label), (x0, y0, x1, y1), float(score)))
        return boxes


class Sam2AutomaticMasksAdapter:
    """SAM2AutomaticMaskGenerator (fallback when Grounding DINO misses a part)."""

    def __init__(self, checkpoint: str, model_cfg: str, device: str | None = None,
                 points_per_side: int | None = None):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.device = device or resolve_device()
        # finding 16: 32 points/side is ~4x slower on CPU; use 16 on CPU, 32 on GPU, unless given.
        self.points_per_side = points_per_side if points_per_side is not None else (
            32 if self.device == "cuda" else 16)
        self._gen = None

    def _load(self):
        if self._gen is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            # finding 13: build_sam2 defaults to device="cuda"; pass the resolved device.
            model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
            self._gen = SAM2AutomaticMaskGenerator(model, points_per_side=self.points_per_side)
        return self._gen

    def generate(self, image: np.ndarray) -> list[np.ndarray]:
        gen = self._load()
        return [m["segmentation"] for m in gen.generate(image)]


# --------------------------------------------------------------------------- CPU fakes

class FakeObjectMasker:
    """CPU fake for the main-object mask: everything with valid depth that is not the board.

    Synthetic frames carry depth with 0 for invalid and a flat board region, so the object mask is
    "has depth and is above the board plane" (the board sits at the smallest valid depth). This
    mirrors a SAM 2 video mask closely enough for the pipeline and tests without any model.
    """

    def masks(self, frames: FrameSet) -> list[np.ndarray]:
        out = []
        for depth in frames.depth_mm:
            valid = depth > 0
            out.append(valid)
        return out


class FakePartSegmenter:
    """CPU fake for Grounded SAM 2 part masks, driven by a known synthetic label image per frame.

    The fixture generator (scripts/make_fixture_assets.py) can hand the pipeline a per-frame label
    image (0 background, 1..N parts) so tests get deterministic, correct part masks without any
    model. ``part_names`` is the fixed name list a fake Claude would have proposed.
    """

    def __init__(self, label_images: list[np.ndarray], part_names: list[str]):
        self.label_images = label_images
        self.part_names = part_names

    def label_masks(self, frames: FrameSet) -> list[np.ndarray]:
        return [np.asarray(li, dtype=np.int32) for li in self.label_images]
