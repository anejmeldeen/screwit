"""Grounded SAM 2 multi-part segmenter for the live path (P3).

Turns a list of part NAMES (from Claude) into per-keyframe integer label images (0 background,
1..N parts) that ``lifting.vote_labels`` lifts onto the mesh. Without this, the live path labels the
whole object as one part (PASSOFF known-problem #4).

Pipeline per object:
1. Grounding DINO finds a box for each named part in a reference keyframe (the clearest view).
2. SAM 2 video propagates each box's mask across all keyframes.
3. The per-frame masks are painted into one int label image per frame; where parts overlap, the
   higher-scoring (earlier, more-confident) detection wins.

Config (env, mirroring adapters.resolve_device):
- SCREWIT_SAM2_CHECKPOINT : path to the SAM 2 .pt (required to run; absent -> segmenter unavailable).
- SCREWIT_SAM2_CONFIG     : SAM 2 config name (default configs/sam2.1/sam2.1_hiera_b+.yaml).
- SCREWIT_DEVICE          : cpu/cuda (via resolve_device()).

Everything is best-effort: if the checkpoint is missing, a model errors, or nothing is detected,
``label_images`` returns None and the caller keeps the single-part fallback.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from screwit.interfaces import FrameSet

log = logging.getLogger(__name__)

DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def _phrase_matches(phrase: str, name: str) -> bool:
    """True when a Grounding DINO phrase grounds to this part name. Per-name detection means the
    phrase should equal or contain the name (minus subword fragments). Rejects '##' fragments and
    empties that DINO emits when a prompt doesn't ground cleanly."""
    p = (phrase or "").strip().lower()
    nm = (name or "").strip().lower()
    if not p or not nm or p.startswith("##"):  # reject subword fragments outright
        return False
    return p == nm or p in nm or nm in p or any(w in p for w in nm.split() if len(w) > 2)


def available() -> bool:
    """True when a SAM 2 checkpoint is configured and present (so the live path can segment parts)."""
    ckpt = os.environ.get("SCREWIT_SAM2_CHECKPOINT")
    return bool(ckpt) and os.path.exists(ckpt)


class GroundedSam2PartSegmenter:
    """Produces per-keyframe label images from part names using Grounding DINO + SAM 2."""

    def __init__(self, checkpoint: str | None = None, config: str | None = None,
                 device: str | None = None, *, box_threshold: float = 0.2,
                 text_threshold: float = 0.2) -> None:
        self.checkpoint = checkpoint or os.environ.get("SCREWIT_SAM2_CHECKPOINT")
        self.config = config or os.environ.get("SCREWIT_SAM2_CONFIG", DEFAULT_SAM2_CONFIG)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def _reference_frame(self, frames: FrameSet) -> int:
        """Pick the keyframe with the most valid depth (the fullest view of the object)."""
        best, best_valid = 0, -1
        for i, d in enumerate(frames.depth_mm):
            v = int(np.count_nonzero(d > 0))
            if v > best_valid:
                best, best_valid = i, v
        return best

    def label_images(self, frames: FrameSet, part_names: list[str],
                     object_masks: list[np.ndarray] | None = None) -> list[np.ndarray] | None:
        """Return one HxW int label image per keyframe (0 bg, 1..N parts), or None on any failure.

        Label i+1 corresponds to ``part_names[i]``, matching PerceptionImpl's ``part_names[label-1]``.
        """
        if not available() or not part_names:
            return None
        try:
            from screwit.perception.adapters import (
                GroundingDinoAdapter,
                Sam2VideoAdapter,
                resolve_device,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("part segmenter unavailable (import): %s", e)
            return None

        device = self.device or resolve_device()
        images = [np.asarray(im, dtype=np.uint8) for im in frames.images]
        if not images:
            return None
        h, w = images[0].shape[:2]
        ref = self._reference_frame(frames)

        # Detect each part name SEPARATELY. Packing all names into one period-joined Grounding DINO
        # prompt makes the text grounding bleed across phrases (e.g. "ear cups headband adjustment
        # sliders" as one box) and emit subword fragments ("##band", ""). One call per name keeps
        # each detection cleanly tied to its part and finds far more of the parts.
        try:
            dino = GroundingDinoAdapter(device=device)
        except Exception as e:  # noqa: BLE001
            log.warning("Grounding DINO unavailable; single-part fallback: %s", e)
            return None

        from screwit.perception.adapters import DinoBox

        tagged: list[tuple[int, DinoBox]] = []  # (label_id, best box for that part)
        for i, name in enumerate(part_names):
            try:
                found = dino.detect(images[ref], [name], box_threshold=self.box_threshold,
                                    text_threshold=self.text_threshold)
            except Exception as e:  # noqa: BLE001
                log.warning("Grounding DINO failed for %r: %s", name, e)
                continue
            # Keep this part's single best (highest-score) box, if any grounded to the name.
            good = [b for b in found if _phrase_matches(b.phrase, name)]
            if good:
                tagged.append((i + 1, max(good, key=lambda b: b.score)))
        if not tagged:
            log.info("Grounding DINO found no named parts; keeping single-part fallback")
            return None

        # Reject a part box that covers nearly the whole object: a detection spanning most of the
        # object (e.g. "padding foam" blown up to the entire headphone) is wrong and would swallow
        # every other part. Measure against the object mask area on the reference frame.
        ref_obj = None
        if object_masks is not None and ref < len(object_masks):
            ref_obj = np.asarray(object_masks[ref], dtype=bool)
            if ref_obj.shape != (h, w):
                ref_obj = None
        obj_area = float(ref_obj.sum()) if ref_obj is not None else float(h * w)

        def _area(b):
            x0, y0, x1, y1 = b.xyxy
            return max(0.0, x1 - x0) * max(0.0, y1 - y0)

        kept = []
        for label, b in tagged:
            # Only reject a box that is essentially the ENTIRE object (>95%); competitive per-pixel
            # assignment below prevents a merely-large part from dominating, so keep as many as we can.
            if obj_area > 0 and _area(b) > 0.95 * obj_area and len(tagged) > 1:
                log.info("dropping whole-object box %r (%.0f%% of object)", b.phrase,
                         100 * _area(b) / obj_area)
                continue
            kept.append((label, b))
        if not kept:
            kept = tagged  # keep at least one

        sam = Sam2VideoAdapter(self.checkpoint, self.config, device=device)
        # Get each part's propagated per-frame mask. Keep them SEPARATE (not overwrite-painted) so we
        # can resolve overlaps competitively and every detected part keeps its own region -- the goal
        # is as many distinct parts as the detector found, not one box dominating.
        part_masks: dict[int, dict[int, np.ndarray]] = {}  # label -> {frame_idx -> bool mask}
        box_area = {label: _area(b) for label, b in kept}
        for label, box in kept:
            try:
                per_frame = sam.segment(images, box=np.asarray(box.xyxy, dtype=np.float32),
                                        ann_frame_idx=ref, obj_id=label)
            except Exception as e:  # noqa: BLE001
                log.warning("SAM 2 propagation failed for %r: %s", box.phrase, e)
                continue
            part_masks[label] = {fi: np.asarray(mk, dtype=bool) for fi, mk in per_frame.items()}
        if not part_masks:
            return None

        # Per-frame competitive assignment: a pixel claimed by several parts goes to the part with
        # the SMALLEST box (the tightest, most specific detection), so a large loose part cannot
        # swallow a small one and all parts survive. Priority = ascending box area.
        order = sorted(part_masks.keys(), key=lambda la: box_area.get(la, 0.0), reverse=True)
        label_imgs = [np.zeros((h, w), dtype=np.int32) for _ in images]
        painted_any = False
        for fidx in range(len(images)):
            target = label_imgs[fidx]
            om = None
            if object_masks is not None and fidx < len(object_masks):
                o = np.asarray(object_masks[fidx], dtype=bool)
                if o.shape == (h, w):
                    om = o
            for label in order:  # largest first; smaller boxes overwrite -> smallest box wins a pixel
                m = part_masks.get(label, {}).get(fidx)
                if m is None or m.shape != (h, w):
                    continue
                if om is not None:
                    m = m & om  # clip to object -> no table/background in any part
                target[m] = label
                if m.any():
                    painted_any = True

        if not painted_any:
            return None
        return [self._smooth_labels(li) for li in label_imgs]

    @staticmethod
    def _smooth_labels(label_img: np.ndarray) -> np.ndarray:
        """Clean per-part mask edges: fill small holes and drop speckle, per label. This removes the
        jagged/pitted boundaries SAM 2 masks can have, so the parts lifted onto the mesh have
        smoother edges. Pure scipy morphology; labels are preserved."""
        try:
            from scipy import ndimage
        except Exception:  # noqa: BLE001
            return label_img
        out = label_img.copy()
        for lab in [v for v in np.unique(label_img) if v > 0]:
            m = label_img == lab
            # close (fill pinholes), then open (remove isolated specks), 2px structuring element.
            m = ndimage.binary_closing(m, iterations=2)
            m = ndimage.binary_opening(m, iterations=1)
            # keep only the largest connected blob of this label (drop detached fragments).
            lbl, n = ndimage.label(m)
            if n > 1:
                sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
                m = lbl == (int(np.argmax(sizes)) + 1)
            out[(label_img == lab) & ~m] = 0  # pixels removed from this label -> background
            out[m] = lab                        # pixels added by closing -> this label
        return out
