"""Grounded SAM 2 multi-part segmenter for the live path (P3).

Turns a list of part NAMES (from Claude, or a chat message) into per-keyframe integer label images
(0 background, 1..N parts) that ``lifting.vote_labels`` lifts onto the mesh. Without this, the live
path labels the whole object as one part (PASSOFF known-problem #4).

Pipeline per object (SAM 3 text/concept segmentation):
1. For each part name, SAM 3 segments that concept directly from a TEXT prompt -- no Grounding DINO
   boxes, no SAM 2 box propagation. SAM 3's native concept segmentation is what makes clean
   multi-part splits ("ear cup", "headband") feasible.
2. Each part's per-frame masks are clipped to the object mask (no table/background), resolved
   competitively (a larger concept cannot overwrite a smaller one), and edge-smoothed.

Config (env): see ``sam3_adapter`` -- SCREWIT_SAM3 (enable), SCREWIT_SAM3_MODEL, SCREWIT_SAM3_BACKEND
(hf | sam3cpu | meta), SCREWIT_DEVICE. The sam3-cpu backend runs on a CPU-only host.

Everything is best-effort: if SAM 3 is unavailable, the weights are not authenticated, or nothing is
detected, ``label_images`` returns None and the caller keeps the single-part fallback.
"""

from __future__ import annotations

import logging

import numpy as np

from screwit.interfaces import FrameSet

log = logging.getLogger(__name__)


def available() -> bool:
    """True when SAM 3 (text-prompted part segmentation) is enabled and a backend is importable.
    The sam3-cpu backend makes this usable on a CPU-only host."""
    from screwit.perception import sam3_adapter

    return sam3_adapter.available()[0]


class Sam3PartSegmenter:
    """Produces per-keyframe label images from part names using SAM 3 text/concept prompts.

    One text prompt per part name (no Grounding DINO, no SAM 2 boxes); SAM 3 segments the concept
    directly. Masks are clipped to the object, resolved competitively so a large concept cannot
    swallow a small one, and edge-smoothed."""

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        self.model = model
        self.device = device

    def label_images(self, frames: FrameSet, part_names: list[str],
                     object_masks: list[np.ndarray] | None = None) -> list[np.ndarray] | None:
        """Return one HxW int label image per keyframe (0 bg, 1..N parts), or None on any failure.

        Label i+1 corresponds to ``part_names[i]``, matching PerceptionImpl's ``part_names[label-1]``.
        """
        if not available() or not part_names:
            return None
        from screwit.perception import sam3_adapter

        images = [np.asarray(im, dtype=np.uint8) for im in frames.images]
        if not images:
            return None
        h, w = images[0].shape[:2]

        # SAM 3 segments by TEXT (concept) prompt directly -- one prompt per part name, no Grounding
        # DINO box step. This replaces the old SAM 2 + DINO path: SAM 3's native concept segmentation
        # is what makes clean multi-part splits ("ear cup", "headband") possible, and the sam3-cpu
        # backend runs it on this CPU host.
        seg = sam3_adapter.Sam3TextSegmenter()
        part_masks: dict[int, dict[int, np.ndarray]] = {}  # label -> {frame_idx -> bool mask}
        mask_size: dict[int, int] = {}                      # label -> total mask pixels (for priority)
        for i, name in enumerate(part_names):
            label = i + 1
            per_frame = seg.segment(images, name)
            if not per_frame:
                continue
            kept_frames = {}
            total = 0
            for fi, m in per_frame.items():
                m = np.asarray(m, dtype=bool)
                if m.shape != (h, w):
                    continue
                kept_frames[fi] = m
                total += int(m.sum())
            if kept_frames:
                part_masks[label] = kept_frames
                mask_size[label] = total
        if not part_masks:
            log.info("SAM 3 found no named parts; keeping single-part fallback")
            return None

        # Per-frame competitive assignment: a pixel claimed by several parts goes to the SMALLER part
        # (by total mask size), so a large concept mask cannot swallow a small one and every detected
        # part survives. Paint largest-first; smaller overwrites.
        order = sorted(part_masks.keys(), key=lambda la: mask_size.get(la, 0), reverse=True)
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
