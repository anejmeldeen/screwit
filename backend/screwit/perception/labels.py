"""Part naming, labelling and materials via Claude, with a deterministic fake (SPEC 6.2 2 & 5).

The real path sends keyframes / rendered thumbnails to Claude (via screwit.llm) and validates the
structured JSON reply with Pydantic. SCREWIT_LLM=fake (the default, used by every test and mock
mode) returns fixed names and labels keyed by scan, so the fake outputs match the hero fixtures
exactly (scn_trap, scn_knob).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class PartNames(BaseModel):
    """Claude's structured reply to "name the visible component parts" (SPEC 6.2 step 2)."""

    names: list[str] = Field(min_length=1)


class PartLabel(BaseModel):
    """Claude's structured reply for one part's label/description/material (SPEC 6.2 step 5)."""

    label: str
    label_confidence: float = Field(ge=0, le=1)
    description: str
    material: str
    material_confidence: float = Field(ge=0, le=1)


# Deterministic fake answers keyed by scan id, matching the committed hero fixtures.
_FAKE_NAMES: dict[str, list[str]] = {
    "scn_trap": ["tailpiece", "J-bend", "slip nut", "slip-joint washer"],
    "scn_knob": ["control panel", "D-shaft"],
}

_FAKE_LABELS: dict[str, dict[str, PartLabel]] = {
    "scn_trap": {
        "tailpiece": PartLabel(label="tailpiece", label_confidence=0.86,
                               description="Vertical tube from the sink drain down into the trap",
                               material="PVC plastic", material_confidence=0.7),
        "J-bend": PartLabel(label="J-bend", label_confidence=0.83,
                            description="U-shaped bend that holds water to block sewer gas",
                            material="PVC plastic", material_confidence=0.7),
        "slip nut": PartLabel(label="slip nut", label_confidence=0.81,
                              description="Plastic slip nut joining the J-bend to the tailpiece",
                              material="PVC/ABS plastic", material_confidence=0.6),
        "slip-joint washer": PartLabel(
            label="slip-joint washer", label_confidence=0.42,
            description="Beveled washer inside the slip nut; only its edge is visible",
            material="polyethylene", material_confidence=0.4),
    },
    "scn_knob": {
        "control panel": PartLabel(label="control panel", label_confidence=0.77,
                                   description="Flat front panel the knob shaft comes out of",
                                   material="painted steel", material_confidence=0.5),
        "D-shaft": PartLabel(
            label="D-shaft", label_confidence=0.74,
            description="Control shaft with one flat side; the original knob is missing",
            material="steel", material_confidence=0.6),
    },
}

def _label_from_name(name: str) -> PartLabel:
    """A sensible fake label for any proposed part name, carrying no category assumption."""
    name = (name or "").strip() or "part"
    return PartLabel(label=name, label_confidence=0.5,
                     description=f"{name} identified from the scan",
                     material="unknown", material_confidence=0.2)


class Labeller:
    """Names and labels parts. provider='fake' uses deterministic data (object-agnostic).

    The fake draws names/labels from data, not from any hard-coded category: hero defaults for the
    two fixtures, plus optional per-call overrides (``fake_names`` / ``fake_labels`` keyed by scan
    id) so any synthetic object can supply its own names. For an unknown name the fake derives a
    sensible label from the name itself rather than inventing a category.
    """

    def __init__(self, provider: str = "fake", *,
                 fake_names: dict[str, list[str]] | None = None,
                 fake_labels: dict[str, dict[str, PartLabel]] | None = None):
        self.provider = provider
        self.fake_names = {**_FAKE_NAMES, **(fake_names or {})}
        self.fake_labels = {**_FAKE_LABELS, **(fake_labels or {})}

    def name_parts(self, scan_id: str, keyframes) -> PartNames:
        if self.provider == "fake":
            return PartNames(names=list(self.fake_names.get(scan_id, ["part"])))
        return self._name_parts_claude(scan_id, keyframes)

    def label_part(self, scan_id: str, proposed_name: str, thumbnail_png: bytes) -> PartLabel:
        if self.provider == "fake":
            table = self.fake_labels.get(scan_id, {})
            if proposed_name in table:
                return table[proposed_name]
            return _label_from_name(proposed_name)
        return self._label_part_claude(scan_id, proposed_name, thumbnail_png)

    # Real Claude paths. A provider failure or an invalid reply falls back to the generic answer, so
    # a scan never fails because the LLM did; the fallback is logged.

    def _name_parts_claude(self, scan_id: str, keyframes) -> PartNames:
        from screwit.perception import claude_vision as cv

        frames = cv.evenly_spaced([p for p in (keyframes or []) if p is not None], 6)
        content = [cv.image_block_from_path(p) for p in frames]
        content.append({"type": "text", "text": (
            "These are frames from a phone video walking around one object. Name the main "
            "physical component parts a person could point to, replace or repair, most important "
            "first, at most 8. Use short common names a hardware store would understand. Do not "
            "name the printed checkerboard, the table or the background.")})
        try:
            return cv.ask(PartNames, "You identify the parts of everyday objects from photos.", content)
        except cv.VisionCallFailed as e:
            log.warning("part naming via Claude failed for %s; using a generic name: %s", scan_id, e)
            return PartNames(names=["part"])

    def _label_part_claude(self, scan_id, name, thumb) -> PartLabel:
        from screwit.perception import claude_vision as cv

        content = [cv.image_block_from_png(thumb), {"type": "text", "text": (
            f"This is a render of one part segmented from a 3D scan; it was proposed as {name!r}. "
            "Give its label (keep the proposed name unless it is clearly wrong), a one-sentence "
            "description of what it does in the object, its likely material, and your confidence "
            "in the label and the material (0 to 1).")}]
        try:
            return cv.ask(PartLabel, "You identify parts of everyday objects from renders.", content)
        except cv.VisionCallFailed as e:
            log.warning("part labelling via Claude failed for %s/%s: %s", scan_id, name, e)
            return _label_from_name(name)
