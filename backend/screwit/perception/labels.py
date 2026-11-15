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


class PartKind(BaseModel):
    """One kind of physical part in Claude's reading of the scan, with how many copies exist."""

    name: str
    copies: int = Field(ge=1, le=8)
    label_confidence: float = Field(ge=0, le=1)
    description: str
    material: str
    material_confidence: float = Field(ge=0, le=1)


class PatchNote(BaseModel):
    """What one highlighted patch covers, and the kind of part it lies on (a name from kinds)."""

    patch: int
    covers: str
    kind: str


class PatchGrouping(BaseModel):
    """Claude's reading of the patch images: the object's part kinds, then each patch's kind.

    Kinds come first, settled from the photos, and each patch is then only classified. Grouping
    patches into named copies directly (the earlier schema) was unstable: the same images gave a
    cup's pad and its housing back as the "left" and "right" pad. Which copy a patch belongs to is
    decided from geometry afterwards (part_atoms.regroup_instances)."""

    object_name: str
    kinds: list[PartKind] = Field(min_length=1, max_length=8)
    patch_notes: list[PatchNote] = Field(min_length=1)


# Independent readings asked for per scan; the most typical one is kept (Claude's answer for the
# same images varies between calls).
GROUPING_SAMPLES = 3


GROUPING_PROMPT = (
    "The first {n_photos} images are photos of one object from a phone video. Then come {n} "
    "images of that object's 3D scan, one per patch, each introduced by its patch number (0 to "
    "{last}). Every patch image shows the whole scan from the same two sides (left and right "
    "halves), grey, with that one patch in red. Every patch lies inside one physical part. "
    "First, in kinds, list the kinds of physical part the object is made of: the components a "
    "person would name separately, remove, replace or repair. Give each kind a short common name "
    "without side words (no left, right, front or back) and the number of identical copies the "
    "object has (the two sides of a symmetric object are one kind with 2 copies), at most "
    "{max_parts} parts in all. Prefer fewer, larger parts when unsure; splitting one piece into "
    "chunks is wrong. For each kind give a one-sentence description of what it does in the "
    "object, its likely material, and your confidence in the name and the material (0 to 1). "
    "Then, in patch_notes, for every patch say in a few words what the red region covers and give "
    "the kind it lies on, written exactly as in kinds. Classify a patch by what it is, not by what "
    "it touches: a patch on a connecting piece is that piece even where it meets another "
    "component. Ignore the printed checkerboard, the table and the background.")




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

    def group_patches(self, scan_id: str, tiles: list[bytes], keyframes, n_patches: int,
                      max_parts: int = 6) -> list[PatchGrouping]:
        """Group the highlighted patch images (``tiles``, one PNG per patch, in patch order) into
        named, labelled parts. Each image goes in as its own content block, introduced by
        "Patch i:"; drawn into one numbered grid, Claude misread which tile was which (a headband
        segment came back as "lower outer section of the right ear cup").

        Claude's grouping of the same images varies from call to call (on the headphones scan: 3
        parts, then 4 with one hinge, then 5 with a cup called "ear cushions"), so it is asked
        GROUPING_SAMPLES times in parallel and every valid reply is returned; the caller keeps the
        most typical one (``part_atoms.choose_grouping``). Empty for the fake provider or when
        every call fails, so the caller falls back."""
        if self.provider == "fake" or n_patches < 2 or len(tiles) != n_patches:
            return []
        from concurrent.futures import ThreadPoolExecutor

        from screwit.perception import claude_vision as cv

        photos = cv.evenly_spaced([p for p in (keyframes or []) if p is not None], 3)
        content = [cv.image_block_from_path(p) for p in photos]
        for i, tile in enumerate(tiles):
            content.append({"type": "text", "text": f"Patch {i}:"})
            content.append(cv.image_block_from_png(tile))
        prompt = GROUPING_PROMPT.format(n_photos=len(photos), n=n_patches, last=n_patches - 1,
                                        max_parts=max_parts)

        def sample(i: int) -> PatchGrouping | None:
            # Distinct text per sample, so the reply cache keeps every sample apart.
            text = prompt if i == 0 else f"{prompt} (Independent answer {i + 1}.)"
            try:
                return cv.ask(PatchGrouping, "You identify the parts of everyday objects from "
                                             "photos and 3D scans.",
                              content + [{"type": "text", "text": text}])
            except cv.VisionCallFailed as e:
                log.warning("patch grouping via Claude failed for %s: %s", scan_id, e)
                return None

        with ThreadPoolExecutor(max_workers=GROUPING_SAMPLES) as pool:
            return [r for r in pool.map(sample, range(GROUPING_SAMPLES)) if r is not None]

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
