"""Relations and scene graph (SPEC 6.2 step 6, 4.4).

Candidate pairs are parts whose surfaces are within 2 mm of each other (nearest-neighbour
distance between the two part point sets). The relation type is decided from geometry cues
(coaxial cylinders suggest threaded or press-fit) plus Claude; SCREWIT_LLM=fake uses a
deterministic classifier keyed by scan so the fake scene graph matches the hero fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from screwit.contracts import (
    FailureHypothesis,
    Part,
    Relation,
    RelationType,
    SceneGraph,
    SceneObject,
)

SURFACE_PROXIMITY_MM = 2.0
COAXIAL_ANGLE_DEG = 10.0
COAXIAL_OFFSET_MM = 5.0


@dataclass(frozen=True)
class PairGeometry:
    a: str
    b: str
    surface_gap_mm: float
    coaxial: bool


def _axis(part: Part):
    for f in part.features:
        if f.type in ("cylinder", "bore", "d_shaft"):
            return np.asarray(f.axis_point_mm, float), np.asarray(f.axis_dir, float)
    return None


def _coaxial(pa: Part, pb: Part) -> bool:
    aa, ab = _axis(pa), _axis(pb)
    if aa is None or ab is None:
        return False
    (pa_pt, pa_dir), (pb_pt, pb_dir) = aa, ab
    cos = abs(float(np.dot(pa_dir, pb_dir)))
    if cos < np.cos(np.radians(COAXIAL_ANGLE_DEG)):
        return False
    # Perpendicular distance between the two axis lines' reference points.
    delta = pb_pt - pa_pt
    perp = delta - np.dot(delta, pa_dir) * pa_dir
    return float(np.linalg.norm(perp)) <= COAXIAL_OFFSET_MM


def surface_gap_mm(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Minimum nearest-neighbour distance between two point sets (mm)."""
    a = np.asarray(points_a, float).reshape(-1, 3)
    b = np.asarray(points_b, float).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    best = float("inf")
    for chunk in np.array_split(a, max(1, len(a) // 512 + 1)):
        d = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def candidate_pairs(parts: list[Part], part_points: dict[str, np.ndarray]) -> list[PairGeometry]:
    pairs: list[PairGeometry] = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pa, pb = parts[i], parts[j]
            gap = surface_gap_mm(part_points.get(pa.part_id, np.empty((0, 3))),
                                 part_points.get(pb.part_id, np.empty((0, 3))))
            if gap <= SURFACE_PROXIMITY_MM:
                pairs.append(PairGeometry(pa.part_id, pb.part_id, gap, _coaxial(pa, pb)))
    return pairs


# Deterministic fake relation/scene classification keyed by scan id (matches hero fixtures).
_FAKE_SCENE: dict[str, dict] = {
    "scn_trap": {
        "object": SceneObject(category="sink drain P-trap",
                              summary="White PVC 1-1/2 in P-trap under a kitchen sink"),
        "function": ("Traps water to block sewer gas; slip nuts compress washers to seal joints."),
        "relations": [
            ("p3", "p2", "threaded", 0.7, None),
            ("p4", "p3", "seals", 0.6, "slip washer inside nut"),
        ],
        "failures": [("p3", "worn or misaligned slip washer",
                      ["water marks below joint in frame 0042"], 0.6)],
    },
    "scn_knob": {
        "object": SceneObject(category="appliance control panel",
                              summary="Steel panel with a 6 mm D-shaft whose knob is missing"),
        "function": ("Turning the shaft operates the control behind the panel; the knob gives "
                     "grip and shows the setting."),
        "relations": [("p2", "p1", "fastened", 0.6, "shaft passes through the panel")],
        "failures": [("p2", "knob missing", ["bare shaft visible in every frame"], 0.95)],
    },
}


class SceneBuilder:
    def __init__(self, provider: str = "fake"):
        self.provider = provider

    def build(self, scan_id: str, parts: list[Part],
              part_points: dict[str, np.ndarray]) -> SceneGraph:
        part_ids = [p.part_id for p in parts]
        if self.provider == "fake" and scan_id in _FAKE_SCENE:
            spec = _FAKE_SCENE[scan_id]
            known = set(part_ids)
            relations = [Relation(a=a, b=b, type=t, confidence=c, note=n)
                         for (a, b, t, c, n) in spec["relations"] if a in known and b in known]
            failures = [FailureHypothesis(part_id=pid, mode=mode, evidence=ev, likelihood=lk)
                        for (pid, mode, ev, lk) in spec["failures"] if pid in known]
            return SceneGraph(scan_id=scan_id, object=spec["object"], parts=part_ids,
                              relations=relations, function=spec["function"],
                              failure_hypotheses=failures)
        # Generic geometry-only answer: relate touching parts, coaxial -> threaded else rests_on.
        pairs = candidate_pairs(parts, part_points)
        if self.provider != "fake":
            graph = self._build_claude(scan_id, parts, pairs)
            if graph is not None:
                return graph
        relations = []
        for pg in pairs:
            rtype: RelationType = "threaded" if pg.coaxial else "rests_on"
            relations.append(Relation(a=pg.a, b=pg.b, type=rtype,
                                      confidence=0.5 if pg.coaxial else 0.4))
        return SceneGraph(scan_id=scan_id,
                          object=SceneObject(category="object", summary="Scanned object"),
                          parts=part_ids, relations=relations, function="", failure_hypotheses=[])

    def _build_claude(self, scan_id: str, parts: list[Part],
                      pairs: list[PairGeometry]) -> SceneGraph | None:
        """Ask Claude for the object, its function, the relation type of each touching pair and
        ranked failure modes. Only part ids from the scan and only geometry-candidate pairs are
        kept; anything else from the reply is dropped. None on any failure (caller uses geometry)."""
        import json
        import logging

        from pydantic import BaseModel, Field

        from screwit.perception import claude_vision as cv

        class _Rel(BaseModel):
            a: str
            b: str
            type: RelationType
            confidence: float = Field(ge=0, le=1)
            note: str | None = None

        class _Reply(BaseModel):
            object: SceneObject
            function: str
            relations: list[_Rel]
            failure_hypotheses: list[FailureHypothesis]

        part_info = [{"part_id": p.part_id, "label": p.label, "description": p.description,
                      "material": p.material_guess.value,
                      "size_mm": [round(float(e), 1) for e in p.obb.extent]} for p in parts]
        pair_info = [{"a": pg.a, "b": pg.b, "surface_gap_mm": round(pg.surface_gap_mm, 2),
                      "coaxial": pg.coaxial} for pg in pairs]
        prompt = (
            "A 3D scan of one object was split into these parts (sizes are the part's box in mm):\n"
            f"{json.dumps(part_info, indent=1)}\n"
            "These pairs touch (from geometry):\n"
            f"{json.dumps(pair_info, indent=1)}\n"
            "Say what the object is (category and a one-sentence summary), what it does, the relation "
            "type of each touching pair, and the most likely ways it fails or wears, each on one of "
            "these part ids, with evidence a person could check and a likelihood. Use only the part "
            "ids above.")
        try:
            reply = cv.ask(_Reply, "You explain how everyday objects work and how they fail.",
                           [{"type": "text", "text": prompt}])
        except cv.VisionCallFailed as e:
            logging.getLogger(__name__).warning(
                "scene graph via Claude failed for %s; using geometry only: %s", scan_id, e)
            return None
        known = {p.part_id for p in parts}
        allowed = {frozenset((pg.a, pg.b)) for pg in pairs}
        relations = [Relation(**r.model_dump()) for r in reply.relations
                     if r.a in known and r.b in known and frozenset((r.a, r.b)) in allowed]
        failures = [f for f in reply.failure_hypotheses if f.part_id in known]
        failures.sort(key=lambda f: -f.likelihood)
        return SceneGraph(scan_id=scan_id, object=reply.object, parts=[p.part_id for p in parts],
                          relations=relations, function=reply.function, failure_hypotheses=failures)
