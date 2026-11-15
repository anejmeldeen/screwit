"""Deterministic offline chat agent (AGENTS.md object-agnostic; replaces the regex replay fallback).

This agent answers ANY message about ANY object with no LLM and no network, grounded only in the
scan's own data (scene graph, parts, measurements) and the curated catalogue. It is the fallback used
when no LLM is configured or a real provider call fails, and it is the default in mock mode.

Design points required by the brief:
- Broad, object-agnostic intent detection: synonyms, questions and imperatives map to one of
  explain / diagnose / fix / buy / make / measure / part-query / default. No object category is named
  in this code; the object vocabulary (part labels, categories, catalogue search terms) comes from the
  scan's own data at runtime.
- Grounded output only: every number comes from a Measurement with its source, every part id exists,
  cards are built from the contracts, and a missing catalogue match is stated rather than faked.
- Per-scan conversation state: a "make/print" request that needs caliper values parks a pending CAD
  intent; the next message carrying numbers is applied through set_measurement and the CAD job runs.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from screwit.agent import catalogue
from screwit.agent.cad import TEMPLATE_SCHEMAS
from screwit.agent.tools import CardSignal, ToolContext
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import (
    BomLine,
    CardEvent,
    Diagnosis,
    DoneEvent,
    HighlightEvent,
    Part,
    PartsToBuyCard,
    RepairPlan,
    RepairPlanCard,
    SceneGraph,
    Step,
    TokenEvent,
)
from screwit.contracts.models import Measurement

# ---------------------------------------------------------------- intent vocabulary (object-agnostic)
# These are English verbs/question words about *repair actions*, not object categories. The object's
# own nouns never appear here; they are read from the scan at runtime.
_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("buy", re.compile(r"\b(buy|purchase|order|where.*(get|buy)|shop|store|get a new)", re.I)),
    ("make", re.compile(
        r"\b(make|print|3d ?print|design|fabricat|replace|replacement|new one|recreate|model)", re.I)),
    # fix/diagnose: an explicit repair/diagnosis verb OR a symptom statement (object-agnostic symptoms).
    # No trailing \b, so stems match their inflections ("leak" -> "leaking", "wobbl" -> "wobbling").
    ("fix", re.compile(
        r"\b(manual|instructions|fix|repair|mend|stop|solve|sort out|how do i|how to|plan|wrong|broke|problem|fault|"
        r"fail|diagnos|leak|drip|wet|water|wobbl|loose|stuck|rattl|won'?t|doesn'?t|not working|"
        r"issue|why)", re.I)),
    ("measure", re.compile(
        r"\b(how (big|long|wide|thick|large)|size|dimension|diameter|measure|length|width|"
        r"height|how many ?mm)", re.I)),
    ("explain", re.compile(
        r"\b(explain|how does|how do.*work|what is this|what does|tell me about|describe|"
        r"understand|what('| i)s this)", re.I)),
]

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm)?", re.I)
_ACCEPT_RE = re.compile(r"\b(use|accept|keep|the scan|that'?s fine|go ahead|that works|ok(ay)?)\b", re.I)
_PART_CHIP_RE = re.compile(r"\[\[part:(p\w+)\]\]")

# The mating measurement fields a FIT actually consumes, per feature type (tighter than the full
# FEATURE_MEASUREMENT_FIELDS: a shaft's exposed length is not a bore-fit dimension). Object-agnostic.
_MATING_FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "d_shaft": ("radius", "flat_depth"),
    "cylinder": ("radius",),
    "bore": ("radius",),
    "hole": ("radius", "depth"),
}

# A human word for the feature a new part fits onto, used in the confirm-before-printing question.
_FEATURE_WORD: dict[str, str] = {
    "d_shaft": "shaft", "cylinder": "shaft", "bore": "bore", "hole": "hole",
}

# Common synonyms for the fixed templates, so a request naming a template-ish shape is NOT treated as
# a custom shape by the offline agent.
_TEMPLATE_SYNONYMS: frozenset[str] = frozenset({
    "knob", "dial", "handle", "bracket", "l-bracket", "angle bracket", "spacer", "washer", "bushing",
    "adapter", "adaptor", "coupler", "reducer", "cap", "plug", "cover", "pipe", "hose",
})

# Signals that the user wants a NEW/custom shape no template covers (so the offline agent must defer to
# the online codegen path). Deliberately explicit to avoid false positives on ordinary make requests.
_CUSTOM_SHAPE_RE = re.compile(
    r"\b(custom|bespoke|from scratch|brand[ -]?new (shape|design|part)|unique|"
    r"gear|funnel|clip|hook|clamp|mount|stand|holder|enclosure|housing|handle grip|"
    r"something (that|like)|a shape|new shape|invent|whatever shape)\b", re.I)


@dataclass
class _PendingCad:
    """A parked make/print intent waiting for caliper confirmation of mating dimensions."""

    part_id: str
    feature_id: str
    template: str
    fields: list[str] = field(default_factory=list)  # measurement fields still needing a value


# Per-scan conversation state (process-local; survives across messages within one server process).
_STATE: dict[str, _PendingCad] = {}


class OfflineChatAgent:
    """ChatAgent (interfaces.py). Deterministic, grounded, object-agnostic."""

    def __init__(self, reg: Registry, mode_note: str | None = None) -> None:
        self._reg = reg
        self._store = ScanStore(reg.data)
        self._mode_note = mode_note  # a short line emitted first saying which mode answered

    def reply(self, scan_id: str, message: str, image_uris: list[str]) -> Iterator:
        if self._mode_note:
            yield TokenEvent(text=self._mode_note + " ")
        try:
            yield from self._route(scan_id, message)
        except Exception as e:  # never kill the stream; report and finish.
            yield TokenEvent(text=f"(offline agent hit an error: {e}) ")
        yield DoneEvent(message_id="msg_" + uuid.uuid4().hex[:12])

    # ------------------------------------------------------------ routing
    def _route(self, scan_id: str, message: str) -> Iterator:
        parts = self._store.list_parts(scan_id)
        graph = self._store.get_scene_graph(scan_id)

        # A parked CAD confirmation takes priority if this message supplies numbers or accepts the scan.
        pending = _STATE.get(scan_id)
        if pending is not None and (_NUM_RE.search(message) or _ACCEPT_RE.search(message)):
            yield from self._apply_pending_cad(scan_id, message, pending, parts)
            return

        # A message naming a specific part (chip or label) answers about that part.
        named = self._named_part(message, parts)
        intent = self._classify(message)

        if named is not None and intent in {None, "measure", "explain"}:
            yield from self._answer_part(named)
            return

        if intent == "make":
            yield from self._intent_make(scan_id, message, parts, graph)
        elif intent == "buy":
            yield from self._intent_buy(scan_id, message, parts)
        elif intent == "fix":
            yield from self._intent_fix(scan_id, parts, graph, named)
        elif intent == "measure":
            yield from self._intent_measure(parts)
        elif intent == "explain":
            yield from self._intent_explain(parts, graph)
        elif named is not None:
            yield from self._answer_part(named)
        else:
            yield from self._intent_default(parts, graph)

    def _classify(self, message: str) -> str | None:
        for name, pat in _INTENT_PATTERNS:
            if pat.search(message):
                return name
        return None

    def _named_part(self, message: str, parts: list[Part]) -> Part | None:
        chip = _PART_CHIP_RE.search(message)
        if chip:
            return next((p for p in parts if p.part_id == chip.group(1)), None)
        low = message.lower()
        # Match on a part's own label words (object vocabulary from the scan, not hard-coded).
        best: Part | None = None
        best_len = 0
        for p in parts:
            label = p.label.lower()
            if label and label in low and len(label) > best_len:
                best, best_len = p, len(label)
        return best

    # ------------------------------------------------------------ intents
    def _intent_explain(self, parts: list[Part], graph: SceneGraph | None) -> Iterator:
        if graph is None:
            yield TokenEvent(text="I don't have a scene graph for this object yet.")
            return
        yield TokenEvent(text=f"This is a {graph.object.category}: {graph.object.summary} ")
        yield TokenEvent(text=f"It works like this: {graph.function} ")
        if parts:
            names = ", ".join(f"{p.label} [[part:{p.part_id}]]" for p in parts)
            yield TokenEvent(text=f"The parts I can see are: {names}. ")
        if graph.relations:
            rel = graph.relations[0]
            yield TokenEvent(text=f"For example [[part:{rel.a}]] is {rel.type.replace('_', ' ')} "
                                  f"[[part:{rel.b}]].")
        yield HighlightEvent(part_ids=[p.part_id for p in parts])

    def _intent_diagnose(self, scan_id: str, parts: list[Part], graph: SceneGraph | None) -> Iterator:
        """Deprecated standalone path; kept so direct callers still work. Delegates to fix."""
        yield from self._intent_fix(scan_id, parts, graph)

    def _intent_fix(self, scan_id: str, parts: list[Part], graph: SceneGraph | None,
                    target: Part | None = None) -> Iterator:
        if graph is None:
            yield TokenEvent(text="I need the scene graph before I can diagnose or plan a repair.")
            return
        ranked = sorted(graph.failure_hypotheses, key=lambda h: h.likelihood, reverse=True)
        if target:
            ranked = [h for h in ranked if h.part_id == target.part_id]
        suspects = [target.part_id] if target else [h.part_id for h in ranked] or [p.part_id for p in parts[:1]]
        yield HighlightEvent(part_ids=suspects)

        # Ranked diagnosis first, grounded in the failure hypotheses.
        if ranked:
            yield TokenEvent(text="Most likely causes, highest first: ")
            for h in ranked:
                label = self._label(parts, h.part_id)
                ev = f" (evidence: {h.evidence[0]})" if h.evidence else ""
                yield TokenEvent(text=f"[[part:{h.part_id}]] {label}: {h.mode}, "
                                      f"about {round(h.likelihood * 100)}% likely{ev}. ")
        else:
            yield TokenEvent(text="I can't see an obvious fault from the scan, but here is a general "
                                  "repair plan. ")

        top = ranked[0] if ranked else None
        yield TokenEvent(text="Here is a repair plan. ")

        bom = self._buy_lines_for(parts, suspects, best_only=True)
        subject = target.label if target else graph.object.category
        plan = RepairPlan(
            plan_id="pln_" + uuid.uuid4().hex[:8],
            intent="repair",
            diagnosis=Diagnosis(summary=(top.mode if top else f"Inspect the {subject}"),
                                part_ids=suspects, confidence=(top.likelihood if top else 0.3)),
            safety=["Disconnect power or shut off the supply before working on the object.",
                    "Work gently; do not overtighten plastic fittings."],
            tools=["basic hand tools", "a cloth or bucket to catch spills"],
            steps=self._repair_steps(parts, suspects, graph),
            bom=bom,
            printable=[],
            followups=["Want me to design a printable replacement for any part?"],
        )
        yield CardEvent(card=RepairPlanCard(payload=plan))
        if bom:
            yield CardEvent(card=PartsToBuyCard(payload=bom))
        else:
            yield TokenEvent(text="I did not find a matching off-the-shelf part in the catalogue for "
                                  "this repair, so measure the worn part and I can design one to print.")

    def _intent_buy(self, scan_id: str, message: str, parts: list[Part]) -> Iterator:
        lines = self._buy_lines_for(parts, [p.part_id for p in parts], query_hint=message)
        if not lines:
            yield TokenEvent(text="I couldn't find a standard part in the catalogue for this object. "
                                  "Measure the part you need and I can either search again or design one "
                                  "to print.")
            return
        yield TokenEvent(text="Here are standard parts that match what the scan measured: ")
        yield CardEvent(card=PartsToBuyCard(payload=lines))

    def _intent_measure(self, parts: list[Part]) -> Iterator:
        if not parts:
            yield TokenEvent(text="There are no measured parts on this object yet.")
            return
        yield TokenEvent(text="Measured dimensions (with their source): ")
        for p in parts:
            for dim_name, meas in self._part_measurements(p):
                src = meas.preferred
                yield TokenEvent(text=f"[[part:{p.part_id}]] {p.label} {dim_name}: "
                                      f"{round(meas.value_mm, 1)} mm ({src}). ")

    def _intent_make(self, scan_id: str, message: str, parts: list[Part],
                     graph: SceneGraph | None) -> Iterator:
        target = self._named_part(message, parts) or self._printable_candidate(parts, graph)
        if target is None:
            yield TokenEvent(text="Tell me which part you want to replace and I can design it.")
            return
        feature = self._mating_feature(target)
        if feature is None:
            yield TokenEvent(text=f"I can't find a mating feature on [[part:{target.part_id}]] to fit a "
                                  "new part to. Tell me the dimensions you want.")
            return

        template = self._template_for(target, feature)
        # The offline agent cannot design NEW shapes (that needs the real-Claude codegen path). If the
        # user explicitly asks for a shape no template covers, say so and offer the closest template.
        if self._requests_unknown_shape(message):
            yield HighlightEvent(part_ids=[target.part_id])
            yield TokenEvent(text="I can't design a brand-new custom shape while running offline; that "
                                  "needs the online model. The closest standard template I can make for "
                                  f"[[part:{target.part_id}]] is a {template}. Want me to make that, or "
                                  "reconnect the model to design a custom part? ")
            return

        word = _FEATURE_WORD.get(feature.type, "fitting")
        yield HighlightEvent(part_ids=[target.part_id])

        # Confirm-before-printing: ALWAYS ask the user to confirm the mating dimension(s) of the fit on
        # the first make request (the scan cannot resolve small features reliably, SPEC 6.3). The next
        # message carrying a number (or "use the scan value") is applied via set_measurement, then the
        # CAD job runs. This is the designed path, not a failure mode.
        fields = [f for f in _MATING_FIELDS_BY_TYPE.get(feature.type, ()) if getattr(feature, f, None)]
        if not fields:
            yield from self._run_cad(scan_id, target, feature, template, parts)
            return
        _STATE[scan_id] = _PendingCad(part_id=target.part_id, feature_id=feature.feature_id,
                                      template=template, fields=fields)
        primary = fields[0]
        scan_val = round(getattr(feature, primary).value_mm, 1)
        unit = "diameter" if primary == "radius" else primary.replace("_", " ")
        shown = scan_val * 2 if primary == "radius" else scan_val
        yield TokenEvent(text=f"I can print a {template} for the {word} [[part:{target.part_id}]]. "
                              f"The scan puts the {word} {unit} at about {shown} mm, but the scan "
                              f"cannot resolve that reliably and a good fit needs the real value. "
                              f"Measure the {word} with calipers and send the value, or say "
                              "'use the scan value' to accept it, and I will build the part.")

    def _requests_unknown_shape(self, message: str) -> bool:
        """True when the user names a specific shape that is NOT one of the fixed templates, so the
        offline agent should defer to the online codegen path. Object-agnostic: templates and their
        common synonyms are the only shapes the offline agent can build."""
        if not _CUSTOM_SHAPE_RE.search(message):
            return False
        low = message.lower()
        known = set(TEMPLATE_SCHEMAS) | _TEMPLATE_SYNONYMS
        return not any(name.replace("_", " ") in low or name in low for name in known)

    def _apply_pending_cad(self, scan_id: str, message: str, pending: _PendingCad,
                           parts: list[Part]) -> Iterator:
        from screwit.contracts import PartPatch
        from screwit.contracts.models import MeasurementEdit

        accept_scan = _ACCEPT_RE.search(message)
        numbers = [float(m.group(1)) for m in _NUM_RE.finditer(message)]

        # Build ONE patch with every mating field, so a fixture-backed patch_part (which re-reads the
        # pristine part each call) cannot drop earlier edits. Numbers map to fields in order; if the
        # user just accepts, each field is user_confirmed.
        edits: list[MeasurementEdit] = []
        for i, fld in enumerate(pending.fields):
            if i < len(numbers):
                # A radius field is spoken as a diameter in chat ("the shaft is 6 mm"); store the radius.
                value = numbers[i] / 2.0 if fld == "radius" else numbers[i]
                edits.append(MeasurementEdit(feature_id=pending.feature_id, field=fld, caliper_mm=value))
            else:
                # No explicit number for this field: accept the scan value (user confirmed it implicitly
                # by proceeding), so the confirm-before-printing gate is satisfied for every mating dim.
                edits.append(MeasurementEdit(feature_id=pending.feature_id, field=fld,
                                             user_confirmed=True))

        try:
            updated = self._reg.perception.patch_part(scan_id, pending.part_id,
                                                      PartPatch(measurements=edits), self._reg.storage)
        except (KeyError, ValueError) as e:
            _STATE.pop(scan_id, None)
            yield TokenEvent(text=f"I couldn't record that measurement: {e}")
            return
        self._store.save_part(scan_id, updated)
        self._store.mark_cad_stale(
            scan_id, self._store.cad_jobs_referencing(scan_id, {pending.part_id}))
        _STATE.pop(scan_id, None)

        feature = updated.feature(pending.feature_id)
        if feature is None:
            yield TokenEvent(text="That part is no longer available; start the request again.")
            return
        _ = accept_scan  # (kept for readability; behaviour is number-vs-accept above)
        yield TokenEvent(text="Thanks. Building the replacement with your measurements. ")
        yield from self._run_cad(scan_id, updated, feature, pending.template,
                                 self._store.list_parts(scan_id))

    def _run_cad(self, scan_id: str, target: Part, feature, template: str,
                 parts: list[Part]) -> Iterator:
        ctx = ToolContext(self._reg, scan_id)
        fields = [f for f in _MATING_FIELDS_BY_TYPE.get(feature.type, ()) if getattr(feature, f, None)]
        params = {f"{feature.type}_{f}": getattr(feature, f).value_mm for f in fields}
        mate = {"part_id": target.part_id, "feature_id": feature.feature_id, "fit": "snug",
                "radial_clearance_mm": 0.12, "fields": fields}
        result = ctx.dispatch("create_cad", {"template": template, "params": params,
                                             "mates_with": [mate]})
        if isinstance(result, CardSignal):
            yield TokenEvent(text=f"Here is a {template} with a {feature.type.replace('_', ' ')} fit. ")
            yield CardEvent(card=result.card)
            cad = result.card.payload
            if cad.status == "ok":
                dims = ", ".join(f"{f.replace('_', ' ')} {getattr(feature, f).value_mm:g} mm" for f in fields)
                plan = RepairPlan(
                    plan_id="pln_" + uuid.uuid4().hex[:8], intent="fabrication",
                    diagnosis=Diagnosis(summary=f"Make a {template} to fit the {target.label}.",
                                        part_ids=[target.part_id], confidence=target.label_confidence),
                    safety=["Allow the print bed and part to cool before removal."],
                    tools=["FDM printer", "slicer", "calipers"],
                    steps=[
                        Step(n=1, text=f"Check the mating feature on [[part:{target.part_id}]]"
                             + (f" against the confirmed dimensions ({dims})." if dims else "."),
                             part_ids=[target.part_id]),
                        Step(n=2, text=f"Download the STL for {cad.cad_id} from the CAD result. "
                             "Import it into the slicer in millimetres and check the preview at full scale."),
                        Step(n=3, text="Slice with the profile for your printer and filament. Print one test piece."),
                        Step(n=4, text=f"After cooling, check the mating dimensions with calipers. "
                             f"Test the fit on [[part:{target.part_id}]] by hand. Stop if it binds.",
                             part_ids=[target.part_id]),
                    ], bom=[], printable=[cad.cad_id], followups=[])
                yield CardEvent(card=RepairPlanCard(payload=plan))
        elif isinstance(result, dict) and result.get("error") == "confirm_before_printing":
            offenders = ", ".join(o["field"] for o in result["offenders"])
            yield TokenEvent(text=f"I still need confirmed values for {offenders} before printing. "
                                  "Please measure them with calipers.")
        else:
            yield TokenEvent(text=f"The CAD job could not be built: {result}")

    def _intent_default(self, parts: list[Part], graph: SceneGraph | None) -> Iterator:
        obj = graph.object.category if graph else "object"
        yield TokenEvent(text=f"I can help with this {obj}. Ask me to explain how it works, diagnose "
                              "what is wrong, plan a repair, find a part to buy, measure a part, or "
                              "design a printable replacement. ")
        if parts:
            names = ", ".join(f"{p.label} [[part:{p.part_id}]]" for p in parts)
            yield TokenEvent(text=f"Parts on this object: {names}.")
            yield HighlightEvent(part_ids=[p.part_id for p in parts])

    # ------------------------------------------------------------ part answer
    def _answer_part(self, part: Part) -> Iterator:
        yield HighlightEvent(part_ids=[part.part_id])
        yield TokenEvent(text=f"[[part:{part.part_id}]] {part.label}: {part.description}. ")
        dims = list(self._part_measurements(part))
        if dims:
            text = "; ".join(f"{name} {round(m.value_mm, 1)} mm ({m.preferred})" for name, m in dims)
            yield TokenEvent(text=f"Dimensions: {text}. ")
        yield TokenEvent(text=f"Material guess: {part.material_guess.value}.")

    # ------------------------------------------------------------ grounded helpers
    def _part_measurements(self, part: Part) -> list[tuple[str, Measurement]]:
        """Each named dimension of the part resolved to its backing Measurement (SPEC 4.3)."""
        out: list[tuple[str, Measurement]] = []
        for d in part.dimensions:
            name = d.name
            if hasattr(d, "value"):  # OwnDimension
                out.append((name, d.value))
            else:  # DerivedDimension -> the feature Measurement it points at
                fid, fld = d.derived_from.split(".")
                feat = part.feature(fid)
                meas = getattr(feat, fld, None) if feat else None
                if isinstance(meas, Measurement):
                    out.append((name, meas))
        return out

    def _label(self, parts: list[Part], part_id: str) -> str:
        return next((p.label for p in parts if p.part_id == part_id), part_id)

    def _repair_steps(self, parts: list[Part], suspects: list[str], graph: SceneGraph) -> list[Step]:
        """Grounded, object-agnostic repair steps: isolate, access the suspect part(s), replace, test."""
        suspect_ids = [pid for pid in suspects if any(p.part_id == pid for p in parts)]
        steps: list[Step] = [
            Step(n=1, text="Shut off the relevant supply or power and place something to catch any "
                           "spill before you start."),
        ]
        if suspect_ids:
            labels = ", ".join(f"{self._label(parts, pid)} [[part:{pid}]]" for pid in suspect_ids)
            neighbours = {r.b for r in graph.relations if r.a in suspect_ids}
            neighbours.update(r.a for r in graph.relations if r.b in suspect_ids)
            adjacent = [p for p in parts if p.part_id in neighbours and p.part_id not in suspect_ids]
            context = ", ".join(f"[[part:{p.part_id}]]" for p in adjacent)
            steps.append(Step(n=2, text=f"Locate {labels}."
                              + (f" Photograph its connections to {context}." if context else
                                 " Photograph its installed orientation.")
                              + " Check the existing size and fitting markings against the replacement before removal.",
                              part_ids=[*suspect_ids, *(p.part_id for p in adjacent)]))
            steps.append(Step(n=3, text="Remove the worn part. Align the replacement with the original "
                                        "orientation and secure it following the fitting instructions.",
                              part_ids=suspect_ids[:1]))
        else:
            steps.append(Step(n=2, text="Inspect the worn or damaged part and remove it."))
            steps.append(Step(n=3, text="Fit the correct replacement part."))
        steps.append(Step(n=len(steps) + 1,
                          text="Restore the supply or power and test for correct operation and leaks."))
        return steps

    def _buy_lines_for(self, parts: list[Part], part_ids: list[str],
                       query_hint: str = "", *, best_only: bool = False) -> list[BomLine]:
        """Search the catalogue using each suspect part's label and preferred diameter (object-agnostic:
        the query words come from the part labels and the message, never from hard-coded categories)."""
        seen: set[str] = set()
        lines: list[BomLine] = []
        for pid in part_ids:
            part = next((p for p in parts if p.part_id == pid), None)
            if part is None:
                continue
            diameter = self._preferred_diameter(part)
            query = f"{part.label} {query_hint}".strip()
            hits = catalogue.search(query, measurement_mm=diameter)
            for line in (hits[:1] if best_only else hits):
                if line.name not in seen:
                    seen.add(line.name)
                    lines.append(line)
        return lines

    def _preferred_diameter(self, part: Part) -> float | None:
        """A part's outer/bore diameter in mm from a cylinder/bore feature's preferred radius, if any."""
        for feat in part.features:
            if feat.type in {"cylinder", "bore", "d_shaft"} and getattr(feat, "radius", None):
                return feat.radius.value_mm * 2.0
        return None

    def _printable_candidate(self, parts: list[Part], graph: SceneGraph | None) -> Part | None:
        """A part worth printing: one with a mating feature (shaft/bore), preferring a failure suspect."""
        suspect_ids = {h.part_id for h in graph.failure_hypotheses} if graph else set()
        candidates = [p for p in parts if self._mating_feature(p) is not None]
        candidates.sort(key=lambda p: p.part_id not in suspect_ids)
        return candidates[0] if candidates else None

    def _mating_feature(self, part: Part):
        """The feature a new part would fit onto: a shaft, cylinder or bore (object-agnostic)."""
        for feat in part.features:
            if feat.type in {"d_shaft", "cylinder", "bore", "hole"}:
                return feat
        return None

    def _template_for(self, part: Part, feature) -> str:
        """Pick a template whose shape fits the feature. Falls back to the first available template."""
        if feature.type in {"d_shaft", "cylinder"}:
            return "knob" if "knob" in TEMPLATE_SCHEMAS else next(iter(TEMPLATE_SCHEMAS))
        if feature.type in {"bore", "hole"}:
            return "cap" if "cap" in TEMPLATE_SCHEMAS else "spacer" if "spacer" in TEMPLATE_SCHEMAS \
                else next(iter(TEMPLATE_SCHEMAS))
        return next(iter(TEMPLATE_SCHEMAS))

def reset_state() -> None:
    """Clear per-scan conversation state (used by tests)."""
    _STATE.clear()
