"""The agent tool layer (SPEC 6.3 table, 7 P4 #3).

Each tool is dispatched against P3/P5 through the registry and the ScanStore, so the same code serves
mock mode (fakes), tests (fake LLM) and live. Tools that need parts return a "not ready yet" marker
before the parts stage finishes (SPEC 6.3 before-parts mode).

The tool set is exactly the SPEC 6.3 table:
  get_scene_graph, get_part, list_parts, measure, render_view, get_keyframes, highlight,
  set_measurement, search_parts, create_cad, emit_repair_plan.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from screwit.agent import catalogue
from screwit.agent.cad import TEMPLATE_SCHEMAS, ConfirmBeforePrinting, check_confirm_before_printing
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import (
    CadRequest,
    MeasureRequest,
    Part,
    PartPatch,
    RepairPlan,
)
from screwit.contracts.models import FEATURE_MEASUREMENT_FIELDS, MeasurementEdit

NOT_READY = {"status": "not_ready", "message": "parts are still being computed; try again shortly"}

# Tools that need the parts stage to have finished.
_NEEDS_PARTS = frozenset({"get_part", "list_parts", "render_view", "set_measurement", "create_cad"})


@dataclass
class HighlightSignal:
    """Returned by highlight() so the loop can emit a HighlightEvent to the viewer."""

    part_ids: list[str]


@dataclass
class CardSignal:
    """Returned by emit_repair_plan() / create_cad() so the loop can emit a CardEvent."""

    card: Any


def _obj(properties: dict | None = None, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return schema


_STR = {"type": "string"}
_NUM = {"type": "number"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}


def _plan_input_schema() -> dict:
    schema = RepairPlan.model_json_schema()
    definitions = schema.pop("$defs", {})
    return {**_obj({"plan": schema}, ["plan"]), "$defs": definitions}


def anthropic_tool_specs() -> list[dict]:
    """The tools, in Anthropic Messages API tool-schema form (exactly the SPEC 6.3 table)."""
    names = sorted(TEMPLATE_SCHEMAS)
    create_desc = (
        "Start a CAD job and return the checks. Two modes:\n"
        f"- mode='template' (default): pick one of the fixed templates ({', '.join(names)}) and fill "
        "its params. Use this whenever a template fits the requested shape.\n"
        "- mode='codegen': for a NEW shape no template covers, omit `template` and give a clear, "
        "self-contained `codegen_prompt` describing the part to build (overall shape, key dimensions "
        "in mm, and every mating feature). An X7 CAD service has Claude write and sandbox-run CadQuery "
        "code from that prompt.\n"
        "In BOTH modes the confirm-before-printing rule applies: every mating dimension must already be "
        "caliper-measured or user-confirmed, or the call is refused."
    )
    return [
        {"name": "get_scene_graph",
         "description": "The scene graph (object, parts, relations, function, failure hypotheses).",
         "input_schema": _obj()},
        {"name": "get_part",
         "description": "Full record for one part, with features and measured dimensions.",
         "input_schema": _obj({"part_id": _STR}, ["part_id"])},
        {"name": "list_parts",
         "description": "Short list of detected parts (id, label, confidence).",
         "input_schema": _obj()},
        {"name": "measure",
         "description": "Distance in mm between two points ([x,y,z] mm) or two part ids.",
         "input_schema": _obj({"a": {}, "b": {}}, ["a", "b"])},
        {"name": "render_view",
         "description": "Render the given parts from an angle (degrees) for visual reasoning.",
         "input_schema": _obj({"part_ids": _STR_LIST, "angle": _NUM}, ["part_ids"])},
        {"name": "get_keyframes",
         "description": "Relevant raw keyframes (close-ups) for the object.",
         "input_schema": _obj({"query": _STR})},
        {"name": "highlight",
         "description": "Highlight parts in the viewer.",
         "input_schema": _obj({"part_ids": _STR_LIST}, ["part_ids"])},
        {"name": "set_measurement",
         "description": "Write a caliper value or user confirmation for a feature measurement "
                        "(bumps the part revision).",
         "input_schema": _obj({"part_id": _STR, "feature_id": _STR, "field": _STR,
                               "caliper_mm": _NUM, "user_confirmed": {"type": "boolean"}},
                              ["part_id", "feature_id", "field"])},
        {"name": "search_parts",
         "description": "Search the curated catalogue for purchasable items; pass a measured "
                        "diameter in mm to size-match.",
         "input_schema": _obj({"query": _STR, "measurement_mm": _NUM, "category": _STR}, ["query"])},
        {"name": "create_cad",
         "description": create_desc,
         "input_schema": _obj({"mode": {"type": "string", "enum": ["template", "codegen"],
                                        "default": "template"},
                               "template": {"type": "string", "enum": names,
                                            "description": "required for mode='template'"},
                               "codegen_prompt": {"type": "string",
                                                  "description": "required for mode='codegen': the new "
                                                  "shape to build, with dimensions in mm"},
                               "params": {"type": "object"},
                               "mates_with": {"type": "array", "items": {"type": "object"}},
                               "material": _STR, "printer_profile": _STR})},
        {"name": "emit_repair_plan",
         "description": "Validate and emit a Repair Plan card (diagnosis, safety, steps, BOM).",
         "input_schema": _plan_input_schema()},
    ]


class ToolContext:
    """Binds the tools to one scan. Returns JSON-able results (plus Highlight/Card signals)."""

    def __init__(self, reg: Registry, scan_id: str) -> None:
        self._reg = reg
        self._store = ScanStore(reg.data)
        self._scan_id = scan_id
        self._cad_seq = 0

    def _parts_ready(self) -> bool:
        scan = self._store.get_scan(self._scan_id)
        return scan is not None and scan.stage("parts").state == "done"

    def dispatch(self, name: str, args: dict) -> Any:
        if name in _NEEDS_PARTS and not self._parts_ready():
            return NOT_READY
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool {name}"}
        return handler(args)

    # -------------------------------------------------------------- tools
    def _tool_get_scene_graph(self, args: dict) -> Any:
        graph = self._store.get_scene_graph(self._scan_id)
        return graph.model_dump(mode="json") if graph else {"error": "no scene graph"}

    def _tool_get_part(self, args: dict) -> Any:
        part = self._store.get_part(self._scan_id, args["part_id"])
        return part.model_dump(mode="json") if part else {"error": "unknown part"}

    def _tool_list_parts(self, args: dict) -> Any:
        return [{"part_id": p.part_id, "label": p.label, "label_confidence": p.label_confidence}
                for p in self._store.list_parts(self._scan_id)]

    def _tool_measure(self, args: dict) -> Any:
        req = MeasureRequest(a=_coerce_point(args["a"]), b=_coerce_point(args["b"]))
        resp = self._reg.perception.measure(self._scan_id, req, self._reg.storage)
        return resp.model_dump(mode="json")

    def _tool_render_view(self, args: dict) -> Any:
        png = self._reg.perception.render_view(self._scan_id, args["part_ids"],
                                                float(args.get("angle", 0.0)), self._reg.storage)
        return {"image_base64": base64.b64encode(png).decode(), "media_type": "image/png"}

    def _tool_get_keyframes(self, args: dict) -> Any:
        recon = self._store.get_reconstruction(self._scan_id)
        frames = recon.keyframes if recon else []
        return {"keyframes": [self._reg.storage.signed_url(k) for k in frames]}

    def _tool_highlight(self, args: dict) -> Any:
        return HighlightSignal(part_ids=list(args["part_ids"]))

    def _tool_set_measurement(self, args: dict) -> Any:
        edit = MeasurementEdit(feature_id=args["feature_id"], field=args["field"],
                               caliper_mm=args.get("caliper_mm"), user_confirmed=args.get("user_confirmed"))
        patch = PartPatch(measurements=[edit])
        try:
            updated = self._reg.perception.patch_part(self._scan_id, args["part_id"], patch, self._reg.storage)
        except (KeyError, ValueError) as e:
            return {"error": str(e)}
        self._store.save_part(self._scan_id, updated)
        self._store.mark_cad_stale(
            self._scan_id, self._store.cad_jobs_referencing(self._scan_id, {args["part_id"]}))
        feat = updated.feature(args["feature_id"])
        meas = getattr(feat, args["field"], None) if feat else None
        return {"part_id": updated.part_id, "measurement_revision": updated.measurement_revision,
                "measurement": meas.model_dump(mode="json") if meas else None}

    def _tool_search_parts(self, args: dict) -> Any:
        lines = catalogue.search(args["query"], args.get("measurement_mm"), args.get("category"))
        return [line.model_dump(mode="json") for line in lines]

    def _tool_create_cad(self, args: dict) -> Any:
        parts = {p.part_id: p for p in self._store.list_parts(self._scan_id)}
        mode = args.get("mode") or ("codegen" if args.get("codegen_prompt") and not args.get("template")
                                    else "template")
        if mode == "template" and not args.get("template"):
            return {"error": "bad_request", "message": "mode='template' needs a template name"}
        if mode == "codegen" and not args.get("codegen_prompt"):
            return {"error": "bad_request", "message": "mode='codegen' needs a codegen_prompt"}

        mates = args.get("mates_with") or []
        # Confirm-before-printing applies to EVERY mating dimension in both modes (SPEC 6.3).
        mating_fields = _mating_fields(parts, mates)
        try:
            check_confirm_before_printing(parts, mating_fields)
        except ConfirmBeforePrinting as e:
            return {"error": "confirm_before_printing", "message": str(e), "offenders":
                    [{"part_id": p, "feature_id": f, "field": fld, "source": src} for p, f, fld, src in e.offenders]}

        req = self._build_cad_request(args, parts, mates, mode)
        result = self._reg.cad.run(req, parts, self._reg.storage)
        self._store.save_cad_job(req, result)
        from screwit.contracts import CadResultCard

        return CardSignal(card=CadResultCard(payload=result))

    def _tool_emit_repair_plan(self, args: dict) -> Any:
        plan = RepairPlan.model_validate(args["plan"])
        from screwit.contracts import RepairPlanCard

        return CardSignal(card=RepairPlanCard(payload=plan))

    # -------------------------------------------------------------- helpers
    def _build_cad_request(self, args: dict, parts: dict, mates: list[dict], mode: str) -> CadRequest:
        from screwit.contracts import Mate, Placement

        scan = self._store.get_scan(self._scan_id)
        self._cad_seq += 1
        cad_id = f"cad_{self._scan_id}_{self._cad_seq:03d}"
        params = {k: float(v) for k, v in (args.get("params") or {}).items()}
        param_sources = {k: "user_confirmed" for k in params}  # confirm-guard already passed.
        mate_models = [Mate(part_id=m["part_id"], feature_id=m["feature_id"],
                            fit=m.get("fit", "snug"), radial_clearance_mm=float(m.get("radial_clearance_mm", 0.12)))
                       for m in mates]
        pmr = {m["part_id"]: parts[m["part_id"]].measurement_revision for m in mates if m["part_id"] in parts}
        return CadRequest(
            cad_id=cad_id, scan_id=self._scan_id,
            reconstruction_revision=scan.reconstruction_revision if scan else 0,
            part_measurement_revisions=pmr, mode=mode,
            template=args.get("template") if mode == "template" else None,
            params=params, param_sources=param_sources,
            codegen_prompt=args.get("codegen_prompt") if mode == "codegen" else None,
            mates_with=mate_models,
            placement=Placement(translate_mm=(0.0, 0.0, 0.0), rotation_quat=(0.0, 0.0, 0.0, 1.0)),
            material=args.get("material", "PETG"),
            printer_profile=args.get("printer_profile", "prusa_mk4_petg_0.2"),
        )


def _coerce_point(ref):
    if isinstance(ref, str):
        return ref
    return tuple(float(x) for x in ref)


def _mating_fields(parts: dict[str, Part], mates: list[dict]) -> list[tuple[str, str, str]]:
    """The feature measurement fields each mate consumes, for the confirm-before-printing guard.

    Mode-independent (template and codegen both mate against the scanned part's features). If a mate
    names explicit `fields`, those are used; otherwise every measurement field the ACTUAL target
    feature carries (SPEC 4.3 FEATURE_MEASUREMENT_FIELDS) is checked, since any can drive a fit.
    """
    out: list[tuple[str, str, str]] = []
    for mate in mates:
        part_id, feature_id = mate["part_id"], mate["feature_id"]
        fields = mate.get("fields")
        if fields:
            out.extend((part_id, feature_id, f) for f in fields)
        else:
            part = parts.get(part_id)
            feat = part.feature(feature_id) if part else None
            feature_fields = FEATURE_MEASUREMENT_FIELDS.get(feat.type, ()) if feat else \
                tuple({f for fs in FEATURE_MEASUREMENT_FIELDS.values() for f in fs})
            out.extend((part_id, feature_id, f) for f in feature_fields)
    # Dedupe preserving order.
    seen: set[tuple[str, str, str]] = set()
    uniq: list[tuple[str, str, str]] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq
