"""Execute unambiguous editor commands directly, even without a model provider."""

from __future__ import annotations

import math
import re

from screwit.agent.add_part import PartAddedSignal, _surface_point
from screwit.agent.tools import ToolContext
from screwit.app.store import ScanStore
from screwit.contracts import HighlightEvent, TokenEvent, ToolCallEvent

_SHAPES = {"box": "box", "cube": "box", "shelf": "box", "sphere": "sphere", "ball": "sphere",
           "cylinder": "cylinder", "cone": "cone", "triangular prism": "triangular_prism",
           "cat ear": "cat_ear"}
_NUM = r"\d+(?:\.\d+)?"
_SIZE = rf"{_NUM}(?:\s*[x×]\s*{_NUM}\s*[x×]\s*{_NUM})?\s*(?:mm|cm)"
_COMMAND = re.compile(
    rf"(?:please\s+)?(?:can you\s+)?(?:add|place|put|build|create|generate|make)\s+"
    rf"(?:(?P<count>a pair of|a|an|one|two|2)\s+)?(?:(?P<size>{_SIZE})\s+)?"
    rf"(?P<shape>{'|'.join(_SHAPES)})(?P<plural>s?)"
    rf"(?:\s+(?:on|onto|to|at|under|below)\s+(?P<location>.+?))?[.!?]*", re.I)


def headphone_ear_command(reg, scan_id, message):
    """Recognise requests for the headphone accessory before the general chat router."""
    text = message.lower().strip()
    if not re.search(r"\b(?:cat|kitty|kitten)[ -]+ears?\b", text):
        return None
    if re.search(r"\b(?:not|never|remove|delete|hide|without|instead|explain|why|how)\b|n['’]t\b", text):
        return None
    if not (re.search(r"\b(?:add|attach|put|place|give|make|create|generate|want|install|fit)\b", text)
            or re.fullmatch(r"(?:cat|kitty|kitten)[ -]+ears[.! ]*", text)):
        return None
    store = ScanStore(reg.data)
    parts = store.list_parts(scan_id)
    graph = store.get_scene_graph(scan_id)
    if not graph or not re.search(r"headphones?|headset", graph.object.category, re.I):
        return None
    band = next((p for p in parts if "headband" in p.label.lower().replace(" ", "")), None)
    if band is None:
        return None
    return _fit_headphone_ears(reg, scan_id, parts, band)


def _fit_headphone_ears(reg, scan_id, parts, band):
    import numpy as np

    ids = []
    _, depth, height = band.obb.extent
    # The example headphones lie flat, with the crown pointing along +Y.
    # Upright scans use +Z. Sample the local slope on each side independently.
    flat = depth > height
    crown, axis = ("back", 1) if flat else ("top", 2)
    for side, name in [(-1, "Left cat ear"), (1, "Right cat ear")]:
        existing = next((p for p in parts if p.label.casefold() == name.casefold()), None)
        if existing:
            ids.append(existing.part_id)
            continue
        u = side * .53
        point, _, _ = _surface_point(reg, band, crown, [u, 0])
        before, _, _ = _surface_point(reg, band, crown, [u-.025, 0])
        after, _, _ = _surface_point(reg, band, crown, [u+.025, 0])
        tangent = after-before
        tangent[3-axis] = 0
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        front = np.array([0., 0., -1.]) if flat else np.array([0., 1., 0.])
        outward = np.cross(tangent, front)
        # The supplied STL is 55.07 mm tall; its curved saddle meets the band at z=2.0694 mm on its
        # centreline. It keeps its own size (no size_mm); its height runs along the outward normal and
        # its width along the band.
        height = 55.06941223
        center = point + outward * (height/2 - 2.06941223)
        kind = "headphone_ear_left" if side < 0 else "headphone_ear_right"
        args = {"label": name, "kind": kind,
                "description": "White headphone accessory from the supplied ear STL.",
                "center_mm": center.tolist(), "up": outward.tolist(), "width_axis": tangent.tolist(),
                "attach_to": band.part_id, "relation": "fastened"}
        yield ToolCallEvent(name="add_part", args=args)
        result = ToolContext(reg, scan_id).dispatch("add_part", args)
        if not isinstance(result, PartAddedSignal):
            yield TokenEvent(text=result.get("message", "The headband mesh is unavailable."))
            return
        ids.append(result.part_id)
    yield TokenEvent(text="The headband now has a pair of preset cat ears: " +
                     " and ".join(f"[[part:{pid}]]" for pid in ids) + ".")
    yield HighlightEvent(part_ids=ids)


def shape_command(reg, scan_id, message):
    preset = headphone_ear_command(reg, scan_id, message)
    if preset is not None:
        return preset
    match = _COMMAND.fullmatch(message.strip().lower())
    if not match:
        return None
    location = match["location"] or ""
    # Leave compound instructions and other modifiers to the language model.
    if re.search(r"\b(and|then|except|but|without|with)\b", location):
        return None
    parts = ScanStore(reg.data).list_parts(scan_id)
    if not parts:
        return None
    face = "bottom" if re.search(r"\b(under|below)\b", message, re.I) else "top"
    side = re.match(r"(?:the )?(top|bottom|left|right|front|back)(?: side)?(?: of |$)", location)
    if side:
        face, location = side[1], location[side.end():]
    target = re.sub(r"^(?:the|my|our|these) ", "", location).strip()
    shape = match["shape"]
    candidates = [p for p in parts if target in (p.label.lower(), p.part_id.lower(), f"[[part:{p.part_id}]]")]
    if target and not candidates:
        candidates = [p for p in parts if target in p.label.lower().split()]
    if not candidates:
        graph = ScanStore(reg.data).get_scene_graph(scan_id)
        generic = {"", "model", "build", "object", "assembly", "it"}
        if graph:
            generic.update({graph.object.category.lower(), graph.object.category.lower().split()[-1]})
        if target not in generic:
            return None
        candidates = [p for p in parts if shape == "cat ear" and "headband" in p.label.lower()] or parts
    elif len(candidates) > 1:
        return None
    anchor = max(candidates, key=lambda p: math.prod(p.obb.extent))
    base = max(1., min(500., max(anchor.obb.extent)*.18))
    size = [base]*3
    if shape == "cat ear":
        size = [base, base*.4, base]
    elif shape == "shelf":
        size = [base*3, base*1.5, base*.12]
    if match["size"]:
        numbers = [float(n) * (10 if match["size"].endswith("cm") else 1) for n in re.findall(_NUM, match["size"])]
        size = numbers if len(numbers) == 3 else [numbers[0]]*3
    paired = match["count"] in {"a pair of", "two", "2"} or (shape == "cat ear" and match["plural"])
    if match["plural"] and not paired:
        return None

    def execute():
        ids = []
        for offset in ([-.48, .48] if paired else [0]):
            args = {"label": shape.capitalize(), "kind": _SHAPES[shape], "size_mm": size,
                    "attach_to": anchor.part_id, "surface": face, "offset_uv": [offset, 0]}
            yield ToolCallEvent(name="add_part", args=args)
            result = ToolContext(reg, scan_id).dispatch("add_part", args)
            if not isinstance(result, PartAddedSignal):
                yield TokenEvent(text=str(result.get("message", "The workspace geometry is not ready yet.")))
                return
            ids.append(result.part_id)
            yield TokenEvent(text=f"Added [[part:{result.part_id}]] on the {face} of {anchor.label} "
                             f"({result.result['shape']}). ")
        yield HighlightEvent(part_ids=ids)

    return execute()
