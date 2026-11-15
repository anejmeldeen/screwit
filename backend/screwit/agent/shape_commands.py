"""Execute unambiguous editor commands directly, even without a model provider."""

from __future__ import annotations

import math
import re

from screwit.agent.add_part import PartAddedSignal
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


def shape_command(reg, scan_id, message):
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
