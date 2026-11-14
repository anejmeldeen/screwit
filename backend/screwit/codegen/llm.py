"""The LLM seam for codegen: propose candidate programs, repair a failure, pick/refine with vision.

Two implementations, chosen by SCREWIT_LLM (AGENTS.md decision 4, screwit.llm):
  - AnthropicCodegenLLM: real Claude (bedrock/anthropic) via the anthropic SDK Messages API.
  - FakeCodegenLLM: deterministic canned programs for tests and SCREWIT_LLM=fake, including a
    candidate that fails first and is repaired, so the whole designer loop is testable with no network.

All prompts are object-agnostic: they describe a part request and constraints, never a specific object
category. A program is always `def build(params) -> cq.Workplane` against the fixed CadQuery API.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from screwit.llm import config

_MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "You design 3D-printable parts as CadQuery programs. Write a single Python function "
    "`def build(params) -> cq.Workplane` that returns one solid in millimetres. You may import only "
    "`cadquery as cq` and `math`; no other imports, no file, network or system access. Read every "
    "dimension from the `params` dict with sensible fallbacks. Keep walls at or above the given "
    "minimum wall thickness and the whole part inside the given build volume. Return ONLY the code, "
    "no prose, no markdown fence."
)


@dataclass(frozen=True)
class DesignBrief:
    """Everything the LLM needs to design a part, object-agnostic."""

    request: str                       # natural-language ask ("a cap with a 3 mm drain hole")
    required_dims_mm: dict[str, float]  # dims that must appear within tolerance (mating = caliper/confirmed)
    build_volume_mm: tuple[float, float, float]
    min_wall_mm: float
    max_overhang_deg: float
    notes: str = ""                    # any extra guidance (fit, orientation)

    def as_prompt(self) -> str:
        dims = ", ".join(f"{k}={v} mm" for k, v in self.required_dims_mm.items()) or "(none specified)"
        bv = "x".join(f"{d:g}" for d in self.build_volume_mm)
        return (
            f"Design request: {self.request}\n"
            f"Required dimensions (must be met within 0.3 mm): {dims}\n"
            f"Build volume: {bv} mm. Minimum wall thickness: {self.min_wall_mm} mm. "
            f"Maximum unsupported overhang: {self.max_overhang_deg} deg.\n"
            f"{self.notes}".strip()
        )


class CodegenLLM(Protocol):
    def propose(self, brief: DesignBrief, k: int) -> list[str]:
        """Return k diverse candidate programs (full `build(params)` source)."""

    def repair(self, brief: DesignBrief, source: str, traceback: str) -> str:
        """Given a failing program and its traceback, return a repaired program."""

    def pick(self, brief: DesignBrief, candidates: list[tuple[str, Path]]) -> int:
        """Given (rationale-label, preview-png) pairs, return the index of the best candidate."""


# --------------------------------------------------------------------------- real

def _strip_fence(text: str) -> str:
    """Pull a code body out of a reply, tolerating a ```python fence if the model adds one."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


class AnthropicCodegenLLM:
    """Real Claude codegen LLM (bedrock/anthropic). Lazy-imports the client from screwit.llm."""

    def __init__(self) -> None:
        from screwit.llm import client

        self._client = client()
        self._model = config().model

    def _complete(self, prompt: str, blocks: list | None = None) -> str:
        content: list = [{"type": "text", "text": prompt}]
        if blocks:
            content.extend(blocks)
        msg = self._client.messages.create(
            model=self._model, max_tokens=_MAX_TOKENS, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def propose(self, brief: DesignBrief, k: int) -> list[str]:
        out: list[str] = []
        approaches = [
            "Approach A: build it from primitive solids unioned together.",
            "Approach B: build a 2D profile and extrude/revolve it.",
            "Approach C: start from a bounding block and cut features out.",
        ]
        for i in range(k):
            hint = approaches[i % len(approaches)]
            out.append(_strip_fence(self._complete(f"{brief.as_prompt()}\n{hint}")))
        return out

    def repair(self, brief: DesignBrief, source: str, traceback: str) -> str:
        prompt = (
            f"{brief.as_prompt()}\n\nThis program failed. Fix it and return the full corrected "
            f"`build(params)` only.\n\n--- program ---\n{source}\n\n--- traceback ---\n{traceback}"
        )
        return _strip_fence(self._complete(prompt))

    def pick(self, brief: DesignBrief, candidates: list[tuple[str, Path]]) -> int:
        blocks: list = []
        for i, (label, png) in enumerate(candidates):
            blocks.append({"type": "text", "text": f"Candidate {i}: {label}"})
            data = base64.standard_b64encode(Path(png).read_bytes()).decode("ascii")
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": data}})
        prompt = (
            f"{brief.as_prompt()}\n\nHere are rendered previews of the best candidates. Reply with a "
            f"JSON object {{\"choice\": <index>}} naming the candidate that best matches the request."
        )
        reply = self._complete(prompt, blocks)
        try:
            return int(json.loads(_first_json(reply))["choice"])
        except Exception:  # noqa: BLE001 - a malformed pick just keeps the current best (index 0)
            return 0


def _first_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else "{}"


# --------------------------------------------------------------------------- fake

# A valid box program, a hole-in-a-plate program, and a program that fails first (bad attribute) then
# is repaired. The fake is deterministic so the designer-loop test can assert the repair happened.
_FAKE_BOX = (
    "def build(params):\n"
    "    import cadquery as cq\n"
    "    w = params.get('width', 20.0)\n"
    "    d = params.get('depth', 20.0)\n"
    "    h = params.get('height', 10.0)\n"
    "    return cq.Workplane('XY').box(w, d, h)\n"
)
_FAKE_CAP = (
    "def build(params):\n"
    "    import cadquery as cq\n"
    "    outer = params.get('outer_diameter', 30.0)\n"
    "    hole = params.get('hole_diameter', 3.0)\n"
    "    h = params.get('height', 12.0)\n"
    "    body = cq.Workplane('XY').circle(outer / 2).extrude(h)\n"
    "    return body.faces('>Z').workplane().hole(hole)\n"
)
_FAKE_BROKEN = (
    "def build(params):\n"
    "    import cadquery as cq\n"
    "    # BUG: Workplane has no method `bOx`; raises AttributeError on first run.\n"
    "    return cq.Workplane('XY').bOx(10, 10, 10)\n"
)
_FAKE_REPAIRED = (
    "def build(params):\n"
    "    import cadquery as cq\n"
    "    return cq.Workplane('XY').box(10, 10, 10)\n"
)


@dataclass
class FakeCodegenLLM:
    """Deterministic codegen LLM for tests and SCREWIT_LLM=fake.

    `propose` returns k programs: a good one, a hole-in-plate one, and one that FAILS first. `repair`
    turns the known-broken program into a working one (and records that it was asked to). `pick`
    always returns index 0, the first survivor.
    """

    repair_calls: int = 0

    def propose(self, brief: DesignBrief, k: int) -> list[str]:
        canned = [_FAKE_BOX, _FAKE_CAP, _FAKE_BROKEN]
        return [canned[i % len(canned)] for i in range(k)]

    def repair(self, brief: DesignBrief, source: str, traceback: str) -> str:
        self.repair_calls += 1
        if "bOx" in source:
            return _FAKE_REPAIRED
        return source  # nothing known to fix; return unchanged so the loop can give up

    def pick(self, brief: DesignBrief, candidates: list[tuple[str, Path]]) -> int:
        return 0


def build_codegen_llm() -> CodegenLLM:
    """Pick the codegen LLM for the current configuration (AGENTS.md decision 4)."""
    if config().provider == "fake":
        return FakeCodegenLLM()
    return AnthropicCodegenLLM()
