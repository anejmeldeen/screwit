"""Codegen mode (SPEC 3.3, 6.4 stretch): Claude designs a NEW part shape as CadQuery code, which runs
ONLY inside a disposable, locked-down container.

Public entry point for the integration layer (P5's CadService codegen branch, P4's create_cad tool):

    design(request, part, params_hint, profile, storage, out_prefix)
        -> (files_dict, rationale, attempts)

`files_dict` is CadFiles-shaped (stl/step/source/preview storage keys) so P5 can wrap it in a
CadResult; `rationale` is the short design explanation for the result card; `attempts` is the
per-candidate record (sources, sandbox results, scores) for logging and the agent's feasibility notes.

The security boundary is the container (codegen/sandbox.py), never this module. Untrusted LLM code is
never executed outside it, including in tests.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from screwit.codegen.designer import DEFAULT_K, DesignOutcome
from screwit.codegen.designer import design as _design_loop
from screwit.codegen.llm import CodegenLLM, DesignBrief, build_codegen_llm
from screwit.codegen.preview import render_stl_preview
from screwit.codegen.sandbox import DEFAULT_IMAGE

__all__ = [
    "design",
    "DesignBrief",
    "DesignOutcome",
    "build_codegen_llm",
    "run_design",
]


def run_design(brief: DesignBrief, *, out_root: Path, llm: CodegenLLM | None = None,
               params: dict | None = None, k: int = DEFAULT_K, image: str = DEFAULT_IMAGE,
               timeout_s: float = 30.0) -> DesignOutcome:
    """Run the designer loop with the configured (or supplied) LLM. Thin convenience wrapper."""
    return _design_loop(brief, llm or build_codegen_llm(), out_root=out_root, params=params, k=k,
                        image=image, timeout_s=timeout_s)


def _brief_from(request: str, params_hint: dict, profile: Any) -> DesignBrief:
    bv = tuple(float(x) for x in profile.build_volume_mm)  # type: ignore[arg-type]
    return DesignBrief(
        request=request,
        required_dims_mm={k: float(v) for k, v in (params_hint or {}).items()},
        build_volume_mm=bv,  # type: ignore[arg-type]
        min_wall_mm=float(profile.min_wall_mm),
        max_overhang_deg=float(profile.max_overhang_deg),
        notes="",
    )


def design(request: str, part: Any, params_hint: dict, profile: Any, storage: Any, out_prefix: str,
           *, llm: CodegenLLM | None = None, k: int = DEFAULT_K, image: str = DEFAULT_IMAGE,
           timeout_s: float = 30.0) -> tuple[dict | None, str, list]:
    """Design a new part shape and persist its files under `out_prefix` in `storage`.

    Returns (files_dict, rationale, attempts). `files_dict` is None when no candidate built; the
    rationale then explains why. `part` is accepted for parity with the template path (mate context)
    but the current loop drives purely off `params_hint` + `profile`; mate-aware scoring is a follow-up.
    """
    brief = _brief_from(request, params_hint, profile)
    scratch = Path(tempfile.mkdtemp(prefix="codegen_"))
    try:
        return _design_into(brief, params_hint, storage, out_prefix, scratch, llm=llm, k=k,
                            image=image, timeout_s=timeout_s)
    finally:
        # Job dirs hold untrusted container output; nothing in them is needed once persisted.
        shutil.rmtree(scratch, ignore_errors=True)


def _design_into(brief: DesignBrief, params_hint: dict, storage: Any, out_prefix: str, scratch: Path,
                 *, llm: CodegenLLM | None, k: int, image: str,
                 timeout_s: float) -> tuple[dict | None, str, list]:
    outcome = _design_loop(brief, llm or build_codegen_llm(), out_root=scratch, params=params_hint,
                           k=k, image=image, timeout_s=timeout_s)

    attempts = [
        {
            "index": c.index,
            "built": c.built,
            "repaired": c.repaired,
            "score": c.score,
            "score_detail": c.score_detail,
            "attempts": [
                {"ok": a.result.ok, "stage": a.result.stage, "error": a.result.error,
                 "repaired_from_error": a.repaired_from_error is not None}
                for a in c.attempts
            ],
        }
        for c in outcome.candidates
    ]

    if outcome.winner is None or outcome.winner.final is None:
        return None, outcome.rationale, attempts

    final = outcome.winner.final
    res = final.result

    # Persist the winner's files under out_prefix. STL/STEP come from the sandbox output dir; the
    # source is the winning program; the preview is rendered from the (trusted) STL on the CPU.
    stl_key = f"{out_prefix}/model.stl"
    step_key = f"{out_prefix}/model.step"
    source_key = f"{out_prefix}/model.py"
    preview_key = f"{out_prefix}/preview.png"

    storage.put_file(stl_key, res.stl_path, "model/stl")
    storage.put_file(step_key, res.step_path, "application/step")
    storage.put_bytes(source_key, final.source.encode("utf-8"), "text/x-python")

    preview_path = scratch / "winner_preview.png"
    render_stl_preview(res.stl_path, preview_path)
    storage.put_file(preview_key, preview_path, "image/png")

    files = {"stl": stl_key, "step": step_key, "source": source_key, "preview": preview_key}
    return files, outcome.rationale, attempts
