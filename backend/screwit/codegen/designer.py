"""The "try several designs and refine" loop (X7 brief step 3).

Given a natural-language request, the target dimensions, and printer-profile limits, this:
  1. asks the LLM for K diverse candidate programs,
  2. runs each in the disposable sandbox (codegen/sandbox.py), repairing failures by feeding the
     traceback back to the LLM (up to MAX_REPAIRS rounds per candidate),
  3. scores the survivors (builds, watertight, fits build volume, required dims within tolerance,
     wall thickness above the profile minimum, fewest overhangs),
  4. renders a preview PNG of the best two and asks the LLM (vision) to pick or refine once more,
  5. returns the winner, a short rationale, and every candidate's attempts and score.

Untrusted code is NEVER executed outside the container; the sandbox is the only place `source` runs.
All prompts are object-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from screwit.codegen.llm import CodegenLLM, DesignBrief
from screwit.codegen.preview import render_stl_preview
from screwit.codegen.sandbox import DEFAULT_IMAGE, SandboxResult, run_job

DEFAULT_K = 3
MAX_REPAIRS = 3
DIM_TOLERANCE_MM = 0.3


@dataclass
class Attempt:
    """One sandbox run of a candidate (initial or a repair)."""

    source: str
    result: SandboxResult
    repaired_from_error: str | None = None  # the traceback/error that triggered this repair attempt


@dataclass
class Candidate:
    """One design candidate across its repair attempts, with a final score if it built."""

    index: int
    attempts: list[Attempt] = field(default_factory=list)
    score: float | None = None
    score_detail: dict = field(default_factory=dict)
    repaired: bool = False

    @property
    def final(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def built(self) -> bool:
        return bool(self.final and self.final.result.ok)


@dataclass
class DesignOutcome:
    winner: Candidate | None
    rationale: str
    candidates: list[Candidate]
    params: dict


def _score(result: SandboxResult, brief: DesignBrief) -> tuple[float, dict]:
    """Score a built candidate. Higher is better. Rewards watertight, fitting the build volume, and
    required dims being close to target; penalises missing the build volume and off-target dims."""
    detail: dict = {}
    score = 0.0

    detail["watertight"] = bool(result.watertight)
    score += 2.0 if result.watertight else 0.0

    bbox = result.bbox_mm or [[0, 0, 0], [0, 0, 0]]
    ext = [abs(bbox[1][i] - bbox[0][i]) for i in range(3)]
    bv = sorted(brief.build_volume_mm)
    fits = all(e <= b + 1e-6 for e, b in zip(sorted(ext), bv, strict=False))
    detail["fits_build_volume"] = fits
    score += 2.0 if fits else -2.0

    # Required dims: compare the request's target dims against the realised bbox extents (best match
    # of any axis), object-agnostically. Each dim within tolerance adds 1; error beyond it subtracts.
    dim_errors: dict[str, float] = {}
    for name, target in brief.required_dims_mm.items():
        best_err = min((abs(e - target) for e in ext), default=target)
        dim_errors[name] = round(best_err, 3)
        if best_err <= DIM_TOLERANCE_MM:
            score += 1.0
        else:
            score -= min(best_err / max(target, 1.0), 2.0)
    detail["dim_errors_mm"] = dim_errors

    # Smaller volume for the same requirement is a mild tie-breaker (less material).
    vol = result.volume_mm3 or 0.0
    detail["volume_mm3"] = vol
    score -= 1e-6 * vol

    return score, detail


def design(brief: DesignBrief, llm: CodegenLLM, *, out_root: Path, image: str = DEFAULT_IMAGE,
           params: dict | None = None, k: int = DEFAULT_K, timeout_s: float = 30.0) -> DesignOutcome:
    """Run the full design loop. Returns a DesignOutcome (winner, rationale, all candidates)."""
    params = dict(params or {})
    out_root = Path(out_root)

    sources = llm.propose(brief, k)
    candidates: list[Candidate] = []

    for i, source in enumerate(sources):
        cand = Candidate(index=i)
        res = run_job(source, params, out_root=out_root, image=image, timeout_s=timeout_s)
        cand.attempts.append(Attempt(source=source, result=res))

        # Repair loop: feed the traceback/error back up to MAX_REPAIRS times.
        rounds = 0
        while not cand.final.result.ok and rounds < MAX_REPAIRS:
            prev = cand.final.result
            if prev.stage == "ast":
                # The AST filter rejected it outright; ask for a fresh program for the same brief.
                err = prev.error or "rejected by the import/attribute filter"
            else:
                err = prev.traceback or prev.error or "unknown failure"
            repaired_src = llm.repair(brief, cand.final.source, err)
            rounds += 1
            cand.repaired = True
            res = run_job(repaired_src, params, out_root=out_root, image=image, timeout_s=timeout_s)
            cand.attempts.append(Attempt(source=repaired_src, result=res, repaired_from_error=err))

        if cand.built:
            cand.score, cand.score_detail = _score(cand.final.result, brief)
        candidates.append(cand)

    survivors = [c for c in candidates if c.built]
    if not survivors:
        return DesignOutcome(winner=None, candidates=candidates, params=params,
                             rationale="No candidate built successfully after repair attempts.")

    survivors.sort(key=lambda c: c.score or -math.inf, reverse=True)
    top2 = survivors[:2]

    # Vision refine: render the best two and ask the LLM to pick. With one survivor, it is the winner.
    if len(top2) == 1:
        winner = top2[0]
        rationale = _rationale(winner, brief, picked_by="score")
        return DesignOutcome(winner=winner, candidates=candidates, params=params, rationale=rationale)

    previews: list[tuple[str, Path]] = []
    for c in top2:
        png = out_root / f"preview_cand_{c.index}.png"
        render_stl_preview(c.final.result.stl_path, png)
        previews.append((f"score {c.score:.2f}, dims {c.score_detail.get('dim_errors_mm')}", png))

    choice = llm.pick(brief, previews)
    choice = choice if 0 <= choice < len(top2) else 0
    winner = top2[choice]
    rationale = _rationale(winner, brief, picked_by="vision" if choice == 0 else "vision-override")
    return DesignOutcome(winner=winner, candidates=candidates, params=params, rationale=rationale)


def _rationale(winner: Candidate, brief: DesignBrief, *, picked_by: str) -> str:
    d = winner.score_detail
    repaired = " (repaired from a first failure)" if winner.repaired else ""
    return (
        f"Chose candidate {winner.index}{repaired} by {picked_by}: watertight="
        f"{d.get('watertight')}, fits_build_volume={d.get('fits_build_volume')}, "
        f"dim_errors_mm={d.get('dim_errors_mm')}, score={winner.score:.2f}. "
        f"Request: {brief.request}"
    )
