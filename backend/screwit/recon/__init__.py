"""Feed-forward reconstruction pipeline (SPEC sections 3.1, 6.1; owner: P2).

Stage order on the GPU worker: frames -> geometry -> scale -> object_mask -> splat + mesh -> parts.
Everything here works in the scan frame (right-handed, millimetres, Z up); the single scan-to-web
transform lives in the contracts, not here. No training and no per-scene optimization (SPEC 1.5).
"""

from __future__ import annotations

__all__ = ["ReconPipelineImpl"]


def __getattr__(name: str):
    # Lazy import so importing a single submodule (e.g. gaussians) does not pull in the whole
    # pipeline and its optional heavy dependencies.
    if name == "ReconPipelineImpl":
        from .pipeline import ReconPipelineImpl

        return ReconPipelineImpl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
