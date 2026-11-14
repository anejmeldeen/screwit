"""In-container runner for one untrusted CadQuery codegen job (SPEC 3.3 "CAD code execution").

Reads a single JSON job on stdin: {"source": "<python>", "params": {...}}. Executes the LLM-written
`build(params) -> cq.Workplane` with only the CadQuery API in scope, exports STL and STEP into /out,
and prints a one-line JSON summary (volume, bbox, watertight, errors with traceback) on stdout.

This file runs INSIDE the disposable container as uid 65534. It is the trusted side of an untrusted
job: the security boundary is the container (no network, read-only root, dropped caps, tmpfs, limits),
not anything this script does. The restricted exec scope here is defence in depth, not the boundary.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

OUT = Path("/out")


def _summary(ok, error=None, tb=None, volume=None, bbox=None, watertight=None,
             stl=None, step=None):
    return {
        "ok": ok,
        "error": error,
        "traceback": tb,
        "volume_mm3": volume,
        "bbox_mm": bbox,
        "watertight": watertight,
        "stl": stl,
        "step": step,
    }


def _run(job):
    import cadquery as cq  # imported here so an import failure is reported as a job error.
    import math

    source = job.get("source")
    params = job.get("params") or {}
    if not isinstance(source, str) or not source.strip():
        return _summary(False, error="job 'source' must be a non-empty string")
    if not isinstance(params, dict):
        return _summary(False, error="job 'params' must be an object")

    # Only the CadQuery API (and math) are put in scope. __builtins__ is left as the default so the
    # LLM can use ordinary Python; the container, not this dict, is what contains a hostile program.
    scope: dict = {"cq": cq, "cadquery": cq, "math": math}
    try:
        exec(compile(source, "<codegen>", "exec"), scope)  # noqa: S102 - sandboxed by the container
    except Exception as e:  # noqa: BLE001
        return _summary(False, error=f"source failed to load: {e!r}", tb=traceback.format_exc())

    build = scope.get("build")
    if not callable(build):
        return _summary(False, error="source must define a callable build(params)")

    try:
        result = build(params)
    except Exception as e:  # noqa: BLE001
        return _summary(False, error=f"build(params) raised: {e!r}", tb=traceback.format_exc())

    # Accept a Workplane or a Shape/Solid; normalise to something exporters.export handles.
    try:
        from cadquery import exporters

        OUT.mkdir(parents=True, exist_ok=True)
        stl_path = OUT / "model.stl"
        step_path = OUT / "model.step"
        exporters.export(result, str(stl_path))
        exporters.export(result, str(step_path))
    except Exception as e:  # noqa: BLE001
        return _summary(False, error=f"export failed: {e!r}", tb=traceback.format_exc())

    # Measure with trimesh off the exported STL (same mesh the checks will see).
    try:
        import trimesh

        mesh = trimesh.load(str(stl_path), force="mesh")
        bbox = mesh.bounds.tolist() if mesh.bounds is not None else None
        volume = float(abs(mesh.volume))
        watertight = bool(mesh.is_watertight)
    except Exception as e:  # noqa: BLE001
        return _summary(False, error=f"measure failed: {e!r}", tb=traceback.format_exc())

    return _summary(True, volume=volume, bbox=bbox, watertight=watertight,
                    stl="model.stl", step="model.step")


def main() -> int:
    raw = sys.stdin.read()
    try:
        job = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print(json.dumps(_summary(False, error=f"invalid job JSON: {e!r}")))
        return 0
    summary = _run(job)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
