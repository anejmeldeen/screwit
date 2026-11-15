"""CAD services, loaded on demand so mesh-only STL exports need no CAD kernel."""

from __future__ import annotations

__all__ = ["CadRunner", "TEMPLATES", "get_template"]


def __getattr__(name: str):
    if name == "CadRunner":
        from screwit.cad.service import CadRunner

        return CadRunner
    if name in {"TEMPLATES", "get_template"}:
        from screwit.cad.templates import TEMPLATES, get_template

        return {"TEMPLATES": TEMPLATES, "get_template": get_template}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
