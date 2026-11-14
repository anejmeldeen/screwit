"""Template registry (SPEC 6.4)."""

from __future__ import annotations

from screwit.cad.templates.base import BuildResult, MateInfo, ParamSpec, Template
from screwit.cad.templates.parts import (
    AdapterTemplate,
    CableClipTemplate,
    CapTemplate,
    EnclosureTemplate,
    HookTemplate,
    KnobTemplate,
    LBracketTemplate,
    SpacerTemplate,
)

_TEMPLATE_CLASSES: tuple[type[Template], ...] = (
    KnobTemplate,
    LBracketTemplate,
    SpacerTemplate,
    AdapterTemplate,
    CableClipTemplate,
    HookTemplate,
    CapTemplate,
    EnclosureTemplate,
)

TEMPLATES: dict[str, Template] = {cls.name: cls() for cls in _TEMPLATE_CLASSES}


def get_template(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise ValueError(f"unknown template {name!r}; known: {sorted(TEMPLATES)}") from None


__all__ = ["BuildResult", "MateInfo", "ParamSpec", "Template", "TEMPLATES", "get_template"]
