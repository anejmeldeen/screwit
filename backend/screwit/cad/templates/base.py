"""Template framework (SPEC 6.4).

A template is a parametric CadQuery part with a typed parameter schema (ranges + defaults) and a
`build(params) -> cq.Workplane` that returns a solid in scan-frame millimetres. Each template documents
the local axis of its mating feature so the CAD service can place it and run mate checks.

Mating-feature convention shared by every template whose part mates with a shaft:
the bore is centred on the local origin and its axis runs along +Z, entering the part from the -Z face
(the face that sits against the panel). A D-bore's flat sits on the +Y side of the bore, so its inward
flat normal (pointing from the flat surface toward the bore axis) is -Y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cadquery as cq
from pydantic import BaseModel, ConfigDict


class ParamSchema(BaseModel):
    """Base class for a template's parameter model. extra='forbid' so an unknown param fails loudly."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ParamSpec:
    """One parameter's range and default, for validation and the agent's JSON Schema."""

    name: str
    default: float
    min: float
    max: float
    description: str = ""


@dataclass(frozen=True)
class MateInfo:
    """How a built part mates with an original feature, in the part's local frame.

    `kind` is the shaft feature the bore accepts: 'cylinder' accepts a plain shaft or a d_shaft;
    'd_shaft' requires the mating feature to be a d_shaft (a round bore would spin on a D-shaft, but a
    D-bore on a round shaft is over-constrained, so a d_shaft bore only mates a d_shaft).

    Geometry is expressed in the template's local frame:
    - `bore_axis_point` / `bore_axis_dir`: a point on the bore axis and its unit direction.
    - `bore_radius_mm`: the modelled bore radius (shaft radius + radial clearance).
    - `engagement_mm`: how deep the shaft can sit in the bore (the bore's usable depth).
    - `flat_normal`: for a D-bore, the unit inward normal of the bore's flat; None for a round bore.
    - `flat_clearance_mm`: for a D-bore, the modelled gap between the bore flat and the shaft flat.
    """

    kind: str  # "cylinder" | "d_shaft"
    bore_axis_point: tuple[float, float, float]
    bore_axis_dir: tuple[float, float, float]
    bore_radius_mm: float
    engagement_mm: float
    flat_normal: tuple[float, float, float] | None = None
    flat_clearance_mm: float | None = None


@dataclass
class BuildResult:
    """A built template: the solid plus optional mate metadata and a structural-load flag."""

    solid: cq.Workplane
    mate: MateInfo | None = None
    carries_load: bool = False  # strength_estimate only computed for load-bearing parts (brackets, hooks)
    load_feature: str = ""      # human-readable description for the strength note


class Template:
    """One parametric part. Subclasses set `name`, `params` and implement `build`."""

    name: str = ""
    params: tuple[ParamSpec, ...] = ()
    # which mate `fit`/clearance applies; the bore radius is shaft_radius + radial_clearance_mm.
    mates: bool = False
    _schema: dict[str, ParamSpec] = field(default_factory=dict)

    def __init__(self) -> None:
        self._schema = {p.name: p for p in self.params}

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}

    def validate_params(self, params: dict[str, float]) -> dict[str, float]:
        """Fill defaults, reject unknown params and out-of-range values. Returns the resolved params."""
        unknown = set(params) - set(self._schema)
        if unknown:
            raise ValueError(f"template {self.name!r} has no parameter(s) {sorted(unknown)}")
        resolved = self.defaults()
        for k, v in params.items():
            fv = float(v)
            spec = self._schema[k]
            if not (spec.min <= fv <= spec.max):
                raise ValueError(
                    f"{self.name}.{k}={fv} out of range [{spec.min}, {spec.max}]"
                )
            resolved[k] = fv
        return resolved

    def json_schema(self) -> dict:
        """JSON Schema of the params, for the agent's create_cad tool description."""
        props = {
            p.name: {
                "type": "number",
                "default": p.default,
                "minimum": p.min,
                "maximum": p.max,
                "description": p.description,
            }
            for p in self.params
        }
        return {
            "type": "object",
            "title": self.name,
            "properties": props,
            "required": [],
            "additionalProperties": False,
        }

    def build(self, params: dict[str, float], *, mate_radial_clearance_mm: float | None = None,
              profile=None) -> BuildResult:
        raise NotImplementedError
