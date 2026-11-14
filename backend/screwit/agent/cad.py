"""CAD orchestration helpers for the agent (SPEC 6.3 create_cad, 6.4 templates, 7 P4 #7).

Template parameter schemas are declared here for the agent's tool description (P5's CadService.templates
supplies them at runtime; this is the fallback used in tests and mock mode). The confirm-before-printing
guard lives here as a CODE-LEVEL check: a mate whose source is scan or snapped is refused before any
create_cad request is built, so the model must get a caliper value or user confirmation first.
"""

from __future__ import annotations

from screwit.contracts import Part
from screwit.contracts.models import ParamSource

# JSON-Schema-ish parameter descriptions per template (SPEC 6.4 library). Ranges guide the model.
TEMPLATE_SCHEMAS: dict[str, dict] = {
    "knob": {
        "description": "Round/D-shaft/knurled control knob with a bore that mates a shaft.",
        "params": {
            "shaft_diameter": {"type": "number", "unit": "mm", "min": 3, "max": 20, "mates": True},
            "shaft_flat_depth": {"type": "number", "unit": "mm", "min": 0, "max": 5, "mates": True},
            "outer_diameter": {"type": "number", "unit": "mm", "min": 15, "max": 80},
            "height": {"type": "number", "unit": "mm", "min": 8, "max": 40},
        },
    },
    "l_bracket": {
        "description": "L-bracket with mounting holes.",
        "params": {
            "leg_a_mm": {"type": "number", "unit": "mm", "min": 15, "max": 200},
            "leg_b_mm": {"type": "number", "unit": "mm", "min": 15, "max": 200},
            "thickness_mm": {"type": "number", "unit": "mm", "min": 2, "max": 10},
            "hole_diameter": {"type": "number", "unit": "mm", "min": 2, "max": 12, "mates": True},
        },
    },
    "spacer": {
        "description": "Spacer / washer with a through bore.",
        "params": {
            "bore_diameter": {"type": "number", "unit": "mm", "min": 2, "max": 40, "mates": True},
            "outer_diameter": {"type": "number", "unit": "mm", "min": 4, "max": 80},
            "height": {"type": "number", "unit": "mm", "min": 1, "max": 40},
        },
    },
    "pipe_adapter": {
        "description": "Two-diameter hose or pipe adapter.",
        "params": {
            "diameter_a": {"type": "number", "unit": "mm", "min": 5, "max": 80, "mates": True},
            "diameter_b": {"type": "number", "unit": "mm", "min": 5, "max": 80, "mates": True},
            "length": {"type": "number", "unit": "mm", "min": 10, "max": 120},
        },
    },
    "cap": {
        "description": "Cap / plug for a round opening.",
        "params": {
            "inner_diameter": {"type": "number", "unit": "mm", "min": 3, "max": 80, "mates": True},
            "depth": {"type": "number", "unit": "mm", "min": 3, "max": 40},
        },
    },
}

# Measurement sources that a mating parameter may NOT be built from without asking the user first.
_UNCONFIRMED_SOURCES: frozenset[ParamSource] = frozenset({"scan", "snapped"})


class ConfirmBeforePrinting(Exception):
    """Raised by the create_cad tool when a mating param's source is scan or snapped (SPEC 6.3)."""

    def __init__(self, offenders: list[tuple[str, str, str, ParamSource]]) -> None:
        self.offenders = offenders
        lines = [f"- {pid}.{fid}.{field}: source is {src}" for pid, fid, field, src in offenders]
        super().__init__(
            "Cannot print yet: these mating dimensions come from the scan and the scan cannot resolve "
            "them reliably. Ask the user to measure each with calipers (set_measurement caliper_mm) or "
            "to accept the scan value (set_measurement user_confirmed=true) before calling create_cad:\n"
            + "\n".join(lines)
        )


def check_confirm_before_printing(parts: dict[str, Part], mating_fields: list[tuple[str, str, str]]) -> None:
    """mating_fields is a list of (part_id, feature_id, measurement_field). Raises ConfirmBeforePrinting
    if any resolves to a scan- or snapped-sourced Measurement (SPEC 6.3 confirm-before-printing)."""
    offenders: list[tuple[str, str, str, ParamSource]] = []
    for part_id, feature_id, field in mating_fields:
        part = parts.get(part_id)
        if part is None:
            continue
        feat = part.feature(feature_id)
        if feat is None:
            continue
        meas = getattr(feat, field, None)
        if meas is None:
            continue
        if meas.param_source in _UNCONFIRMED_SOURCES:
            offenders.append((part_id, feature_id, field, meas.param_source))
    if offenders:
        raise ConfirmBeforePrinting(offenders)
