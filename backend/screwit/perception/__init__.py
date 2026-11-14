"""Perception: parts, measurement and scene graph (SPEC sections 4.3, 4.4, 6.2; P3).

Public surface:
- PerceptionImpl: the PerceptionService implementation the API and GPU worker call.
- snap_diameter: the role-specific standard-size snapping of SPEC 6.2 step 4.
"""

from __future__ import annotations

from screwit.perception.service import PerceptionImpl
from screwit.perception.snapping import SnapResult, snap_diameter

__all__ = ["PerceptionImpl", "SnapResult", "snap_diameter"]
