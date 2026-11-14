"""CAD service: parametric template library, fit and printability checks, exports (SPEC 6.4).

Everything is built in scan-frame millimetres (SPEC 3.1). The public entry points are the template
registry (`screwit.cad.templates`) and `CadRunner` (`screwit.cad.service`), which together implement
the CadService protocol in screwit.interfaces.
"""

from __future__ import annotations

from screwit.cad.service import CadRunner
from screwit.cad.templates import TEMPLATES, get_template

__all__ = ["CadRunner", "TEMPLATES", "get_template"]
