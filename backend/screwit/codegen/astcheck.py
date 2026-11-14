"""AST pre-check for LLM-written CadQuery programs (SPEC 3.3, X7 brief).

This is a CONVENIENCE FILTER, not a security boundary. The security boundary is the disposable
container (--network none, --read-only, dropped caps, no mounts, limits); see codegen/sandbox.py. This
check rejects the obviously-wrong-shaped program early (an import of something other than cadquery or
math, or dunder attribute access) so we do not pay a container start to watch it fail. A program that
evades this check still cannot reach the network or the host, because the container, not this function,
contains it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# Only these modules may be imported by a codegen program. Everything else (socket, os, subprocess,
# ctypes, importlib, ...) is rejected here as an early, convenience filter.
ALLOWED_IMPORTS: frozenset[str] = frozenset({"cadquery", "cq", "math"})


@dataclass(frozen=True)
class AstCheckResult:
    ok: bool
    reason: str | None = None


def check_source(source: str) -> AstCheckResult:
    """Parse `source` and reject disallowed imports and any dunder attribute access.

    Returns an AstCheckResult; never raises for a hostile program (only for a programming bug). A
    SyntaxError in the source is reported as a non-ok result, not raised.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return AstCheckResult(False, f"source does not parse: {e}")

    for node in ast.walk(tree):
        # import X / import X.Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    return AstCheckResult(False, f"import of {alias.name!r} is not allowed "
                                                 f"(only {sorted(ALLOWED_IMPORTS)})")
        # from X import ...
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level != 0:
                return AstCheckResult(False, "relative imports are not allowed")
            if top not in ALLOWED_IMPORTS:
                return AstCheckResult(False, f"import from {node.module!r} is not allowed "
                                             f"(only {sorted(ALLOWED_IMPORTS)})")
        # __import__(...), getattr/setattr tricks, and any dunder attribute access.
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return AstCheckResult(False, f"dunder attribute access {node.attr!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                return AstCheckResult(False, f"dunder name {node.id!r} is not allowed")

    return AstCheckResult(True)
