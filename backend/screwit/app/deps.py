"""FastAPI dependencies shared by the route modules."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from screwit.app.registry import Registry


def get_registry(request: Request) -> Registry:
    return request.app.state.registry


# Annotated dependency form (avoids a Depends() call in a default argument).
RegistryDep = Annotated[Registry, Depends(get_registry)]
