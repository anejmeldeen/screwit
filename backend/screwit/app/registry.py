"""Service registry: the one place the API binds the seams (SPEC 7, interfaces.py).

Mock mode wires fixture-backed fakes (screwit.app.fakes). Live mode wires the real owners' classes;
until those land, live mode raises a clear error when a seam is touched, so nothing silently uses a
fake in production. The registry is attached to the FastAPI app state and read through a dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from screwit.app.fakes import (
    FakeCadService,
    FakeMediaValidator,
    FakePerceptionService,
    FakeReconPipeline,
    FixtureData,
    FixtureStorage,
)
from screwit.app.queue import Queue, make_queue
from screwit.app.settings import _REPO_ROOT, Settings
from screwit.interfaces import (
    CadService,
    MediaValidator,
    PerceptionService,
    ReconPipeline,
    Storage,
)


class _LiveNotWired:
    """Placeholder for a seam whose real owner has not landed yet (live mode only)."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item):
        raise RuntimeError(
            f"{self._name} is not wired in live mode yet. Run with SCREWIT_MODE=mock, "
            f"or provide the real {self._name} implementation in screwit.app.registry."
        )


@dataclass
class Registry:
    settings: Settings
    storage: Storage
    media_validator: MediaValidator
    recon: ReconPipeline
    perception: PerceptionService
    cad: CadService
    queue: Queue
    data: FixtureData  # fixture loader; mock routes and the agent read it directly


def build_registry(settings: Settings) -> Registry:
    data = FixtureData(settings.fixtures_dir)
    # Live mode must not run a minutes-long pipeline inside the request: "inline" becomes a background
    # thread there. Mock mode and tests keep the synchronous inline queue.
    backend = "thread" if (not settings.is_mock and settings.queue == "inline") else settings.queue
    queue = make_queue(backend, settings.redis_url)

    if settings.is_mock:
        return Registry(
            settings=settings,
            storage=FixtureStorage(settings.fixtures_dir),
            media_validator=FakeMediaValidator(),
            recon=FakeReconPipeline(data),
            perception=FakePerceptionService(data),
            cad=FakeCadService(data),
            queue=queue,
            data=data,
        )

    # Live mode: the real implementations. A component whose module is not installed yet stays a
    # _LiveNotWired placeholder, which fails loudly on first use instead of silently faking.
    return Registry(
        settings=settings,
        storage=_live_storage(settings),
        media_validator=_live("screwit.storage.validation", "MediaValidator", "MediaValidator (P5)"),
        recon=_live("screwit.recon.pipeline", "ReconPipelineImpl", "ReconPipeline (P2)"),
        perception=_live("screwit.perception", "PerceptionImpl", "PerceptionService (P3)",
                         provider=os.environ.get("SCREWIT_LLM", "fake")),
        cad=_live("screwit.cad", "CadRunner", "CadService (P5)", profiles_dir=_REPO_ROOT / "profiles"),
        queue=queue,
        data=data,
    )


def _live(module: str, cls: str, name: str, **kwargs):
    import importlib

    try:
        return getattr(importlib.import_module(module), cls)(**kwargs)
    except (ImportError, AttributeError):
        return _LiveNotWired(name)


def _live_storage(settings: Settings):
    """Private S3 when SCREWIT_S3_BUCKET is set (deployed), local disk under the data dir otherwise."""
    bucket = os.environ.get("SCREWIT_S3_BUCKET")
    if bucket:
        from screwit.storage import S3Storage

        return S3Storage(bucket, region=os.environ.get("AWS_REGION"))
    from screwit.storage import LocalStorage

    return LocalStorage(settings.data_dir)
