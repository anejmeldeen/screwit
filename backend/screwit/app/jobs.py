"""The scan processing job: validate the upload, run the pipeline, publish its results (SPEC 3, 5).

One top-level function taking only a scan id, so it can run inline (dev, tests) or on an RQ GPU
worker (`python -m screwit.app.worker`) that rebuilds the registry from the environment.

Every stage transition is written to the database as it happens, which is what the progress SSE
stream reads; nothing is kept only in memory. Any failure ends the scan in `failed` with a message a
person can act on, never a bare 500.
"""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path

from pydantic import TypeAdapter

from screwit.contracts import Part, Reconstruction, Scan, SceneGraph

log = logging.getLogger("screwit.jobs")

_PARTS = TypeAdapter(list[Part])


class _Progress:
    """Persists stage updates for one scan. Stages only move forward; the scan row is the record."""

    def __init__(self, store, scan_id: str) -> None:
        self.store = store
        self.scan_id = scan_id

    def _scan(self) -> Scan:
        scan = self.store.get_scan(self.scan_id)
        assert scan is not None
        return scan

    def set(self, name, state, progress=None, message=None) -> None:
        scan = self._scan()
        stages = [s.model_copy(update={"state": state, "progress": progress, "message": message})
                  if s.name == name else s for s in scan.stages]
        self.store.save_scan(scan.model_copy(update={"stages": stages}))

    def status(self, status: str, *, fail_stage: str | None = None, message: str | None = None) -> None:
        scan = self._scan()
        stages = scan.stages
        if fail_stage is not None:
            stages = [s.model_copy(update={"state": "failed", "message": message}) if s.name == fail_stage
                      else s for s in stages]
        self.store.save_scan(scan.model_copy(update={"status": status, "stages": stages}))

    def first_open_stage(self) -> str:
        for s in self._scan().stages:
            if s.state in ("pending", "running"):
                return s.name
        return "parts"


def _publish(reg, store, scan: Scan) -> None:
    """Copy what the pipeline wrote to storage into the database the API serves from."""
    key = f"scans/{scan.scan_id}"
    recon = Reconstruction.model_validate_json(reg.storage.get_bytes(f"{key}/reconstruction.json"))
    store.save_reconstruction(recon)
    parts = _PARTS.validate_json(reg.storage.get_bytes(f"{key}/parts.json"))
    store.save_parts(scan.scan_id, parts)
    graph = SceneGraph.model_validate_json(reg.storage.get_bytes(f"{key}/scene_graph.json"))
    store.save_scene_graph(graph, recon.reconstruction_revision)


def process_scan(scan_id: str, registry=None) -> None:
    from screwit.app.registry import build_registry
    from screwit.app.settings import settings
    from screwit.app.store import ScanStore

    reg = registry or build_registry(settings())
    store = ScanStore(reg.data)
    progress = _Progress(store, scan_id)
    scan = store.get_scan(scan_id)
    if scan is None:
        log.error("process_scan: unknown scan %s", scan_id)
        return

    # 1. Contained validation of the untrusted upload (SPEC 3.3).
    progress.status("validating")
    try:
        photo_keys = []
        if scan.input.type == "photos":
            from screwit.app.db import PhotoUploadRow, db_session

            with db_session() as session:
                manifest = session.get(PhotoUploadRow, scan_id)
                if manifest:
                    photo_keys = json.loads(manifest.keys_json)
        info = None
        for key in photo_keys or [scan.input.uri]:
            photo_scan = scan.model_copy(update={"input": scan.input.model_copy(update={"uri": key})})
            checked = reg.media_validator.validate(photo_scan, reg.storage)
            info = info or checked
        assert info is not None
    except ValueError as e:  # the validator's user-facing rejection
        first = "frames" if scan.input.type != "mesh" else "mesh"
        progress.status("failed", fail_stage=first, message=f"Upload rejected: {e}")
        return
    except Exception as e:  # noqa: BLE001 - any validator crash must end the scan cleanly
        log.exception("validation crashed for %s", scan_id)
        progress.status("failed", fail_stage="frames", message=f"Could not check the upload ({type(e).__name__})")
        return
    upd = {k: v for k, v in (("duration_s", info.duration_s), ("container", info.container),
                             ("codec", info.codec), ("hdr", info.hdr)) if v is not None}
    scan = store.get_scan(scan_id)
    store.save_scan(scan.model_copy(update={"status": "processing",
                                            "input": scan.input.model_copy(update=upd)}))

    # 2. The pipeline reports every stage transition through `progress`.
    try:
        storage = (_PhotoBatchStorage(reg.storage, scan.input.uri, photo_keys)
                   if photo_keys else reg.storage)
        reg.recon.run(store.get_scan(scan_id), storage, reg.perception, progress.set)
        _publish(reg, store, store.get_scan(scan_id))
    except Exception as e:  # noqa: BLE001
        log.error("pipeline failed for %s:\n%s", scan_id, traceback.format_exc())
        stage = progress.first_open_stage()
        progress.status("failed", fail_stage=stage, message=f"{stage} failed: {str(e)[:200]}")
        return
    progress.status("ready")


class _PhotoBatchStorage:
    """Expose all validated photos as one local directory to the existing pipeline."""

    def __init__(self, storage, input_key: str, keys: list[str]) -> None:
        self._storage, self._input_key, self._keys = storage, input_key, keys

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def local_path(self, key: str, scratch: Path) -> Path:
        if key != self._input_key:
            return self._storage.local_path(key, scratch)
        directory = Path(scratch) / "photo-inputs"
        directory.mkdir(parents=True, exist_ok=True)
        for i, photo_key in enumerate(self._keys):
            source = self._storage.local_path(photo_key, scratch)
            shutil.copyfile(source, directory / f"{i:02d}{Path(photo_key).suffix}")
        return directory
