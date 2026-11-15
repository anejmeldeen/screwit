"""Scan lifecycle routes: create, start, get, progress SSE (SPEC 5)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator

from screwit.app.accounts import UserDep
from screwit.app.db import BuildTitleRow, PhotoUploadRow, db_session
from screwit.app.deps import RegistryDep
from screwit.app.registry import Registry
from screwit.app.settings import MAX_UPLOAD_BYTES
from screwit.app.sse import sse_response_with_keepalive
from screwit.app.store import ScanStore, utcnow
from screwit.contracts import (
    CreateScanRequest,
    CreateScanResponse,
    Scan,
    ScanInput,
    Stage,
)
from screwit.contracts.models import MESH_INPUT_SKIPPED, STAGE_ORDER
from screwit.contracts.sse import encode_progress


class CreateWorkspaceRequest(CreateScanRequest):
    title: str | None = Field(default=None, max_length=120)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Enter a workspace title")
        return value


router = APIRouter()

# Content types accepted per input kind (the server re-validates on the worker; SPEC 3.3).
_CONTENT_TYPES: dict[str, list[str]] = {
    "video": ["video/mp4", "video/quicktime", "video/webm"],
    "photos": ["image/jpeg", "image/png", "image/heic"],
    "mesh": ["model/gltf-binary", "application/octet-stream"],
}


def _store(reg: Registry) -> ScanStore:
    return ScanStore(reg.data)


def _new_scan(scan_id: str, req: CreateScanRequest) -> Scan:
    stages = [
        Stage(name=name,
              state="skipped" if (req.input_type == "mesh" and name in MESH_INPUT_SKIPPED) else "pending")
        for name in STAGE_ORDER
    ]
    ext = {"video": "mov", "photos": "jpg", "mesh": "glb"}[req.input_type]
    uri = (f"scans/{scan_id}/input/photo_00.jpg" if req.input_type == "photos"
           else f"scans/{scan_id}/input/upload.{ext}")
    scan_input = ScanInput(type=req.input_type, uri=uri, mesh_units=req.mesh_units)
    return Scan(scan_id=scan_id, created_at=utcnow(), input=scan_input,
                status="uploaded", reconstruction_revision=0, stages=stages)


@router.post("/scans", response_model=CreateScanResponse)
def create_scan(req: CreateWorkspaceRequest, reg: RegistryDep, user: UserDep) -> CreateScanResponse:
    scan_id = "scn_" + uuid.uuid4().hex[:12]
    scan = _new_scan(scan_id, req)
    _store(reg).save_scan(scan, owner=user.username)
    if req.title is not None:
        with db_session() as session:
            session.add(BuildTitleRow(scan_id=scan_id, title=req.title))
            session.commit()

    count = req.photo_count or 1 if req.input_type == "photos" else 1
    types = _CONTENT_TYPES[req.input_type]
    uploads = []
    for i in range(count):
        key = scan.input.uri if count == 1 else f"scans/{scan_id}/input/photo_{i:02d}.jpg"
        uploads.append(reg.storage.presign_upload(key, types, MAX_UPLOAD_BYTES))
    if req.input_type == "photos":
        with db_session() as session:
            session.add(PhotoUploadRow(scan_id=scan_id, keys_json=json.dumps([u.key for u in uploads])))
            session.commit()
    return CreateScanResponse(scan_id=scan_id, uploads=uploads)


@router.post("/scans/{scan_id}/start", response_model=Scan)
def start_scan(scan_id: str, reg: RegistryDep) -> Scan:
    store = _store(reg)
    scan = store.get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    # Check every upload, not just the first photo, before starting work.
    keys = [scan.input.uri]
    if scan.input.type == "photos":
        with db_session() as session:
            manifest = session.get(PhotoUploadRow, scan_id)
            if manifest:
                keys = json.loads(manifest.keys_json)
    total_bytes = 0
    for key in keys:
        head = reg.storage.head(key)
        if head is None:
            raise HTTPException(status_code=400, detail="no upload found for this scan; upload every selected file")
        total_bytes += head["size"]
    if total_bytes > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="upload exceeds the 500 MB limit")

    if not reg.settings.is_mock:
        # Live: one job validates the upload in containment, runs the pipeline and publishes the
        # results, persisting every stage change for the progress stream (screwit.app.jobs).
        from screwit.app.jobs import process_scan

        scan = scan.model_copy(update={"status": "validating"})
        store.save_scan(scan)
        if reg.settings.queue == "rq":
            reg.queue.enqueue("gpu", process_scan, scan_id)
        else:
            reg.queue.enqueue("gpu", process_scan, scan_id, registry=reg)
        return scan

    # Mock: the inline queue runs the fakes, which report the fixture's own stage states.
    scan = scan.model_copy(update={"status": "validating"})
    store.save_scan(scan)
    reg.queue.enqueue("cpu", reg.media_validator.validate, scan, reg.storage)

    scan = scan.model_copy(update={"status": "processing"})
    store.save_scan(scan)

    def _progress(name, state, progress, message) -> None:
        pass

    reg.queue.enqueue("gpu", reg.recon.run, scan, reg.storage, reg.perception, _progress)
    return store.get_scan(scan_id) or scan


@router.get("/scans/{scan_id}", response_model=Scan)
def get_scan(scan_id: str, reg: RegistryDep) -> Scan:
    scan = _store(reg).get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return scan


@router.get("/scans/{scan_id}/events")
def scan_events(scan_id: str, reg: RegistryDep):
    """SSE stream of stage progress: one `scan` event per change, KEEPALIVE every 20 s, close on a
    terminal status (SPEC 5). In mock mode the scan is already terminal, so it emits one frame."""
    scan = _store(reg).get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    store = _store(reg)

    def frames() -> Iterator[str]:
        # One frame now, then one per change, until the scan is ready or failed. The source is the
        # database row the processing job updates, so this works across processes and restarts.
        import time

        last = scan.model_dump_json()
        yield encode_progress(scan)
        current = scan
        while current.status not in ("ready", "failed"):
            time.sleep(0.5)
            current = store.get_scan(scan_id) or current
            now = current.model_dump_json()
            if now != last:
                last = now
                yield encode_progress(current)

    return sse_response_with_keepalive(frames())
