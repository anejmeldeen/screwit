"""CAD routes (SPEC 5; handlers owned by P5, mounted in P4's shell).

POST validates the request's pinned revisions against the current parts and rejects a mismatch with
409, then runs CadService on the CPU queue and stores the result. GET recomputes `stale` against the
current part revisions on every read, so a part edited after the job stales its result.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from screwit.app import signing
from screwit.app.deps import RegistryDep
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import CadRequest, CadResult, Part

router = APIRouter()


def _store(reg: Registry) -> ScanStore:
    return ScanStore(reg.data)


def _revisions_match(req: CadRequest, parts: dict[str, Part], recon_revision: int) -> bool:
    if req.reconstruction_revision != recon_revision:
        return False
    for part_id, pinned in req.part_measurement_revisions.items():
        part = parts.get(part_id)
        if part is None or part.measurement_revision != pinned:
            return False
    return True


def _is_stale(req: CadRequest, parts: dict[str, Part], recon_revision: int) -> bool:
    return not _revisions_match(req, parts, recon_revision)


@router.post("/scans/{scan_id}/cad", response_model=CadResult)
def create_cad(scan_id: str, req: CadRequest, reg: RegistryDep) -> CadResult:
    if req.scan_id != scan_id:
        raise HTTPException(status_code=400, detail="scan_id in body does not match the path")
    store = _store(reg)
    scan = store.get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")

    parts = {p.part_id: p for p in store.list_parts(scan_id)}
    if not _revisions_match(req, parts, scan.reconstruction_revision):
        raise HTTPException(status_code=409, detail="request revisions do not match the current part")

    result = _run_cad(reg, req, parts)
    store.save_cad_job(req, result)
    return signing.cad_result(reg.storage, result)


def _run_cad(reg: Registry, req: CadRequest, parts: dict[str, Part]) -> CadResult:
    out: dict = {}

    def job() -> None:
        out["result"] = reg.cad.run(req, parts, reg.storage)

    reg.queue.enqueue("cpu", job)
    return out["result"]


@router.get("/scans/{scan_id}/cad/{cad_id}", response_model=CadResult)
def get_cad(scan_id: str, cad_id: str, reg: RegistryDep) -> CadResult:
    store = _store(reg)
    job = store.get_cad_job(scan_id, cad_id)
    if job is None:
        raise HTTPException(status_code=404, detail="cad result not found")
    req, result = job

    scan = store.get_scan(scan_id)
    parts = {p.part_id: p for p in store.list_parts(scan_id)}
    recon_revision = scan.reconstruction_revision if scan else req.reconstruction_revision
    return signing.cad_result(reg.storage, result.model_copy(update={"stale": _is_stale(req, parts, recon_revision)}))
