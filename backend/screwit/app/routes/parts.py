"""Parts, scene graph and measure routes (SPEC 5; handlers owned by P3, mounted in P4's shell).

PATCH renames/relabels (no revision change) or edits a Measurement (bumps measurement_revision and
stales CAD referencing the part). Merge retires the sources and creates a new part. Both targeted
staleness paths touch only CAD results that pinned the affected parts.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from screwit.app import signing
from screwit.app.deps import RegistryDep
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import (
    MeasureRequest,
    MeasureResponse,
    MergeRequest,
    Part,
    PartPatch,
)

router = APIRouter()


def _store(reg: Registry) -> ScanStore:
    return ScanStore(reg.data)


@router.get("/scans/{scan_id}/parts", response_model=list[Part])
def list_parts(scan_id: str, reg: RegistryDep) -> list[Part]:
    return signing.parts(reg.storage, _store(reg).list_parts(scan_id))


@router.patch("/scans/{scan_id}/parts/{part_id}", response_model=Part)
def patch_part(scan_id: str, part_id: str, patch: PartPatch,
               reg: RegistryDep) -> Part:
    store = _store(reg)
    before = store.get_part(scan_id, part_id)
    if before is None:
        raise HTTPException(status_code=404, detail="part not found")
    try:
        updated = reg.perception.patch_part(scan_id, part_id, patch, reg.storage)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    store.save_part(scan_id, updated)

    # A measurement change (not a pure rename) stales CAD results that referenced this part (SPEC 5).
    if updated.measurement_revision != before.measurement_revision:
        store.mark_cad_stale(scan_id, store.cad_jobs_referencing(scan_id, {part_id}))
    return signing.part(reg.storage, updated)


@router.post("/scans/{scan_id}/parts/merge", response_model=Part)
def merge_parts(scan_id: str, req: MergeRequest, reg: RegistryDep) -> Part:
    store = _store(reg)
    before_ids = {p.part_id for p in store.list_parts(scan_id)}
    missing = [pid for pid in req.part_ids if pid not in before_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown part(s): {missing}")

    merged = reg.perception.merge_parts(scan_id, req, reg.storage)

    # Retire the sources, add the merged part, stale CAD that referenced any retired part (SPEC 5).
    kept = [p for p in store.list_parts(scan_id) if p.part_id not in set(req.part_ids)]
    kept.append(merged)
    store.save_parts(scan_id, kept)

    retired = _retired_ids(reg, req, before_ids, kept)
    store.mark_cad_stale(scan_id, store.cad_jobs_referencing(scan_id, retired))
    return signing.part(reg.storage, merged)


def _retired_ids(reg: RegistryDep, req: MergeRequest, before_ids: set[str],
                 kept: list[Part]) -> set[str]:
    """Part ids the merge removed. Prefer P3's retired_ids(); fall back to diffing the parts list so a
    merge always stales CAD for every retired id even if the service does not report them."""
    reporter = getattr(reg.perception, "retired_ids", None)
    if callable(reporter):
        try:
            reported = set(reporter())
            if reported:
                return reported
        except Exception:
            pass
    after_ids = {p.part_id for p in kept}
    diffed = before_ids - after_ids
    return diffed or set(req.part_ids)


@router.get("/scans/{scan_id}/scene-graph")
def scene_graph(scan_id: str, reg: RegistryDep):
    graph = _store(reg).get_scene_graph(scan_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="scene graph not found")
    return graph


@router.post("/scans/{scan_id}/measure", response_model=MeasureResponse)
def measure(scan_id: str, req: MeasureRequest, reg: RegistryDep) -> MeasureResponse:
    try:
        return reg.perception.measure(scan_id, req, reg.storage)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"unknown part {e}") from None
