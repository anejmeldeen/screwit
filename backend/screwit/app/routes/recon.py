"""Reconstruction and scale routes (SPEC 5; handlers owned by P2, mounted in P4's shell).

GET /reconstruction returns the 4.2 object with its asset URIs replaced by short-lived signed URLs.
POST /scale is an analytic uniform rescale: it bumps reconstruction_revision and every part's
measurement_revision, stales every CAD result for the scan, and rejects a stale expected_revision (409).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from screwit.app import signing
from screwit.app.deps import RegistryDep
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import Reconstruction, ScaleRequest
from screwit.interfaces import StaleRevision

router = APIRouter()


def _store(reg: Registry) -> ScanStore:
    return ScanStore(reg.data)


def _sign(reg: Registry, recon: Reconstruction) -> Reconstruction:
    return signing.reconstruction(reg.storage, recon)


@router.get("/scans/{scan_id}/reconstruction", response_model=Reconstruction)
def get_reconstruction(scan_id: str, reg: RegistryDep) -> Reconstruction:
    recon = _store(reg).get_reconstruction(scan_id)
    if recon is None:
        raise HTTPException(status_code=404, detail="reconstruction not found")
    return _sign(reg, recon)


@router.post("/scans/{scan_id}/scale", response_model=Reconstruction)
def scale(scan_id: str, req: ScaleRequest, reg: RegistryDep) -> Reconstruction:
    store = _store(reg)
    scan = store.get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    if req.expected_revision != scan.reconstruction_revision:
        raise HTTPException(status_code=409, detail="stale expected_revision")

    try:
        recon = reg.recon.rescale(scan_id, req, reg.storage, reg.perception)
    except StaleRevision:
        raise HTTPException(status_code=409, detail="stale expected_revision") from None

    # Scaling stales every CAD result for the scan (SPEC 5 /scale).
    store.mark_scan_cad_stale(scan_id)
    return _sign(reg, recon)
