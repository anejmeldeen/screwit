"""Dev storage routes and demo-scans (SPEC 5, 8, AGENTS.md decision 3).

In dev, storage is local disk with pre-signed POST semantics emulated by the API. P5's LocalStorage
provides accept_upload(fields, filename, stream) and verify_signed(key, exp, sig); these routes call
them through getattr and return 501 when the deployed (S3) storage lacks them, since S3 handles upload
and signing itself. Assets are streamed back through the signed GET route.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import Response, StreamingResponse

from screwit.app import signing
from screwit.app.accounts import UserDep, require_asset_owner
from screwit.app.db import BuildOwnerRow, db_session
from screwit.app.deps import RegistryDep
from screwit.app.registry import fixture_asset
from screwit.contracts import DemoScan

router = APIRouter()


@router.get("/demo-scans", response_model=list[DemoScan])
def demo_scans(reg: RegistryDep, user: UserDep) -> list[DemoScan]:
    import json

    path = reg.settings.fixtures_dir / "demo_scans.json"
    if not path.exists():
        return []
    with db_session() as session:
        demos = []
        for data in json.loads(path.read_text()):
            demo = DemoScan.model_validate(data)
            owner = session.get(BuildOwnerRow, demo.scan_id)
            if owner is not None and owner.username == user.username:
                demos.append(signing.demo_scan(reg.storage, demo))
        return demos


@router.post("/storage/upload")
async def storage_upload(request: Request, reg: RegistryDep, user: UserDep) -> Response:
    """Multipart upload: form fields first (the pre-signed POST fields), then the file. Delegated to
    P5's LocalStorage.accept_upload; 501 when storage does not implement it (deployed S3 path)."""
    accept = getattr(reg.storage, "accept_upload", None)
    if accept is None:
        raise HTTPException(status_code=501, detail="local upload not supported by this storage backend")

    form = await request.form()
    fields = {k: str(v) for k, v in form.items() if not hasattr(v, "filename")}
    require_asset_owner(fields.get("key", ""), user)
    upload = next((v for v in form.values() if hasattr(v, "filename")), None)
    if upload is None:
        raise HTTPException(status_code=400, detail="no file part in the multipart body")

    try:
        accept(fields, upload.filename, upload.file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return Response(status_code=204)


@router.get("/assets/{key:path}")
def get_asset(key: str, exp: str, sig: str, reg: RegistryDep):
    """Signed GET for a stored asset. Delegated to P5's LocalStorage.verify_signed; 501 when the
    backend does not implement it (deployed S3 serves signed URLs itself)."""
    verify = getattr(reg.storage, "verify_signed", None)
    if verify is None:
        raise HTTPException(status_code=501, detail="signed GET not supported by this storage backend")
    if not verify(key, exp, sig):
        raise HTTPException(status_code=403, detail="invalid or expired signature")

    head = reg.storage.head(key)
    if head is None:
        # Demo (hero) scans live in the read-only fixtures dir, not the live data dir. In live mode
        # storage reads data-live, so demo assets 404 there; fall back to fixtures for them so the
        # landing-page example previews load regardless of mode.
        fb = fixture_asset(reg, key)
        if fb is not None:
            data, ctype = fb
            return StreamingResponse(iter([data]), media_type=ctype)
        raise HTTPException(status_code=404, detail="asset not found")
    data = reg.storage.get_bytes(key)

    def body():
        yield data

    return StreamingResponse(body(), media_type=head.get("content_type") or "application/octet-stream")
