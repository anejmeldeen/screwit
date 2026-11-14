"""The signed-in account's saved builds, including admin's existing examples."""

import json
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import select

from screwit.app.accounts import UserDep
from screwit.app.db import BuildOwnerRow, db_session
from screwit.app.deps import RegistryDep
from screwit.app.store import ScanStore
from screwit.contracts import DemoScan

router = APIRouter()


class BuildSummary(BaseModel):
    scan_id: str
    title: str
    status: str
    created_at: datetime
    is_example: bool


@router.get("/builds", response_model=list[BuildSummary])
def list_builds(reg: RegistryDep, user: UserDep) -> list[BuildSummary]:
    path = reg.settings.fixtures_dir / "demo_scans.json"
    demos = [DemoScan.model_validate(item) for item in json.loads(path.read_text())] if path.exists() else []
    titles = {demo.scan_id: demo.title for demo in demos}
    with db_session() as session:
        owned = session.exec(select(BuildOwnerRow).where(BuildOwnerRow.username == user.username)).all()
    store = ScanStore(reg.data)
    builds = []
    for owner in owned:
        scan = store.get_scan(owner.scan_id)
        if scan is None:
            continue
        default_title = f"{scan.input.type.capitalize()} build · {scan.scan_id.removeprefix('scn_')[:6]}"
        builds.append(BuildSummary(
            scan_id=scan.scan_id,
            title=titles.get(scan.scan_id, default_title),
            status=scan.status, created_at=scan.created_at, is_example=scan.scan_id in titles,
        ))
    return sorted(builds, key=lambda build: build.created_at, reverse=True)
