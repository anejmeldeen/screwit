"""The signed-in account's saved builds, including admin's existing examples."""

import io
import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, StringConstraints
from sqlmodel import select

from screwit.app.accounts import UserDep
from screwit.app.db import BuildActivityRow, BuildOpenRow, BuildOwnerRow, BuildTitleRow, DeletedBuildRow, db_session
from screwit.app.deps import RegistryDep
from screwit.app.store import ScanStore
from screwit.contracts import DemoScan

router = APIRouter()


class BuildSummary(BaseModel):
    scan_id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    opened_at: datetime | None
    input_type: str
    is_example: bool


@router.get("/builds", response_model=list[BuildSummary])
def list_builds(reg: RegistryDep, user: UserDep) -> list[BuildSummary]:
    path = reg.settings.fixtures_dir / "demo_scans.json"
    demos = [DemoScan.model_validate(item) for item in json.loads(path.read_text())] if path.exists() else []
    titles = {demo.scan_id: demo.title for demo in demos}
    with db_session() as session:
        owned = session.exec(
            select(BuildOwnerRow, BuildOpenRow.opened_at)
            .outerjoin(BuildOpenRow, BuildOwnerRow.scan_id == BuildOpenRow.scan_id)
            .where(BuildOwnerRow.username == user.username)
        ).all()
        saved_titles = {
            row.scan_id: row.title for row in session.exec(
                select(BuildTitleRow).join(BuildOwnerRow, BuildTitleRow.scan_id == BuildOwnerRow.scan_id)
                .where(BuildOwnerRow.username == user.username)
            ).all()
        }
        activity = {
            row.scan_id: row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at
            for row in session.exec(
                select(BuildActivityRow).join(BuildOwnerRow, BuildActivityRow.scan_id == BuildOwnerRow.scan_id)
                .where(BuildOwnerRow.username == user.username)
            ).all()
        }
    store = ScanStore(reg.data)
    builds = []
    for owner, opened_at in owned:
        scan = store.get_scan(owner.scan_id)
        if scan is None:
            continue
        default_title = f"{scan.input.type.capitalize()} build"
        builds.append(BuildSummary(
            scan_id=scan.scan_id,
            title=saved_titles.get(scan.scan_id, titles.get(scan.scan_id, default_title)),
            opened_at=opened_at.replace(tzinfo=UTC) if opened_at and opened_at.tzinfo is None else opened_at,
            status=scan.status, created_at=scan.created_at, is_example=scan.scan_id in titles,
            updated_at=activity.get(scan.scan_id, scan.created_at), input_type=scan.input.type,
        ))
    return sorted(
        builds, key=lambda build: (build.opened_at is not None, build.opened_at or build.created_at), reverse=True
    )


@router.post("/builds/{scan_id}/open", status_code=204)
def open_build(scan_id: str) -> Response:
    with db_session() as session:
        session.merge(BuildOpenRow(scan_id=scan_id, opened_at=datetime.now(UTC)))
        session.commit()
    return Response(status_code=204)


class RenameBuildRequest(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


@router.patch("/builds/{scan_id}", status_code=204)
def rename_build(scan_id: str, body: RenameBuildRequest, user: UserDep) -> Response:
    with db_session() as session:
        owner = session.get(BuildOwnerRow, scan_id)
        if owner is None or owner.username != user.username:
            raise HTTPException(404, "Build not found.")
        session.merge(BuildTitleRow(scan_id=scan_id, title=body.title))
        session.merge(BuildActivityRow(scan_id=scan_id, updated_at=datetime.now(UTC)))
        session.commit()
    return Response(status_code=204)


@router.delete("/builds/{scan_id}", status_code=204)
def delete_build(scan_id: str, user: UserDep) -> Response:
    with db_session() as session:
        owner = session.get(BuildOwnerRow, scan_id)
        if owner is None or owner.username != user.username:
            raise HTTPException(404, "Build not found.")
        session.delete(owner)
        session.add(DeletedBuildRow(scan_id=scan_id))
        session.commit()
    return Response(status_code=204)


@router.get("/builds/{scan_id}/preview.png")
def build_preview(scan_id: str, reg: RegistryDep) -> Response:
    recon = ScanStore(reg.data).get_reconstruction(scan_id)
    if recon is None:
        raise HTTPException(404, "Model preview is not available yet.")
    import hashlib

    import numpy as np
    import trimesh

    from screwit.app.preview_renderer import render_mesh_png
    from screwit.app.settings import _REPO_ROOT

    parts = ScanStore(reg.data).list_parts(scan_id)
    signature = hashlib.sha256("".join(p.model_dump_json() for p in parts).encode()).hexdigest()[:16]
    key = f"scans/{scan_id}/renders/workspace-palette-v2-{recon.reconstruction_revision}-{signature}.png"
    if reg.storage.head(key):
        png = reg.storage.get_bytes(key)
    else:
        palette = json.loads((_REPO_ROOT / "web/lib/partPalette.json").read_text())
        meshes = []
        for i, part in enumerate(parts):
            raw = reg.storage.get_bytes(part.mesh_uri)
            mesh = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene").to_geometry()
            color_id = part.color_source_id or part.part_id
            slot = int(color_id[1:]) - 1 if color_id.startswith("p") and color_id[1:].isdigit() else i
            color = [*bytes.fromhex(palette[slot % len(palette)][1:]), 255]
            mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            meshes.append(mesh)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raw = reg.storage.get_bytes(recon.mesh_uri)
            mesh = trimesh.load(io.BytesIO(raw), file_type="glb", force="scene").to_geometry()
            mesh.visual.vertex_colors = [*bytes.fromhex(palette[0][1:]), 255]
        if not len(mesh.faces):
            raise HTTPException(404, "Model has no visible faces.")
        png = render_mesh_png(mesh.vertices, mesh.faces, mesh.visual.vertex_colors,
                              angle_deg=-45, width=960, height=576)
        if reg.settings.mode != "mock":
            reg.storage.put_bytes(key, png, "image/png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, no-cache"})
