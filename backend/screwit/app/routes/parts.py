"""Parts, scene graph and measure routes (SPEC 5; handlers owned by P3, mounted in P4's shell).

PATCH renames/relabels (no revision change) or edits a Measurement (bumps measurement_revision and
stales CAD referencing the part). Merge retires the sources and creates a new part. Both targeted
staleness paths touch only CAD results that pinned the affected parts.
"""

from __future__ import annotations

import io
import json
import uuid
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import select
from starlette.responses import Response

from screwit.app import signing
from screwit.app.db import MergeHistoryRow, db_session
from screwit.app.deps import RegistryDep
from screwit.app.fakes import _apply_patch
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import (
    MeasureRequest,
    MeasureResponse,
    MergeRequest,
    Part,
    PartPatch,
    SceneGraph,
)

router = APIRouter()


def _store(reg: Registry) -> ScanStore:
    return ScanStore(reg.data)


@router.get("/scans/{scan_id}/parts", response_model=list[Part])
def list_parts(scan_id: str, reg: RegistryDep) -> list[Part]:
    return signing.parts(reg.storage, _store(reg).list_parts(scan_id))


@router.get("/scans/{scan_id}/parts/{part_id}/export")
def export_part(scan_id: str, part_id: str, reg: RegistryDep,
                format: Literal["stp", "stl"] = "stp") -> Response:
    from screwit.cad.part_export import MAX_MESH_BYTES, export_part_mesh

    part = _store(reg).get_part(scan_id, part_id)
    if part is None:
        raise HTTPException(status_code=404, detail="part not found")
    head = reg.storage.head(part.mesh_uri)
    if head is None:
        raise HTTPException(status_code=404, detail="component mesh not found")
    if head["size"] > MAX_MESH_BYTES:
        raise HTTPException(status_code=413, detail="This mesh is too large to export (64 MB limit).")
    try:
        data = export_part_mesh(reg.storage.get_bytes(part.mesh_uri), format)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    filename = quote(f"{part.label}.{format}", safe="")
    return Response(data, media_type="model/step" if format == "stp" else "model/stl",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
                             "Cache-Control": "no-store"})


@router.get("/scans/{scan_id}/export")
def export_object(scan_id: str, reg: RegistryDep,
                  format: Literal["stp", "stl"] = "stl") -> Response:
    import trimesh

    from screwit.cad.part_export import MAX_MESH_BYTES, export_part_mesh

    parts = _store(reg).list_parts(scan_id)
    if not parts:
        raise HTTPException(status_code=404, detail="No object parts are available to export yet")
    total = 0
    meshes = []
    for part in parts:
        head = reg.storage.head(part.mesh_uri)
        if head is None:
            raise HTTPException(status_code=404, detail="An object part mesh is missing")
        total += head["size"]
        if total > MAX_MESH_BYTES:
            raise HTTPException(status_code=413, detail="This object is too large to export (64 MB limit).")
        scene = trimesh.load(io.BytesIO(reg.storage.get_bytes(part.mesh_uri)), file_type="glb", force="scene")
        meshes.append(scene.to_geometry())
    try:
        data = export_part_mesh(trimesh.util.concatenate(meshes).export(file_type="glb"), format)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    filename = quote(f"{scan_id}.{format}", safe="")
    return Response(data, media_type="model/step" if format == "stp" else "model/stl",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
                             "Cache-Control": "no-store"})


@router.patch("/scans/{scan_id}/parts/{part_id}", response_model=Part)
def patch_part(scan_id: str, part_id: str, patch: PartPatch,
               reg: RegistryDep) -> Part:
    store = _store(reg)
    before = store.get_part(scan_id, part_id)
    if before is None:
        raise HTTPException(status_code=404, detail="part not found")
    try:
        updated = (_apply_patch(before, patch) if reg.settings.is_mock
                   else reg.perception.patch_part(scan_id, part_id, patch, reg.storage))
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
    before = store.list_parts(scan_id)
    before_ids = {p.part_id for p in before}
    if len(set(req.part_ids)) < 2:
        raise HTTPException(status_code=400, detail="Choose two different parts")
    missing = [pid for pid in req.part_ids if pid not in before_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown part(s): {missing}")

    graph = store.get_scene_graph(scan_id)
    merged = (_merge_demo(reg, before, req, scan_id) if reg.settings.is_mock
              else reg.perception.merge_parts(scan_id, req, reg.storage))

    # Retire the sources, add the merged part, stale CAD that referenced any retired part (SPEC 5).
    kept = [p for p in store.list_parts(scan_id) if p.part_id not in set(req.part_ids)]
    kept.append(merged)
    store.save_parts(scan_id, kept)

    retired = _retired_ids(reg, req, before_ids, kept)
    store.mark_cad_stale(scan_id, store.cad_jobs_referencing(scan_id, retired))
    with db_session() as session:
        session.add(MergeHistoryRow(scan_id=scan_id, before_json=_parts_json(before),
                                   after_json=_parts_json(kept),
                                   graph_json=graph.model_dump_json() if graph else None))
        session.commit()
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


class EditHistory(BaseModel):
    can_undo: bool
    can_restore_example: bool


def _parts_json(parts: list[Part]) -> str:
    return json.dumps([p.model_dump(mode="json") for p in sorted(parts, key=lambda p: p.part_id)], sort_keys=True)


def _merge_demo(reg: Registry, before: list[Part], req: MergeRequest, scan_id: str) -> Part:
    import numpy as np
    import trimesh

    by_id = {p.part_id: p for p in before}
    sources = [by_id[pid] for pid in req.part_ids]
    meshes = []
    for source in sources:
        scene = trimesh.load(io.BytesIO(reg.storage.get_bytes(source.mesh_uri)), file_type="glb", force="scene")
        meshes.append(scene.to_geometry())
    mesh = trimesh.util.concatenate(meshes)
    part_id = "merged_" + uuid.uuid4().hex[:12]
    key = f"scans/{scan_id}/parts/{part_id}.glb"
    dest = reg.settings.data_dir / "generated" / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(mesh.export(file_type="glb"))
    center = mesh.bounds.mean(axis=0).tolist()
    raw = sources[0].model_dump()
    raw.update(color_source_id=sources[0].color_source_id or sources[0].part_id,
               part_id=part_id, label=req.label or " + ".join(p.label for p in sources),
               mesh_uri=key, centroid_mm=center, measurement_revision=1, source="user_edited",
               features=[], dimensions=[],
               obb={"center": center, "extent": np.maximum(mesh.extents, 0.001).tolist(),
                    "rotation_quat": [0, 0, 0, 1]},
               description="Combined parts: " + ", ".join(p.label for p in sources))
    return Part.model_validate(raw)


@router.get("/scans/{scan_id}/parts/history", response_model=EditHistory)
def edit_history(scan_id: str, reg: RegistryDep) -> EditHistory:
    with db_session() as session:
        row = session.exec(select(MergeHistoryRow).where(
            MergeHistoryRow.scan_id == scan_id, MergeHistoryRow.undone == False  # noqa: E712
        ).order_by(MergeHistoryRow.id.desc())).first()
        return EditHistory(can_undo=row is not None,
                           can_restore_example=reg.settings.is_mock and bool(reg.data.parts(scan_id)))


@router.post("/scans/{scan_id}/parts/undo", response_model=list[Part])
def undo_merge(scan_id: str, reg: RegistryDep) -> list[Part]:
    store = _store(reg)
    with db_session() as session:
        row = session.exec(select(MergeHistoryRow).where(
            MergeHistoryRow.scan_id == scan_id, MergeHistoryRow.undone == False  # noqa: E712
        ).order_by(MergeHistoryRow.id.desc())).first()
        if row is None:
            raise HTTPException(status_code=409, detail="No merge to undo")
        current = store.list_parts(scan_id)
        if _parts_json(current) != row.after_json:
            raise HTTPException(status_code=409, detail="Parts changed after this merge; undo would overwrite edits")
        restored = [Part.model_validate(p) for p in json.loads(row.before_json)]
        if not reg.settings.is_mock:
            reg.storage.put_bytes(f"scans/{scan_id}/parts.json", row.before_json.encode(), "application/json")
            if row.graph_json:
                reg.storage.put_bytes(f"scans/{scan_id}/scene_graph.json", row.graph_json.encode(), "application/json")
        store.save_parts(scan_id, restored)
        if row.graph_json:
            graph = SceneGraph.model_validate_json(row.graph_json)
            store.save_scene_graph(graph, restored[0].reconstruction_revision)
        store.mark_cad_stale(scan_id, store.cad_jobs_referencing(scan_id, {p.part_id for p in current}))
        row.undone = True
        session.add(row)
        session.commit()
    return signing.parts(reg.storage, restored)


@router.post("/scans/{scan_id}/parts/restore-example", response_model=list[Part])
def restore_example(scan_id: str, reg: RegistryDep) -> list[Part]:
    original = reg.data.parts(scan_id) if reg.settings.is_mock else []
    if not original:
        raise HTTPException(status_code=404, detail="This is not a demo example")
    store = _store(reg)
    current = store.list_parts(scan_id)
    store.mark_cad_stale(scan_id, store.cad_jobs_referencing(scan_id, {p.part_id for p in current}))
    store.save_parts(scan_id, original)
    with db_session() as session:
        for row in session.exec(select(MergeHistoryRow).where(MergeHistoryRow.scan_id == scan_id)).all():
            row.undone = True
            session.add(row)
        session.commit()
    return signing.parts(reg.storage, original)
