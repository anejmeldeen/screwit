"""Contract-aware persistence over db.py (SPEC 4, 5).

Reads prefer the database, then fall back to the fixtures tree, so a freshly created scan is served
from the DB while the pre-processed hero scans are served from fixtures in mock mode. Every value
written or returned has been through a contract model (AGENTS.md: never persist an unvalidated dict).
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlmodel import select

from screwit.app.db import (
    CadJobRow,
    ChatMessageRow,
    PartRow,
    ReconstructionRow,
    ScanRow,
    SceneGraphRow,
    db_session,
)
from screwit.app.fakes import FixtureData
from screwit.contracts import (
    CadRequest,
    CadResult,
    ChatMessage,
    Part,
    Reconstruction,
    Scan,
    SceneGraph,
)


class ScanStore:
    def __init__(self, data: FixtureData) -> None:
        self._data = data

    # -------------------------------------------------------------- scans
    def get_scan(self, scan_id: str) -> Scan | None:
        with db_session() as s:
            row = s.get(ScanRow, scan_id)
            if row is not None:
                return Scan.model_validate_json(row.scan_json)
        return self._data.scan(scan_id)

    def save_scan(self, scan: Scan) -> None:
        with db_session() as s:
            row = s.get(ScanRow, scan.scan_id)
            if row is None:
                row = ScanRow(scan_id=scan.scan_id, created_at=scan.created_at, status=scan.status,
                              reconstruction_revision=scan.reconstruction_revision,
                              scan_json=scan.model_dump_json())
            else:
                row.status = scan.status
                row.reconstruction_revision = scan.reconstruction_revision
                row.scan_json = scan.model_dump_json()
            s.add(row)
            s.commit()

    # -------------------------------------------------------------- reconstruction
    def get_reconstruction(self, scan_id: str) -> Reconstruction | None:
        with db_session() as s:
            rows = s.exec(
                select(ReconstructionRow).where(ReconstructionRow.scan_id == scan_id)
            ).all()
            if rows:
                latest = max(rows, key=lambda r: r.reconstruction_revision)
                return Reconstruction.model_validate_json(latest.recon_json)
        return self._data.reconstruction(scan_id)

    def save_reconstruction(self, recon: Reconstruction) -> None:
        with db_session() as s:
            s.add(ReconstructionRow(scan_id=recon.scan_id,
                                    reconstruction_revision=recon.reconstruction_revision,
                                    recon_json=recon.model_dump_json()))
            s.commit()

    # -------------------------------------------------------------- parts
    def list_parts(self, scan_id: str) -> list[Part]:
        with db_session() as s:
            rows = s.exec(select(PartRow).where(PartRow.scan_id == scan_id)).all()
            if rows:
                return [Part.model_validate_json(r.part_json) for r in rows]
        return self._data.parts(scan_id)

    def get_part(self, scan_id: str, part_id: str) -> Part | None:
        return next((p for p in self.list_parts(scan_id) if p.part_id == part_id), None)

    def save_parts(self, scan_id: str, parts: list[Part]) -> None:
        with db_session() as s:
            for r in s.exec(select(PartRow).where(PartRow.scan_id == scan_id)).all():
                s.delete(r)
            for p in parts:
                s.add(PartRow(scan_id=scan_id, part_id=p.part_id,
                              reconstruction_revision=p.reconstruction_revision,
                              measurement_revision=p.measurement_revision,
                              part_json=p.model_dump_json()))
            s.commit()

    def save_part(self, scan_id: str, part: Part) -> None:
        parts = self.list_parts(scan_id)
        replaced = [part if p.part_id == part.part_id else p for p in parts]
        if part.part_id not in {p.part_id for p in parts}:
            replaced.append(part)
        self.save_parts(scan_id, replaced)

    # -------------------------------------------------------------- scene graph
    def get_scene_graph(self, scan_id: str) -> SceneGraph | None:
        with db_session() as s:
            rows = s.exec(select(SceneGraphRow).where(SceneGraphRow.scan_id == scan_id)).all()
            if rows:
                latest = max(rows, key=lambda r: r.reconstruction_revision)
                return SceneGraph.model_validate_json(latest.graph_json)
        return self._data.scene_graph(scan_id)

    def save_scene_graph(self, graph: SceneGraph, reconstruction_revision: int) -> None:
        with db_session() as s:
            s.add(SceneGraphRow(scan_id=graph.scan_id,
                                reconstruction_revision=reconstruction_revision,
                                graph_json=graph.model_dump_json()))
            s.commit()

    # -------------------------------------------------------------- chat
    def list_chat(self, scan_id: str) -> list[ChatMessage]:
        with db_session() as s:
            rows = s.exec(
                select(ChatMessageRow).where(ChatMessageRow.scan_id == scan_id).order_by(ChatMessageRow.id)
            ).all()
            return [ChatMessage.model_validate_json(r.message_json) for r in rows]

    def append_chat(self, scan_id: str, message: ChatMessage) -> None:
        with db_session() as s:
            s.add(ChatMessageRow(scan_id=scan_id, message_id=message.message_id, role=message.role,
                                 created_at=message.created_at, message_json=message.model_dump_json()))
            s.commit()

    # -------------------------------------------------------------- CAD jobs
    def save_cad_job(self, req: CadRequest, result: CadResult) -> None:
        with db_session() as s:
            row = s.exec(
                select(CadJobRow).where(CadJobRow.cad_id == req.cad_id)
            ).first()
            if row is None:
                row = CadJobRow(scan_id=req.scan_id, cad_id=req.cad_id)
            row.reconstruction_revision = req.reconstruction_revision
            row.part_measurement_revisions = json.dumps(req.part_measurement_revisions)
            row.request_json = req.model_dump_json()
            row.result_json = result.model_dump_json()
            s.add(row)
            s.commit()

    def get_cad_job(self, scan_id: str, cad_id: str) -> tuple[CadRequest, CadResult] | None:
        with db_session() as s:
            row = s.exec(
                select(CadJobRow).where(CadJobRow.scan_id == scan_id, CadJobRow.cad_id == cad_id)
            ).first()
            if row is not None and row.result_json is not None:
                return (CadRequest.model_validate_json(row.request_json),
                        CadResult.model_validate_json(row.result_json))
            if row is not None:
                return None
        # Pre-processed hero scans ship one CAD job as fixtures (same fallback as scans, parts, ...).
        req, result = self._data.cad_request(scan_id), self._data.cad_result(scan_id)
        if req is not None and result is not None and req.cad_id == cad_id == result.cad_id:
            return req, result
        return None

    def mark_scan_cad_stale(self, scan_id: str) -> None:
        """Mark every stored CAD result for this scan stale (SPEC 5 /scale, PATCH, merge)."""
        with db_session() as s:
            for row in s.exec(select(CadJobRow).where(CadJobRow.scan_id == scan_id)).all():
                if row.result_json is None:
                    continue
                result = CadResult.model_validate_json(row.result_json)
                if not result.stale:
                    row.result_json = result.model_copy(update={"stale": True}).model_dump_json()
                    s.add(row)
            s.commit()

    def cad_jobs_referencing(self, scan_id: str, part_ids: set[str]) -> list[str]:
        """cad_ids whose request pinned any of these parts (for targeted staleness on PATCH/merge)."""
        out: list[str] = []
        with db_session() as s:
            for row in s.exec(select(CadJobRow).where(CadJobRow.scan_id == scan_id)).all():
                pinned = set(json.loads(row.part_measurement_revisions))
                if pinned & part_ids:
                    out.append(row.cad_id)
        return out

    def mark_cad_stale(self, scan_id: str, cad_ids: list[str]) -> None:
        with db_session() as s:
            for cid in cad_ids:
                row = s.exec(
                    select(CadJobRow).where(CadJobRow.scan_id == scan_id, CadJobRow.cad_id == cid)
                ).first()
                if row is None or row.result_json is None:
                    continue
                result = CadResult.model_validate_json(row.result_json)
                row.result_json = result.model_copy(update={"stale": True}).model_dump_json()
                s.add(row)
            s.commit()


def utcnow() -> datetime:
    return datetime.now().astimezone()
