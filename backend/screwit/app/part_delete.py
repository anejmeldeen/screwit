"""Delete one part of a workspace so that undo can bring it back (SPEC 5 part edits: PATCH, merge, delete).

The part leaves the database's parts list and the scene graph, together with the relations and failure
hypotheses that name it. In live mode it also leaves parts.json and scene_graph.json in storage, which
the perception service reads for PATCH, merge, measure and rescale. CAD results that pinned the part go
stale. A MergeHistoryRow records the parts before and after plus the scene graph before, so
POST /parts/undo puts the part back the same way it undoes a combine. The part's mesh and thumbnail stay
in storage because undo needs them.

Object-agnostic: nothing here knows what kind of object the part belongs to.
"""

from __future__ import annotations

import json
import logging

from sqlmodel import select

from screwit.app.db import MergeHistoryRow, SceneGraphRow, db_session
from screwit.app.registry import Registry
from screwit.app.store import ScanStore
from screwit.contracts import Part, SceneGraph

log = logging.getLogger(__name__)


class PartDeleteRefused(ValueError):
    """The part exists but deleting it is refused; the message says why and is shown to the user."""


def parts_json(parts: list[Part]) -> str:
    """Canonical JSON of a parts list, sorted by part id. The undo route compares the current parts
    with a history row's after_json in this form, so every writer of a history row must use it."""
    return json.dumps([p.model_dump(mode="json") for p in sorted(parts, key=lambda p: p.part_id)],
                      sort_keys=True)


def latest_graph_revision(scan_id: str) -> int:
    """The highest revision any stored scene graph row of this scan carries. The store reads back the
    row with the highest (revision, id), so a graph saved at this revision or above becomes the latest."""
    with db_session() as session:
        revisions = session.exec(select(SceneGraphRow.reconstruction_revision)
                                 .where(SceneGraphRow.scan_id == scan_id)).all()
    return max(revisions, default=0)


def graph_revision(store: ScanStore, scan_id: str, parts: list[Part]) -> int:
    """The revision to save a new scene graph row at so that it reads back as the latest."""
    scan = store.get_scan(scan_id)
    revision = scan.reconstruction_revision if scan else max(
        (p.reconstruction_revision for p in parts), default=0)
    return max(revision, latest_graph_revision(scan_id))


def graph_for_parts(graph: SceneGraph, part_ids: list[str]) -> SceneGraph:
    """The graph with its part list matched to part_ids (graph order kept, ids it lacks appended), and
    every relation or failure hypothesis that names a part outside part_ids dropped, so it validates."""
    known = set(part_ids)
    order = [pid for pid in graph.parts if pid in known]
    order += [pid for pid in dict.fromkeys(part_ids) if pid not in order]
    return SceneGraph.model_validate({
        **graph.model_dump(mode="json"),
        "parts": order,
        "relations": [r.model_dump(mode="json") for r in graph.relations if r.a in known and r.b in known],
        "failure_hypotheses": [h.model_dump(mode="json") for h in graph.failure_hypotheses
                               if h.part_id in known],
    })


def delete_part(reg: Registry, scan_id: str, part_id: str) -> Part:
    """Delete part_id from the scan and return the deleted Part.

    Raises KeyError(part_id) when the scan has no such part, and PartDeleteRefused for the last part of
    an example whose parts come from the fixtures. Works the same in mock and live mode; only live mode
    has storage files to update.
    """
    store = ScanStore(reg.data)
    before = store.list_parts(scan_id)
    part = next((p for p in before if p.part_id == part_id), None)
    if part is None:
        raise KeyError(part_id)
    remaining = [p for p in before if p.part_id != part_id]
    if not remaining and reg.data.parts(scan_id):
        # The store serves an example's fixture parts whenever the database holds none for it, so an
        # empty list would bring every original part back instead of leaving the example empty.
        raise PartDeleteRefused("An example workspace must keep at least one part.")

    graph = store.get_scene_graph(scan_id)
    graph_before = graph_for_parts(graph, [p.part_id for p in before]) if graph is not None else None
    graph_after = graph_for_parts(graph, [p.part_id for p in remaining]) if graph is not None else None

    # The undo step is written before anything is removed, so a deleted part always has one.
    history_id = _record_history(scan_id, before, remaining, graph_before)
    try:
        store.save_parts(scan_id, remaining)
    except Exception:
        _forget_history(history_id)  # nothing was deleted, so the undo step would undo nothing
        raise
    if graph_after is not None:
        store.save_scene_graph(graph_after, graph_revision(store, scan_id, before))
    store.mark_cad_stale(scan_id, store.cad_jobs_referencing(scan_id, {part_id}))
    if not reg.settings.is_mock:
        _update_live_storage(reg, store, scan_id, part_id)
    return part


def _record_history(scan_id: str, before: list[Part], after: list[Part], graph: SceneGraph | None) -> int | None:
    """One undo step. before_json keeps the list order, so undo puts the part back where it was in the
    list; after_json is canonical for the undo route's "changed since?" check."""
    with db_session() as session:
        row = MergeHistoryRow(scan_id=scan_id,
                              before_json=json.dumps([p.model_dump(mode="json") for p in before], sort_keys=True),
                              after_json=parts_json(after),
                              graph_json=graph.model_dump_json() if graph is not None else None)
        session.add(row)
        session.commit()
        return row.id


def _forget_history(history_id: int | None) -> None:
    try:
        with db_session() as session:
            row = session.get(MergeHistoryRow, history_id) if history_id is not None else None
            if row is not None:
                session.delete(row)
                session.commit()
    except Exception:  # noqa: BLE001 - the original error is the one to report
        log.exception("could not remove the undo step of a failed part delete")


def _update_live_storage(reg: Registry, store: ScanStore, scan_id: str, part_id: str) -> None:
    """Live mode: drop the part from parts.json and scene_graph.json as stored (add_part extends them
    the same way), so nothing another writer put there is lost. Runs after the database is updated and
    logs instead of raising: the part is already deleted for the user, and a stale entry left in
    storage only means the perception service still knows the part's id."""
    parts_key, graph_key = f"scans/{scan_id}/parts.json", f"scans/{scan_id}/scene_graph.json"
    try:
        if reg.storage.head(parts_key) is not None:
            stored = [Part.model_validate(p) for p in json.loads(reg.storage.get_bytes(parts_key))]
        else:
            stored = store.list_parts(scan_id)
        kept = [p for p in stored if p.part_id != part_id]
        payload = json.dumps([p.model_dump(mode="json") for p in kept], indent=2).encode()
        reg.storage.put_bytes(parts_key, payload, "application/json")
    except Exception:  # noqa: BLE001
        log.exception("parts.json of %s was not updated after deleting %s", scan_id, part_id)
        return
    try:
        if reg.storage.head(graph_key) is not None:
            graph = SceneGraph.model_validate_json(reg.storage.get_bytes(graph_key))
        else:
            graph = store.get_scene_graph(scan_id)
        if graph is not None:
            updated = graph_for_parts(graph, [p.part_id for p in kept])
            reg.storage.put_bytes(graph_key, updated.model_dump_json(indent=2).encode(), "application/json")
    except Exception:  # noqa: BLE001
        log.exception("scene_graph.json of %s was not updated after deleting %s", scan_id, part_id)
