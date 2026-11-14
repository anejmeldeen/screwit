"""Database layer: SQLModel tables and the session factory (SPEC 7 P4).

SQLite by default (file in the data dir), Postgres when DATABASE_URL points at one. Rows store the
contract JSON blobs as text; the contract models in screwit.contracts stay the single source of truth,
so nothing is persisted that has not been validated by a contract first.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from screwit.app.settings import settings


class ScanRow(SQLModel, table=True):
    __tablename__ = "scans"
    scan_id: str = Field(primary_key=True)
    created_at: datetime
    status: str
    reconstruction_revision: int = 0
    scan_json: str  # full Scan contract (4.1), includes stages


class ReconstructionRow(SQLModel, table=True):
    __tablename__ = "reconstruction_revisions"
    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True)
    reconstruction_revision: int
    recon_json: str  # Reconstruction contract (4.2)


class PartRow(SQLModel, table=True):
    __tablename__ = "parts"
    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True)
    part_id: str = Field(index=True)
    reconstruction_revision: int
    measurement_revision: int
    part_json: str  # Part contract (4.3)


class SceneGraphRow(SQLModel, table=True):
    __tablename__ = "scene_graphs"
    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True)
    reconstruction_revision: int
    graph_json: str  # SceneGraph contract (4.4)


class ChatMessageRow(SQLModel, table=True):
    __tablename__ = "chat_messages"
    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True)
    message_id: str
    role: str
    created_at: datetime
    message_json: str  # ChatMessage contract (4.8)


class CadJobRow(SQLModel, table=True):
    __tablename__ = "cad_jobs"
    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(index=True)
    cad_id: str = Field(index=True)
    reconstruction_revision: int
    part_measurement_revisions: str  # JSON dict {part_id: revision} the request pinned
    request_json: str  # CadRequest (4.6)
    result_json: str | None = None  # CadResult (4.7), filled when the job finishes


_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        url = settings().database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        if url.startswith("sqlite:///"):
            settings().data_dir.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


def init_db() -> None:
    SQLModel.metadata.create_all(engine())


def reset_engine() -> None:
    """Tests point DATABASE_URL at a temp file, then call this to rebuild the engine."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def db_session() -> Iterator[Session]:
    with Session(engine()) as session:
        yield session
