"""Local accounts, cookie sessions and build ownership, persisted in the app database."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlmodel import select

from screwit.app.db import BuildOwnerRow, DeletedBuildRow, ScanRow, SessionRow, UserRow, db_session

COOKIE_NAME = "screwit_session"
SESSION_SECONDS = 30 * 24 * 60 * 60


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower() if isinstance(value, str) else value


class Account(BaseModel):
    username: str


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000)
    return f"{salt}:{digest.hex()}"


def password_matches(password: str, stored: str) -> bool:
    return hmac.compare_digest(hash_password(password, stored.split(":", 1)[0]), stored)


def initialize_accounts(fixtures_dir: Path) -> None:
    """Seed admin once and preserve existing builds, without ever reassigning an owner."""
    with db_session() as session:
        if session.get(UserRow, "admin") is None:
            session.add(UserRow(username="admin", password_hash=hash_password("password123")))
        scan_ids = {row.scan_id for row in session.exec(select(ScanRow)).all()}
        scan_ids.update(path.parent.name for path in (fixtures_dir / "scans").glob("*/scan.json"))
        deleted_ids = set(session.exec(select(DeletedBuildRow.scan_id)).all())
        for scan_id in scan_ids - deleted_ids:
            if session.get(BuildOwnerRow, scan_id) is None:
                session.add(BuildOwnerRow(scan_id=scan_id, username="admin"))
        session.commit()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(username: str, request: Request, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    with db_session() as session:
        old = session.get(SessionRow, token_hash(request.cookies.get(COOKIE_NAME, "")))
        if old is not None:
            session.delete(old)
        for expired in session.exec(select(SessionRow).where(SessionRow.expires_at <= int(time.time()))).all():
            session.delete(expired)
        session.add(SessionRow(token_hash=token_hash(token), username=username,
                               expires_at=int(time.time()) + SESSION_SECONDS))
        session.commit()
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                        samesite="strict", secure=request.url.scheme == "https", path="/")
    response.headers["Cache-Control"] = "no-store"


def require_user(request: Request) -> Account:
    with db_session() as session:
        row = session.get(SessionRow, token_hash(request.cookies.get(COOKIE_NAME, "")))
        if row is None or row.expires_at <= int(time.time()):
            raise HTTPException(status_code=401, detail="Please log in to continue.")
        return Account(username=row.username)


UserDep = Annotated[Account, Depends(require_user)]


def require_owner(scan_id: str, user: Account) -> None:
    with db_session() as session:
        row = session.get(BuildOwnerRow, scan_id)
        if row is None or row.username != user.username:
            raise HTTPException(status_code=404, detail="Build not found.")


def require_asset_owner(key: str, user: Account) -> None:
    parts = key.split("/")
    if len(parts) < 3 or parts[0] != "scans" or any(p in {"", ".", ".."} for p in parts) or "\\" in key:
        raise HTTPException(status_code=404, detail="Asset not found.")
    require_owner(parts[1], user)


def require_build_access(request: Request, user: UserDep) -> None:
    # Apply centrally to every scan route, including exports, chat and event streams.
    if scan_id := request.path_params.get("scan_id"):
        require_owner(scan_id, user)
    if key := request.path_params.get("key"):
        require_asset_owner(key, user)
