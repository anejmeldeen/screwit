"""Basic local sign-up, sign-in and sign-out."""

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError

from screwit.app.accounts import (
    COOKIE_NAME,
    Account,
    Credentials,
    UserDep,
    hash_password,
    password_matches,
    start_session,
    token_hash,
)
from screwit.app.db import BuildOwnerRow, SeedAccountRow, SessionRow, UserRow, db_session

router = APIRouter(prefix="/auth")


@router.post("/register", response_model=Account, status_code=201)
def register(credentials: Credentials, request: Request, response: Response) -> Account:
    with db_session() as session:
        session.add(UserRow(username=credentials.username, password_hash=hash_password(credentials.password)))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status_code=409, detail="That username is already taken.") from None
    start_session(credentials.username, request, response)
    return Account(username=credentials.username)


@router.post("/login", response_model=Account)
def login(credentials: Credentials, request: Request, response: Response) -> Account:
    with db_session() as session:
        user = session.get(UserRow, credentials.username)
        # Still run the password derivation for unknown users.
        stored = user.password_hash if user else f"{'0' * 32}:{'0' * 64}"
        valid = password_matches(credentials.password, stored)
        if user is None or not valid:
            raise HTTPException(status_code=401, detail="Incorrect username or password.")
    start_session(credentials.username, request, response)
    return Account(username=credentials.username)


@router.get("/me", response_model=Account)
def me(user: UserDep, response: Response) -> Account:
    response.headers["Cache-Control"] = "no-store"
    return user


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    with db_session() as session:
        row = session.get(SessionRow, token_hash(request.cookies.get(COOKIE_NAME, "")))
        if row is not None:
            session.delete(row)
            session.commit()
    response = Response(status_code=204, headers={"Cache-Control": "no-store"})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@router.patch("/username", response_model=Account)
def change_username(credentials: Credentials, user: UserDep) -> Account:
    with db_session() as session:
        row = session.get(UserRow, user.username)
        if row is None or not password_matches(credentials.password, row.password_hash):
            raise HTTPException(403, "Current password is incorrect.")
        if row.username != credentials.username:
            try:
                row.username = credentials.username
                session.flush()
                for table in (BuildOwnerRow, SessionRow, SeedAccountRow):
                    session.execute(update(table).where(table.username == user.username)
                                    .values(username=credentials.username))
                session.commit()
            except IntegrityError:
                session.rollback()
                raise HTTPException(409, "That username is already taken.") from None
    return Account(username=credentials.username)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


@router.patch("/password", status_code=204)
def change_password(body: PasswordChange, user: UserDep, request: Request) -> Response:
    with db_session() as session:
        row = session.get(UserRow, user.username)
        if row is None or not password_matches(body.current_password, row.password_hash):
            raise HTTPException(403, "Current password is incorrect.")
        row.password_hash = hash_password(body.new_password)
        current = token_hash(request.cookies.get(COOKIE_NAME, ""))
        session.execute(delete(SessionRow).where(SessionRow.username == user.username,
                                                 SessionRow.token_hash != current))
        session.commit()
    return Response(status_code=204)
