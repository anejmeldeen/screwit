"""Exercise real cookie sessions and ownership with separate browser clients."""

import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from screwit.app.accounts import COOKIE_NAME, initialize_accounts
from screwit.app.db import BuildOwnerRow, SessionRow, UserRow, db_session


def other_browser(client):
    return TestClient(client.app, client=("127.0.0.1", 50001))


def register(client, username="alice"):
    response = client.post("/api/auth/register", json={"username": username, "password": "a-password"})
    assert response.status_code == 201, response.text
    return response


def create_build(client):
    response = client.post("/api/scans", json={"input_type": "video"})
    assert response.status_code == 200, response.text
    return response.json()["scan_id"]


def test_login_required_and_bad_password(make_client):
    client = make_client(login=False)
    for path in ("/api/auth/me", "/api/builds", "/api/demo-scans", "/api/scans/scn_trap",
                 "/api/assets/scans/scn_trap/mesh/object.glb?exp=0&sig=fixture"):
        assert client.get(path).status_code == 401
    assert client.post("/api/scans", json={"input_type": "video"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "incorrect"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "nobody", "password": "incorrect"}).status_code == 401


def test_admin_examples_and_cookie(client):
    assert client.get("/api/auth/me").json() == {"username": "admin"}
    assert {item["scan_id"] for item in client.get("/api/builds").json()} >= {"scn_trap", "scn_knob"}
    login = client.post("/api/auth/login", json={"username": "ADMIN", "password": "password123"})
    assert login.status_code == 200
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    with db_session() as session:
        assert "password123" not in session.get(UserRow, "admin").password_hash
        assert all(row.token_hash != client.cookies[COOKIE_NAME] for row in session.exec(select(SessionRow)).all())


def test_registration_and_builds_are_separate(client):
    alice = other_browser(client)
    register(alice, " Alice ")
    assert alice.get("/api/auth/me").json() == {"username": "alice"}
    assert alice.get("/api/builds").json() == []
    assert alice.get("/api/demo-scans").json() == []
    duplicate = alice.post("/api/auth/register", json={"username": "ALICE", "password": "another-password"})
    assert duplicate.status_code == 409
    assert alice.post("/api/auth/register",
                      json={"username": "admin", "password": "another-password"}).status_code == 409
    alice_id, admin_id = create_build(alice), create_build(client)
    assert [item["scan_id"] for item in alice.get("/api/builds").json()] == [alice_id]
    assert alice_id not in {item["scan_id"] for item in client.get("/api/builds").json()}
    assert alice.get(f"/api/scans/{admin_id}").status_code == 404
    assert client.get(f"/api/scans/{alice_id}").status_code == 404
    assert alice.get(f"/api/scans/{alice_id}").status_code == 200
    # A startup migration must never claim another account's builds for admin.
    initialize_accounts(client.app.state.settings.fixtures_dir)
    assert alice.get(f"/api/scans/{alice_id}").status_code == 200


@pytest.mark.parametrize("method,suffix,payload", [
    ("GET", "", None), ("GET", "/parts", None), ("GET", "/reconstruction", None),
    ("GET", "/scene-graph", None), ("GET", "/events", None),
    ("GET", "/export?format=stl", None), ("GET", "/parts/p1/export?format=stl", None),
    ("GET", "/cad/cad_001", None), ("GET", "/parts/history", None),
    ("POST", "/start", None), ("POST", "/chat", {"message": "hello"}),
    ("PATCH", "/parts/p1", {"label": "changed"}),
    ("POST", "/parts/restore-example", None), ("POST", "/parts/undo", None),
    ("POST", "/scale", {}), ("POST", "/parts/merge", {}), ("POST", "/measure", {}),
    ("GET", "/chat", None), ("POST", "/cad", {}),
])
def test_other_accounts_cannot_read_or_modify_examples(client, method, suffix, payload):
    alice = other_browser(client)
    register(alice)
    assert alice.request(method, f"/api/scans/scn_trap{suffix}", json=payload).status_code == 404


def test_assets_and_uploads_check_owner_even_with_valid_signature(client):
    alice = other_browser(client)
    register(alice)
    assert alice.get("/api/assets/scans/scn_trap/mesh/object.glb?exp=0&sig=fixture").status_code == 404
    reg = client.app.state.registry

    class Uploadable:
        def accept_upload(self, *args):
            raise AssertionError("An unauthorized upload reached storage")

    reg.storage = Uploadable()
    assert alice.post("/api/storage/upload", data={"key": "scans/scn_trap/input/a.mov"},
                      files={"file": ("a.mov", b"bad", "video/quicktime")}).status_code == 404
    scan_id = create_build_for_path_test(alice)
    assert alice.post("/api/storage/upload", data={"key": f"scans/{scan_id}/../../scn_trap/input/a.mov"},
                      files={"file": ("a.mov", b"bad", "video/quicktime")}).status_code == 404


def create_build_for_path_test(client):
    with db_session() as session:
        session.add(BuildOwnerRow(scan_id="owned", username="alice"))
        session.commit()
    return "owned"


def test_logout_revokes_session_and_expiry_is_enforced(client):
    cookie = client.cookies[COOKIE_NAME]
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.set(COOKIE_NAME, cookie)
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.clear()
    client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
    with db_session() as session:
        for row in session.exec(select(SessionRow)).all():
            row.expires_at = int(time.time()) - 1
            session.add(row)
        session.commit()
    assert client.get("/api/auth/me").status_code == 401


def test_accounts_builds_and_sessions_survive_restart(client):
    alice = other_browser(client)
    register(alice)
    scan_id = create_build(alice)
    cookie = alice.cookies[COOKIE_NAME]
    from screwit.app.db import reset_engine
    from screwit.app.main import create_app

    reset_engine()
    restarted = TestClient(create_app(), client=("127.0.0.1", 50002))
    restarted.cookies.set(COOKIE_NAME, cookie)
    assert restarted.get("/api/auth/me").json() == {"username": "alice"}
    assert [item["scan_id"] for item in restarted.get("/api/builds").json()] == [scan_id]
    restarted.cookies.clear()
    assert restarted.post("/api/auth/login", json={"username": "alice", "password": "a-password"}).status_code == 200


def test_existing_unowned_build_is_adopted_by_admin(client):
    scan_id = create_build(client)
    with db_session() as session:
        session.delete(session.get(BuildOwnerRow, scan_id))
        session.commit()
    initialize_accounts(client.app.state.settings.fixtures_dir)
    assert client.get(f"/api/scans/{scan_id}").status_code == 200


def test_cross_site_login_is_rejected(client):
    assert client.post("/api/auth/login", json={"username": "admin", "password": "password123"},
                       headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_cad_ids_cannot_overwrite_another_build(client):
    from screwit.app.store import ScanStore

    reg = client.app.state.registry
    store = ScanStore(reg.data)
    request = reg.data.cad_request("scn_knob")
    result = reg.data.cad_result("scn_knob")
    assert request is not None and result is not None
    store.save_cad_job(request, result)
    another_request = request.model_copy(update={"scan_id": "another_scan"})
    another_result = result.model_copy(update={"scan_id": "another_scan", "stale": True})
    store.save_cad_job(another_request, another_result)
    assert store.get_cad_job("scn_knob", request.cad_id)[1].stale == result.stale
    assert store.get_cad_job("another_scan", request.cad_id)[1].stale is True
