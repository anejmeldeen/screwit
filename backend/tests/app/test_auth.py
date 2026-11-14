"""Auth gate (AGENTS.md decision 1): non-loopback 403, midway 401, loopback admitted."""

from __future__ import annotations


def test_loopback_client_is_admitted(client):
    assert client.get("/api/demo-scans").status_code == 200


def test_non_loopback_client_is_forbidden(make_client):
    c = make_client(client_host=("203.0.113.7", 40000))
    r = c.get("/api/demo-scans")
    assert r.status_code == 403
    assert "loopback" in r.json()["detail"].lower()


def test_midway_mode_always_401(make_client):
    c = make_client(auth="midway")  # even from a loopback client
    r = c.get("/api/demo-scans")
    assert r.status_code == 401
    assert r.json()["detail"] == "Midway integration not configured"


def test_midway_rejects_every_route(make_client):
    c = make_client(auth="midway")
    assert c.get("/api/scans/scn_trap").status_code == 401
    assert c.post("/api/scans/scn_trap/chat", json={"message": "hi"}).status_code == 401
