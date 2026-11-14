"""Dev storage routes (SPEC 5): delegate to P5's LocalStorage, 501 when it is absent.

The mock FixtureStorage deliberately does not implement accept_upload / verify_signed (those are P5's
LocalStorage methods), so these routes return 501 here. This documents the getattr-delegation contract;
the real local-disk path is exercised in P5's storage tests.
"""

from __future__ import annotations


def test_upload_501_without_local_storage(client):
    r = client.post("/api/storage/upload", data={"key": "scans/x/input/a.mov"},
                    files={"file": ("a.mov", b"data", "video/quicktime")})
    assert r.status_code == 501


def test_mock_assets_served_and_bad_signature_refused(client):
    ok = client.get("/api/assets/scans/scn_trap/mesh/object.glb", params={"exp": "0", "sig": "fixture"})
    assert ok.status_code == 200 and ok.content[:4] == b"glTF"
    assert ok.headers["content-type"] == "model/gltf-binary"
    bad = client.get("/api/assets/scans/scn_trap/mesh/object.glb", params={"exp": "0", "sig": "x"})
    assert bad.status_code == 403
    escape = client.get("/api/assets/scans/../../AGENTS.md", params={"exp": "0", "sig": "fixture"})
    assert escape.status_code in (403, 404)


def test_signed_get_501_without_verify_signed(client):
    reg = client.app.state.registry
    original = reg.storage

    class NoVerify:
        def __getattr__(self, name):
            if name == "verify_signed":
                raise AttributeError(name)
            return getattr(original, name)

    reg.storage = NoVerify()
    try:
        r = client.get("/api/assets/scans/scn_trap/mesh/object.glb", params={"exp": "0", "sig": "x"})
        assert r.status_code == 501
    finally:
        reg.storage = original


def test_fake_storage_delegation_when_present(client):
    """When the storage backend implements the methods, the route uses them (simulated here)."""
    from screwit.app.registry import Registry

    reg: Registry = client.app.state.registry

    class Localish:
        def __init__(self, inner):
            self._inner = inner
            self.last = None

        def accept_upload(self, fields, filename, stream):
            self.last = (fields, filename, stream.read())

        def verify_signed(self, key, exp, sig):
            return sig == "good"

        def head(self, key):
            return {"size": 3, "content_type": "text/plain"}

        def get_bytes(self, key):
            return b"abc"

    reg.storage = Localish(reg.storage)  # type: ignore[assignment]

    up = client.post("/api/storage/upload", data={"key": "scans/x/input/a.mov"},
                     files={"file": ("a.mov", b"data", "video/quicktime")})
    assert up.status_code == 204
    assert reg.storage.last[1] == "a.mov"

    bad = client.get("/api/assets/scans/x/a.txt", params={"exp": "0", "sig": "bad"})
    assert bad.status_code == 403
    good = client.get("/api/assets/scans/x/a.txt", params={"exp": "0", "sig": "good"})
    assert good.status_code == 200 and good.content == b"abc"
