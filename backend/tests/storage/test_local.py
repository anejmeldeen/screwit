"""LocalStorage: pre-signed POST emulation, signed GET, object round-trip (SPEC 3.3)."""

from __future__ import annotations

import io
import time

import pytest

from screwit.storage.local import LocalStorage

KEY = "scans/scn_t/input/video.mov"
SIGNING_KEY = b"test-signing-key-0123456789abcdef"


@pytest.fixture
def store(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path / "data", signing_key=SIGNING_KEY)


def test_presign_fields(store):
    pre = store.presign_upload(KEY, ["video/quicktime", "video/mp4"], 500_000_000, expires_s=600)
    assert pre.url == "/api/storage/upload"
    assert pre.key == KEY
    for f in ("key", "content_type_allow", "max_bytes", "expires", "signature"):
        assert f in pre.fields
    assert pre.fields["key"] == KEY
    assert pre.fields["max_bytes"] == "500000000"


def test_accept_upload_round_trip(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 1_000_000)
    body = b"\x00\x00\x00\x18ftypqt  " + b"x" * 1000
    store.accept_upload(pre.fields, "video.mov", io.BytesIO(body))
    assert store.head(KEY)["size"] == len(body)
    assert store.get_bytes(KEY) == body


def test_accept_upload_rejects_bad_signature(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 1_000_000)
    pre.fields["signature"] = "deadbeef"
    with pytest.raises(ValueError, match="signature"):
        store.accept_upload(pre.fields, "video.mov", io.BytesIO(b"abc"))


def test_accept_upload_rejects_expired(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 1_000_000, expires_s=-1)
    with pytest.raises(ValueError, match="expired"):
        store.accept_upload(pre.fields, "video.mov", io.BytesIO(b"abc"))


def test_accept_upload_rejects_wrong_content_type(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 1_000_000)
    with pytest.raises(ValueError, match="content type"):
        store.accept_upload(pre.fields, "evil.exe", io.BytesIO(b"abc"))


def test_accept_upload_rejects_oversize(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 100)
    with pytest.raises(ValueError, match="max_bytes"):
        store.accept_upload(pre.fields, "video.mov", io.BytesIO(b"x" * 500))
    # Nothing persisted on failure.
    assert store.head(KEY) is None


def test_accept_upload_rejects_empty(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 1000)
    with pytest.raises(ValueError, match="empty"):
        store.accept_upload(pre.fields, "video.mov", io.BytesIO(b""))


def test_signed_url_and_verify(store):
    url = store.signed_url(KEY, expires_s=900)
    assert url.startswith("/api/assets/scans/scn_t/input/video.mov?exp=")
    exp = url.split("exp=")[1].split("&")[0]
    sig = url.split("sig=")[1]
    assert store.verify_signed(KEY, exp, sig) is True
    assert store.verify_signed(KEY, exp, "bad") is False
    assert store.verify_signed(KEY, str(int(time.time()) - 10), sig) is False


def test_put_get_bytes_and_file(store, tmp_path):
    store.put_bytes("scans/scn_t/cad/cad_1/model.py", b"print(1)", "text/x-python")
    assert store.get_bytes("scans/scn_t/cad/cad_1/model.py") == b"print(1)"
    f = tmp_path / "x.stl"
    f.write_bytes(b"solid")
    store.put_file("scans/scn_t/cad/cad_1/model.stl", f, "model/stl")
    assert store.head("scans/scn_t/cad/cad_1/model.stl")["size"] == 5
    local = store.local_path("scans/scn_t/cad/cad_1/model.stl", tmp_path / "scratch")
    assert local.read_bytes() == b"solid"


def test_traversal_key_rejected(store):
    from screwit.storage.keys import KeyError as StorageKeyError

    with pytest.raises(StorageKeyError):
        store.put_bytes("scans/../etc/x", b"x", "text/plain")


def test_generated_signing_key_is_random():
    from screwit.storage import signing

    a = signing.load_signing_key()
    b = signing.load_signing_key()
    assert a != b  # fresh random each call when the env var is unset
    assert len(a) == 32
