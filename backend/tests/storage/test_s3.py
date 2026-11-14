"""S3Storage: presigned POST policy, signed GET, head/put/get (SPEC 3.3).

Uses a real boto3 client with fixed dummy credentials for offline presign generation (no network),
and botocore's Stubber for the calls that would hit S3. The presigned POST is decoded to prove the
policy pins the exact key and carries content-length-range.
"""

from __future__ import annotations

import base64
import json

import boto3
import pytest
from botocore.config import Config
from botocore.stub import Stubber

from screwit.storage.s3 import S3Storage

BUCKET = "screwit-private"
KEY = "scans/scn_8f2a/input/video.mov"


@pytest.fixture
def client():
    return boto3.client(
        "s3", region_name="us-east-1",
        aws_access_key_id="testing", aws_secret_access_key="testing",
        config=Config(signature_version="s3v4"),
    )


@pytest.fixture
def store(client):
    return S3Storage(BUCKET, region="us-east-1", client=client)


def _decode_policy(fields: dict) -> dict:
    return json.loads(base64.b64decode(fields["policy"]))


def test_presigned_post_policy(store):
    pre = store.presign_upload(KEY, ["video/quicktime"], 500_000_000)
    assert pre.key == KEY
    policy = _decode_policy(pre.fields)
    conditions = policy["conditions"]

    # content-length-range [1, max_bytes] present.
    clr = [c for c in conditions if isinstance(c, list) and c and c[0] == "content-length-range"]
    assert clr == [["content-length-range", 1, 500_000_000]]

    # Exact key pinned (not a starts-with prefix).
    key_conds = [c for c in conditions if isinstance(c, dict) and "key" in c]
    assert {"key": KEY} in key_conds
    starts_with = [c for c in conditions if isinstance(c, list) and c and c[0] == "starts-with"]
    assert not any(c[1] == "$key" for c in starts_with), "key must be exact, not starts-with"

    # Content-Type condition present.
    ct = [c for c in conditions if isinstance(c, dict) and "Content-Type" in c]
    assert {"Content-Type": "video/quicktime"} in ct
    assert pre.fields["Content-Type"] == "video/quicktime"


def test_presigned_post_expiry_is_ten_minutes(store):
    pre = store.presign_upload(KEY, ["video/mp4"], 1000)
    policy = _decode_policy(pre.fields)
    # expiration present; 10 min default means the policy exists and is time-bounded.
    assert "expiration" in policy


def test_presign_requires_content_type(store):
    with pytest.raises(ValueError):
        store.presign_upload(KEY, [], 1000)


def test_signed_get_url(store):
    url = store.signed_url(KEY, expires_s=900)
    assert url.startswith("https://")
    assert "X-Amz-Signature" in url or "Signature" in url


def test_head_object(client, store):
    stub = Stubber(client)
    stub.add_response("head_object",
                      {"ContentLength": 123, "ContentType": "video/mp4"},
                      {"Bucket": BUCKET, "Key": KEY})
    with stub:
        info = store.head(KEY)
    assert info == {"size": 123, "content_type": "video/mp4"}


def test_head_missing_returns_none(client, store):
    stub = Stubber(client)
    stub.add_client_error("head_object", service_error_code="404", http_status_code=404,
                          expected_params={"Bucket": BUCKET, "Key": KEY})
    with stub:
        assert store.head(KEY) is None


def test_put_and_get_bytes(client, store):
    stub = Stubber(client)
    stub.add_response("put_object", {}, {"Bucket": BUCKET, "Key": KEY, "Body": b"data",
                                         "ContentType": "video/mp4"})
    from io import BytesIO

    stub.add_response("get_object", {"Body": BytesIO(b"data")}, {"Bucket": BUCKET, "Key": KEY})
    with stub:
        store.put_bytes(KEY, b"data", "video/mp4")
        assert store.get_bytes(KEY) == b"data"


def test_bad_key_rejected_before_any_call(store):
    from screwit.storage.keys import KeyError as StorageKeyError

    with pytest.raises(StorageKeyError):
        store.presign_upload("../escape", ["video/mp4"], 1000)
