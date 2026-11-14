"""Private-S3 object store for deployment (SPEC 3.3).

Uploads use a pre-signed POST locking the exact key, a Content-Type condition and a
content-length-range so S3 itself enforces the size cap; assets are read through short-lived signed
GET URLs. boto3 is imported lazily so dev and tests that never touch S3 do not need credentials.
"""

from __future__ import annotations

from pathlib import Path

from screwit.contracts import PresignedPost
from screwit.storage.keys import validate_key

_UPLOAD_EXPIRY_S = 600  # 10 minutes (SPEC 3.3)


class S3Storage:
    """Storage protocol backed by a private S3 bucket."""

    def __init__(self, bucket: str, region: str | None = None, client=None) -> None:
        self.bucket = bucket
        self.region = region
        if client is not None:
            self._client = client
        else:
            import boto3  # lazy: only when a real S3 store is constructed
            from botocore.config import Config

            self._client = boto3.client(
                "s3", region_name=region, config=Config(signature_version="s3v4")
            )

    def presign_upload(self, key: str, content_types: list[str], max_bytes: int,
                       expires_s: int = _UPLOAD_EXPIRY_S) -> PresignedPost:
        """Pre-signed POST with the exact key, a Content-Type condition and content-length-range.

        A single content type is pinned as a POST field + condition; if several are allowed, the
        first is pinned (S3 POST policy matches one Content-Type per form). The size cap uses
        ["content-length-range", 1, max_bytes] so S3 rejects an over-limit body itself (SPEC 3.3).
        """
        validate_key(key)
        if not content_types:
            raise ValueError("at least one content type is required")
        ct = content_types[0]
        conditions: list = [
            {"key": key},
            {"Content-Type": ct},
            ["content-length-range", 1, int(max_bytes)],
        ]
        resp = self._client.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Fields={"Content-Type": ct},
            Conditions=conditions,
            ExpiresIn=int(expires_s),
        )
        return PresignedPost(url=resp["url"], fields=dict(resp["fields"]), key=key)

    def signed_url(self, key: str, expires_s: int = 900) -> str:
        validate_key(key)
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=int(expires_s)
        )

    def head(self, key: str) -> dict | None:
        validate_key(key)
        from botocore.exceptions import ClientError

        try:
            r = self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return {"size": int(r["ContentLength"]),
                "content_type": r.get("ContentType", "application/octet-stream")}

    def get_bytes(self, key: str) -> bytes:
        validate_key(key)
        r = self._client.get_object(Bucket=self.bucket, Key=key)
        return r["Body"].read()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        validate_key(key)
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        validate_key(key)
        self._client.upload_file(str(path), self.bucket, key,
                                 ExtraArgs={"ContentType": content_type})

    def local_path(self, key: str, scratch: Path) -> Path:
        validate_key(key)
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        dest = scratch / Path(key).name
        self._client.download_file(self.bucket, key, str(dest))
        return dest
