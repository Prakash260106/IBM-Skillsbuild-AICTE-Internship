"""
utils/cos_client.py
--------------------
IBM Cloud Object Storage (COS) client wrapper.
Handles all bucket read/write operations with retry logic and
structured error reporting.  Uses ibm-boto3 under the hood.
"""

import json
import io
import logging
from typing import Any, Optional
from functools import lru_cache

import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Client factory (singleton per process via lru_cache)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cos_client(endpoint: str, api_key: str, instance_crn: str):
    """
    Build and cache a single ibm_boto3 S3 client for the lifetime of the
    Streamlit worker process.
    """
    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=api_key,
        ibm_service_instance_id=instance_crn,
        config=Config(signature_version="oauth"),
        endpoint_url=endpoint,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retry decorator for transient network / throttle errors
# ──────────────────────────────────────────────────────────────────────────────

_cos_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(ClientError),
    reraise=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# High-level helpers
# ──────────────────────────────────────────────────────────────────────────────

class COSClient:
    """
    Thin convenience wrapper around ibm_boto3 that provides:
      - put_json / get_json  for metadata objects
      - put_text / get_text  for raw assignment bodies
      - put_bytes / get_bytes for binary blobs (PDF uploads)
      - list_prefix          for directory-style key listing
      - object_exists        for safe conditional checks
    All public methods are decorated with retry logic.
    """

    def __init__(self, endpoint: str, api_key: str, instance_crn: str, bucket: str):
        self._client = get_cos_client(endpoint, api_key, instance_crn)
        self.bucket = bucket
        self._ensure_bucket()

    # ── bucket bootstrap ─────────────────────────────────────────────────────

    def _ensure_bucket(self):
        """Create bucket if it does not already exist (idempotent)."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self.bucket)
                logger.info("Created COS bucket: %s", self.bucket)
            else:
                raise

    # ── JSON objects ─────────────────────────────────────────────────────────

    @_cos_retry
    def put_json(self, key: str, data: Any) -> None:
        """Serialise *data* to JSON and upload it under *key*."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self._client.put_object(Bucket=self.bucket, Key=key, Body=body,
                                ContentType="application/json")

    @_cos_retry
    def get_json(self, key: str) -> Optional[Any]:
        """Download and deserialise JSON from *key*; returns None if missing."""
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise

    # ── Plain-text objects ────────────────────────────────────────────────────

    @_cos_retry
    def put_text(self, key: str, text: str) -> None:
        """Upload UTF-8 text under *key*."""
        self._client.put_object(Bucket=self.bucket, Key=key,
                                Body=text.encode("utf-8"),
                                ContentType="text/plain; charset=utf-8")

    @_cos_retry
    def get_text(self, key: str) -> Optional[str]:
        """Download UTF-8 text from *key*; returns None if missing."""
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode("utf-8")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise

    # ── Binary blobs ──────────────────────────────────────────────────────────

    @_cos_retry
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Upload raw bytes under *key*."""
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data,
                                ContentType=content_type)

    @_cos_retry
    def get_bytes(self, key: str) -> Optional[bytes]:
        """Download raw bytes from *key*; returns None if missing."""
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise

    # ── Directory-style listing ───────────────────────────────────────────────

    @_cos_retry
    def list_prefix(self, prefix: str) -> list[str]:
        """
        Return all object keys that start with *prefix*.
        Handles COS pagination transparently.
        """
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    @_cos_retry
    def object_exists(self, key: str) -> bool:
        """Return True if *key* exists in the bucket."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    @_cos_retry
    def delete_object(self, key: str) -> None:
        """Remove *key* from the bucket (no-op if already absent)."""
        self._client.delete_object(Bucket=self.bucket, Key=key)
