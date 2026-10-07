"""Settings for uploads, object storage and the background worker (Phase 3).

Kept apart from app/core/config.py on purpose: nothing here is needed by the
agent/RAG side, and the values are read fresh on every call (no cache), so a
test or an operator changing the environment never sees a stale object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.config import PROJECT_ROOT, _env, _env_bool

CSV_CONTENT_TYPES = frozenset({"text/csv", "application/csv", "text/plain",
                               "application/vnd.ms-excel"})


def _int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class IngestSettings:
    # --- object storage ---------------------------------------------------
    storage_backend: str          # "local" (default) or "s3" (MinIO / any S3-compatible)
    storage_local_dir: Path
    s3_endpoint_url: str | None   # e.g. http://minio:9000 ; None = AWS
    s3_bucket: str
    s3_region: str
    s3_access_key: str | None
    s3_secret_key: str | None
    # --- uploads ----------------------------------------------------------
    upload_max_bytes: int
    upload_rate_limit_per_minute: int
    preview_max_rows: int
    mapping_dry_run_rows: int
    # --- worker -----------------------------------------------------------
    worker_database_url: str | None
    job_max_retries: int
    job_backoff_base_seconds: float
    job_stale_seconds: int
    worker_poll_seconds: float


def ingest_settings() -> IngestSettings:
    backend = (_env("STORAGE_BACKEND", "local") or "local").lower()
    if backend not in ("local", "s3"):
        raise ValueError(f"STORAGE_BACKEND must be 'local' or 's3', got {backend!r}")
    local_dir = Path(_env("STORAGE_LOCAL_DIR", str(PROJECT_ROOT / "data" / "storage")))
    return IngestSettings(
        storage_backend=backend,
        storage_local_dir=local_dir,
        s3_endpoint_url=_env("S3_ENDPOINT_URL"),
        s3_bucket=_env("S3_BUCKET", "iia-uploads") or "iia-uploads",
        s3_region=_env("S3_REGION", "us-east-1") or "us-east-1",
        s3_access_key=_env("S3_ACCESS_KEY_ID"),
        s3_secret_key=_env("S3_SECRET_ACCESS_KEY"),
        upload_max_bytes=_int("UPLOAD_MAX_BYTES", 50 * 1024 * 1024),
        upload_rate_limit_per_minute=_int("UPLOAD_RATE_LIMIT_PER_MINUTE", 20),
        preview_max_rows=_int("PREVIEW_MAX_ROWS", 50),
        mapping_dry_run_rows=_int("MAPPING_DRY_RUN_ROWS", 5000),
        worker_database_url=_env("WORKER_DATABASE_URL"),
        job_max_retries=_int("JOB_MAX_RETRIES", 3),
        job_backoff_base_seconds=float(_env("JOB_BACKOFF_BASE_SECONDS", "5") or 5),
        job_stale_seconds=_int("JOB_STALE_SECONDS", 300),
        worker_poll_seconds=float(_env("WORKER_POLL_SECONDS", "2") or 2),
    )
