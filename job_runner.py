"""
job_runner.py
Cloud Run Job worker for batch inference.

Usage: container entrypoint should run this script. Configuration via ENV vars:
- INPUT_PATH: b2://bucket/path/to/input.csv
- OUTPUT_PREFIX: outputs/YYYY-MM-DD/job-<id>
- ARTIFACTS_PREFIX: (optional) b2 path where model artifacts live, e.g. artifacts/lead-v1
- B2_S3_ENDPOINT: (optional) S3 endpoint, e.g. https://s3.us-west-001.backblazeb2.com
- B2_KEY_ID, B2_APP_KEY: credentials (mounted via Secret Manager -> env)
- MODEL_VERSION: (optional) human label
- TMP_DIR: local working dir (default /tmp)
- SCHEMA_VERSION: for results.json
"""

import os
import sys
import json
import logging
import tempfile
import uuid
import requests
from pathlib import Path
from datetime import datetime, timezone

import boto3
import botocore
import pandas as pd
from urllib.parse import urlparse

# import your inference module
from service.inference_service import (
    load_artifacts,
    run_bulk_inference,
    validate_dataframe_columns,
    clean_dataframe,
    Artifacts,
    InferenceError,
)

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("job_runner")

# ----------------------
# Helpers - S3 / B2
# ----------------------
def parse_input_path(path: str):
    """
    Parse input path and return a tuple (mode, bucket, key, http_url)
      - mode: "s3" or "http"
      - bucket, key: for s3 mode
      - http_url: for http mode (direct download)
    Supports:
      - b2://bucket/key
      - s3://bucket/key
      - https://bucket.s3.<region>-004.backblazeb2.com/key...
      - any https://... (treated as http mode)
    """
    if not isinstance(path, str):
        raise ValueError("INPUT_PATH must be a string")
    path = path.strip()

    # b2:// or s3://
    if path.startswith("b2://") or path.startswith("s3://"):
        proto, rest = path.split("://", 1)
        bucket, _, key = rest.partition("/")
        if bucket == "" or key == "":
            raise ValueError("Invalid S3/B2 path (expected b2://bucket/key)")
        return ("s3", bucket, key, None)

    # https:// or http://
    parsed = urlparse(path)
    if parsed.scheme in ("http", "https"):
        # try to see if host is like "<bucket>.s3.<region>-004.backblazeb2.com"
        host_parts = parsed.netloc.split(".")
        if len(host_parts) >= 4 and host_parts[1] == "s3":
            bucket = host_parts[0]
            key = parsed.path.lstrip("/")
            if bucket and key:
                # use s3 mode (useful if you prefer boto3 download with credentials)
                return ("s3", bucket, key, None)
        # fallback: use http download
        return ("http", None, None, path)

    raise ValueError("Unsupported INPUT_PATH format. Use b2:// or https:// or s3://")

# ----------------------
# Additional helpers (paste after parse_input_path / download_from_b2)
# ----------------------
def s3_client_from_env():
    """
    Create a boto3 S3 client configured from environment variables:
      - B2_KEY_ID
      - B2_APP_KEY
      - B2_S3_ENDPOINT (optional; must be full https://... endpoint)
    Returns: boto3 S3 client
    """
    key_id = os.environ.get("B2_KEY_ID")
    app_key = os.environ.get("B2_APP_KEY")
    endpoint = os.environ.get("B2_S3_ENDPOINT")  # e.g. https://s3.us-west-004.backblazeb2.com

    if not key_id or not app_key:
        raise RuntimeError("B2_KEY_ID and B2_APP_KEY must be set in env")

    # create session + client
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        endpoint_url=endpoint if endpoint else None,
        config=botocore.client.Config(signature_version="s3v4"),
    )
    return s3


def parse_b2_path(b2_path: str):
    """
    Small convenience wrapper: accept:
      - b2://bucket/key/...
      - s3://bucket/key/...
      - also accept paths returned by parse_input_path (i.e. s3 mode)
    Returns (bucket, key)
    Raises ValueError on invalid format.
    """
    if not isinstance(b2_path, str):
        raise ValueError("b2 path must be a string")
    b2_path = b2_path.strip()
    if b2_path.startswith("b2://") or b2_path.startswith("s3://"):
        _, rest = b2_path.split("://", 1)
        bucket, _, key = rest.partition("/")
        if not bucket:
            raise ValueError("invalid b2 path, missing bucket")
        # key may be empty for a bucket root; return '' in that case
        return bucket, key
    # If it's an https URL that matches Backblaze s3-style host, extract bucket/key
    parsed = urlparse(b2_path)
    if parsed.scheme in ("http", "https"):
        host_parts = parsed.netloc.split(".")
        if len(host_parts) >= 4 and host_parts[1] == "s3":
            bucket = host_parts[0]
            key = parsed.path.lstrip("/")
            return bucket, key
    raise ValueError("Unsupported B2 path format. Use b2://bucket/key or s3://bucket/key or https s3 endpoint URL")

def download_from_b2(s3_client, input_path: str, local_destination: str, timeout: int = 120):
    """
    Download the file pointed by input_path into local_destination.
    Accepts s3:// or b2:// or https:// URLs.
    - s3 mode uses s3_client.download_file(bucket, key, local_destination)
    - http mode uses requests.get
    """
    mode, bucket, key, http_url = parse_input_path(input_path)

    if mode == "s3":
        # ensure bucket/key present
        if not bucket or not key:
            raise RuntimeError("Invalid S3/B2 path parsed")
        # Use boto3 client's download_file (handles credentials from env)
        try:
            s3_client.download_file(bucket, key, local_destination)
            return local_destination
        except Exception as e:
            # if download_file fails, raise informative error
            raise RuntimeError(f"S3 download failed for {bucket}/{key}: {e}")

    elif mode == "http":
        # HTTP(S) GET, stream to file
        try:
            resp = requests.get(http_url, stream=True, timeout=timeout)
            resp.raise_for_status()
            with open(local_destination, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            return local_destination
        except Exception as e:
            raise RuntimeError(f"HTTP download failed for {http_url}: {e}")

    else:
        raise RuntimeError(f"Unsupported mode from parse_input_path: {mode}")

def upload_to_b2_atomic(s3, local_path: str, b2_target: str):
    """
    Upload as temp, then copy to final (copy_object) and delete temp.
    This reduces window of partial file exposure.
    """
    bucket, key = parse_input_path(b2_target)
    tmp_key = f"{key}.tmp-{uuid.uuid4().hex}"
    logger.info(f"Uploading {local_path} to b2://{bucket}/{tmp_key} (tmp)")
    s3.upload_file(local_path, bucket, tmp_key)

    # Copy tmp -> final
    logger.info(f"Copying tmp object to final b2://{bucket}/{key}")
    copy_source = {"Bucket": bucket, "Key": tmp_key}
    s3.copy_object(Bucket=bucket, CopySource=copy_source, Key=key)

    # Delete tmp
    s3.delete_object(Bucket=bucket, Key=tmp_key)
    logger.info("Upload atomic move complete")

# ----------------------
# Main flow
# ----------------------
def main():
    try:
        # ENV config
        INPUT_PATH = os.environ.get("INPUT_PATH")
        OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "outputs/" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        ARTIFACTS_PREFIX = os.environ.get("ARTIFACTS_PREFIX")
        TMP_DIR = os.environ.get("TMP_DIR", "/tmp")
        SCHEMA_VERSION = os.environ.get("SCHEMA_VERSION", "1.0")
        MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

        if not INPUT_PATH:
            raise RuntimeError("INPUT_PATH must be set (b2://bucket/path/to/file.csv)")

        logger.info("Starting batch worker")
        logger.info(f"INPUT_PATH={INPUT_PATH}, OUTPUT_PREFIX={OUTPUT_PREFIX}, ARTIFACTS_PREFIX={ARTIFACTS_PREFIX}")

        s3 = s3_client_from_env()

        # 1) Download artifacts (if ARTIFACTS_PREFIX set) into local ./artifacts
        local_artifacts_dir = Path("/app/artifacts")
        if ARTIFACTS_PREFIX:
            # Expect ARTIFACTS_PREFIX points to a folder with required files
            local_artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_files = [
                "CatBoost_calibrated_model.pkl",
                "preprocessor.pkl",
                "label_encoders.pkl",
                "feature_names.pkl",
                "shap_explainer.pkl",
            ]
            bucket, _ = parse_b2_path(ARTIFACTS_PREFIX)
            # If ARTIFACTS_PREFIX points to a folder like b2://bucket/artifacts/lead-v1
            for fname in artifact_files:
                remote = f"{ARTIFACTS_PREFIX.rstrip('/')}/{fname}"
                local = str(local_artifacts_dir / fname)
                download_from_b2(s3, remote, local)
        else:
            logger.info("No ARTIFACTS_PREFIX provided — expecting artifacts baked (or present) in image")

        # 2) Load artifacts (calls inference_service.load_artifacts)
        artifacts = load_artifacts(str(local_artifacts_dir))

        # 3) Download CSV input to local temp
        tmp_input = os.path.join(TMP_DIR, f"input-{uuid.uuid4().hex}.csv")
        download_from_b2(s3, INPUT_PATH, tmp_input)

        # 4) Read CSV and validate/clean
        df = pd.read_csv(tmp_input)
        total_rows = len(df)
        validation = validate_dataframe_columns(df)
        if validation["missing_columns"]:
            raise RuntimeError(f"Input file missing required columns: {validation['missing_columns']}")

        clean_result = clean_dataframe(df)
        cleaned_df = clean_result["cleaned_df"]
        original_indices = clean_result["original_indices"]
        dropped_indices = clean_result["dropped_indices"]
        invalid_rows = clean_result["invalid_rows"]

        # 5) Run bulk inference
        results = []
        if len(cleaned_df) > 0:
            results = run_bulk_inference(cleaned_df, original_indices, artifacts)

        # 6) Build output JSON (schema)
        output_payload = {
            "schema_version": SCHEMA_VERSION,
            "job_id": OUTPUT_PREFIX.split("/")[-1],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "rows_total": total_rows,
            "rows_processed": len(cleaned_df),
            "rows_failed": len(dropped_indices) + len(invalid_rows),
            "predictions": results,
            "invalid_rows": [
                {"row_index": r["_original_index"] if "_original_index" in r else r["row_index"], "reason": r.get("reason", "")}
                for r in (invalid_rows or [])
            ],
            "stats": {
                "avg_probability": round(sum((r["probability"] for r in results), 0.0) / len(results), 4) if results else None,
                "conversion_high": sum(1 for r in results if r["risk_level"] == "High"),
                "conversion_medium": sum(1 for r in results if r["risk_level"] == "Medium"),
                "conversion_low": sum(1 for r in results if r["risk_level"] == "Low"),
            }
        }

        # 7) Write results locally and upload atomically
        tmp_out = os.path.join(TMP_DIR, f"results-{uuid.uuid4().hex}.json")
        with open(tmp_out, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        # meta
        meta = {
            "job_id": OUTPUT_PREFIX.split("/")[-1],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "rows_total": total_rows,
            "rows_processed": len(cleaned_df),
            "rows_failed": len(dropped_indices) + len(invalid_rows),
            "output_path": f"{OUTPUT_PREFIX.rstrip('/')}/results.json"
        }
        tmp_meta = os.path.join(TMP_DIR, f"meta-{uuid.uuid4().hex}.json")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Upload files
        results_b2 = f"b2://{parse_b2_path(INPUT_PATH)[0]}/{OUTPUT_PREFIX.rstrip('/')}/results.json"
        meta_b2 = f"b2://{parse_b2_path(INPUT_PATH)[0]}/{OUTPUT_PREFIX.rstrip('/')}/results.meta.json"
        done_b2 = f"b2://{parse_b2_path(INPUT_PATH)[0]}/{OUTPUT_PREFIX.rstrip('/')}/results.done"

        upload_to_b2_atomic(s3, tmp_out, results_b2)
        upload_to_b2_atomic(s3, tmp_meta, meta_b2)

        # create done marker (zero-byte)
        bucket, done_key = parse_b2_path(done_b2)
        s3.put_object(Bucket=bucket, Key=done_key, Body=b"")
        logger.info(f"Job finished successfully. Results at {results_b2}")

        # Exit success
        sys.exit(0)

    except InferenceError as ie:
        logger.exception("InferenceError")
        sys.exit(2)
    except Exception as e:
        logger.exception("Unhandled error")
        sys.exit(1)

if __name__ == "__main__":
    main()