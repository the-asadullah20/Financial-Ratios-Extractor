import json
import os
import re
import threading
import logging
from datetime import datetime

from app.config import settings

logger = logging.getLogger("utils")


def sanitize_filename_part(text: str) -> str:
    """Turns a company name into a filesystem-safe filename fragment."""
    if not text:
        text = "unknown_company"
    text = text.strip()
    text = re.sub(r"[^\w\s\-&]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("_") or "unknown_company"


def build_output_filename(company_name: str) -> str:
    """
    Builds the '<company>_<timestamp>.json' filename, e.g.
    'Nike_Inc_20260719_154213.json'
    """
    safe_company = sanitize_filename_part(company_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_company}_{timestamp}.json"


def _async_upload_json(path: str, filename: str):
    if settings.S3_BUCKET_NAME:
        try:
            from app.storage.raw_store import raw_store
            s3_client = raw_store._get_s3_client()
            if s3_client:
                s3_key = f"output/{filename}"
                s3_client.upload_file(path, settings.S3_BUCKET_NAME, s3_key)
                logger.info("Uploaded output JSON to S3/R2 Bucket s3://%s/%s", settings.S3_BUCKET_NAME, s3_key)
        except Exception as exc:
            logger.warning("Failed uploading JSON output to Cloud S3/R2: %s", exc)


def save_json_output(data: dict, company_name: str) -> str:
    """Saves extracted JSON locally and syncs to Cloud S3/R2 Bucket in background thread."""
    filename = build_output_filename(company_name)
    path = os.path.join(settings.OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Async background upload to Cloud S3 / Cloudflare R2
    if settings.S3_BUCKET_NAME:
        threading.Thread(target=_async_upload_json, args=(path, filename), daemon=True).start()

    return path


def extract_json_object(text: str) -> dict:
    """
    Strips markdown fences or stray commentary and extracts valid JSON object from LLM response.
    """
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Could not locate a valid JSON object in model output.")
