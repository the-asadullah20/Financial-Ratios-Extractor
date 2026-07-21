"""
Raw Markdown Storage module.
Stores raw OCR Markdown files locally and syncs asynchronously with Cloud Bucket Storage (AWS S3 / Cloudflare R2 / GCS).
"""
import os
import io
import threading
import logging
from typing import Optional
from app.config import settings

logger = logging.getLogger("raw_store")


class RawMarkdownStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.RAW_MARKDOWN_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_s3_client(self):
        """Creates boto3 S3 client supporting AWS S3, Cloudflare R2, and GCS S3-compatibility."""
        if not settings.S3_BUCKET_NAME:
            return None
        try:
            import boto3
            from botocore.client import Config

            kwargs = {}
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
                kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
            
            # GCS Interoperability configuration requirements (path-style addressing & us-east-1 region)
            if settings.S3_ENDPOINT_URL and "googleapis.com" in settings.S3_ENDPOINT_URL:
                kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
                kwargs["region_name"] = "us-east-1"
                kwargs["config"] = Config(signature_version="s3v4", s3={"addressing_style": "path"})
            else:
                if settings.AWS_REGION:
                    kwargs["region_name"] = settings.AWS_REGION
                if settings.S3_ENDPOINT_URL:
                    kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
                    kwargs["config"] = Config(s3={"addressing_style": "path"})

            return boto3.client("s3", **kwargs)
        except Exception as exc:
            logger.warning("Could not initialize S3 client: %s", exc)
            return None

    def _async_upload_s3(self, file_path: str, document_id: str):
        s3_client = self._get_s3_client()
        if s3_client and settings.S3_BUCKET_NAME:
            try:
                s3_key = f"raw_markdown/{document_id}.md"
                s3_client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
                logger.info("Uploaded %s to GCS/S3 Cloud Bucket s3://%s/%s", document_id, settings.S3_BUCKET_NAME, s3_key)
            except Exception as exc:
                logger.warning("Failed uploading markdown to GCS/S3: %s", exc)

    def save_markdown(self, document_id: str, markdown_content: str) -> str:
        """Saves raw markdown text locally and uploads to GCS/S3 Bucket in background thread."""
        file_path = os.path.join(self.base_dir, f"{document_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        logger.info("Saved raw markdown locally for doc_id=%s to %s", document_id, file_path)

        # Async background sync to GCS / S3 Bucket
        if settings.S3_BUCKET_NAME:
            threading.Thread(target=self._async_upload_s3, args=(file_path, document_id), daemon=True).start()

        return file_path

    def get_markdown(self, document_id: str) -> Optional[str]:
        """Retrieves raw markdown content from local disk or streams from GCS/S3."""
        file_path = os.path.join(self.base_dir, f"{document_id}.md")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        s3_client = self._get_s3_client()
        if s3_client and settings.S3_BUCKET_NAME:
            try:
                s3_key = f"raw_markdown/{document_id}.md"
                response = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
                content = response["Body"].read().decode("utf-8")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return content
            except Exception as exc:
                logger.warning("Failed streaming markdown from GCS/S3: %s", exc)

        return None


raw_store = RawMarkdownStore()
