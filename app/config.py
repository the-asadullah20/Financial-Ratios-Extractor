"""
Central configuration, loaded from environment variables (.env file).
Supports Cloud Services (AWS S3 / Cloudflare R2, Qdrant Cloud, Gemini API).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- OCR backend selection ---
    # "gemini" -> Gemini 2.5 Flash Vision OCR (fast, reliable, multimodal)
    # "vllm"   -> OpenAI-compatible chat-completions endpoint
    # "hf"     -> legacy HF Inference Providers path
    OCR_BACKEND: str = os.getenv("OCR_BACKEND", "gemini")

    # --- vLLM / Modal (OpenAI-compatible) backend ---
    VLLM_API_BASE: str = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "chandra")

    # --- Hugging Face ---
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    CHANDRA_MODEL_ID: str = os.getenv("CHANDRA_MODEL_ID", "datalab-to/chandra")
    HF_PROVIDER: str = os.getenv("HF_PROVIDER", "")

    # --- Gemini API ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # --- OCR / processing behaviour ---
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "35"))

    # --- Local Storage Paths ---
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads_tmp")
    RAW_MARKDOWN_DIR: str = os.getenv("RAW_MARKDOWN_DIR", "raw_markdown")
    QDRANT_STORAGE_PATH: str = os.getenv("QDRANT_STORAGE_PATH", "qdrant_db")
    EVAL_DATA_DIR: str = os.getenv("EVAL_DATA_DIR", "eval_data")

    # --- Qdrant Cloud Cluster ---
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # --- Cloud Object Storage (AWS S3 / Cloudflare R2) ---
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "")  # Optional: for R2 / MinIO / Custom S3


settings = Settings()

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RAW_MARKDOWN_DIR, exist_ok=True)
os.makedirs(settings.QDRANT_STORAGE_PATH, exist_ok=True)
os.makedirs(settings.EVAL_DATA_DIR, exist_ok=True)
