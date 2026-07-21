"""
OCR Service module.
Fast hybrid extraction: Uses PyMuPDF native text extraction for digital PDFs (<0.1s),
falling back to Google Gemini 2.5 Flash Vision OCR for scanned pages/documents.
"""
import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple

import fitz  # PyMuPDF
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from app.config import settings

logger = logging.getLogger("ocr_service")

OCR_PROMPT = (
    "Extract all financial text, tables, line items, and numbers from this financial statement page. "
    "Format tables neatly using Markdown syntax with headers. Maintain exact monetary values and line item labels."
)

_RETRYABLE_EXCEPTIONS = (Exception,)


def pdf_to_images(pdf_bytes: bytes, max_pages: int = None) -> List[Image.Image]:
    """Renders each page of the PDF into a PIL image cleanly and rapidly."""
    max_pages = max_pages or settings.MAX_PAGES
    images = []
    dpi = min(settings.OCR_DPI, 150)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        if total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages, which exceeds the maximum allowed limit of {max_pages} pages."
            )
        for i in range(total_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
            if max(img.width, img.height) > 1280:
                img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            images.append(img)

    return images


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _get_gemini_client():
    import google.generativeai as genai

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(model_name=settings.GEMINI_MODEL)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
)
def _ocr_page_gemini(client, image: Image.Image) -> str:
    response = client.generate_content([OCR_PROMPT, image])
    return response.text if response and response.text else ""


def _process_single_page_vision(args: Tuple[int, Image.Image, Any]) -> Tuple[int, str]:
    idx, image, client = args
    logger.info("Running Gemini Vision OCR on page %d", idx)
    try:
        text = _ocr_page_gemini(client, image)
    except Exception as exc:
        logger.warning("Gemini Vision OCR failed on page %d: %s", idx, exc)
        text = f"[OCR FAILED ON PAGE {idx}: {exc}]"
    return idx, text


def ocr_pdf(pdf_bytes: bytes) -> str:
    """
    ULTRA-FAST HYBRID PDF EXTRACTION:
    1. Extracts native text via PyMuPDF (0.05s).
    2. If native text is extracted (non-scanned), returns structured pages immediately.
    3. If pages are scanned images, runs Gemini 2.5 Flash Vision OCR.
    """
    max_pages = settings.MAX_PAGES
    pages_text: List[str] = []
    scanned_pages: List[Tuple[int, Image.Image]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        if total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages, which exceeds the maximum allowed limit of {max_pages} pages."
            )
        
        zoom = 150 / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for i in range(total_pages):
            page_num = i + 1
            page = doc.load_page(i)
            native_text = page.get_text("text").strip()

            # If page contains substantial text (>40 chars), use native text (instant speed)
            if len(native_text) > 40:
                pages_text.append((page_num, f"--- PAGE {page_num} ---\n{native_text}"))
            else:
                # Scanned or image-heavy page: queue for Vision OCR
                pix = page.get_pixmap(matrix=matrix)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                if max(img.width, img.height) > 1280:
                    img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                scanned_pages.append((page_num, img))

    # If all or some pages required Vision OCR
    if scanned_pages:
        logger.info("Running Vision OCR for %d scanned pages...", len(scanned_pages))
        client = _get_gemini_client()
        tasks = [(page_num, img, client) for page_num, img in scanned_pages]
        
        # Max 2 workers to keep memory well under Render 512MB limit
        with ThreadPoolExecutor(max_workers=min(2, len(scanned_pages))) as executor:
            futures = [executor.submit(_process_single_page_vision, task) for task in tasks]
            for future in as_completed(futures):
                page_num, text = future.result()
                pages_text.append((page_num, f"--- PAGE {page_num} ---\n{text}"))

    # Sort pages in original order
    pages_text.sort(key=lambda x: x[0])
    combined_markdown = "\n\n".join([text for _, text in pages_text])
    logger.info("PDF OCR completed successfully (%d pages processed)", len(pages_text))
    return combined_markdown
