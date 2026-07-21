"""
FastAPI Server - Financial Ratios Extractor Architecture Pipeline
Integrates: PDF Parsing -> Qdrant Vector Storage -> LangGraph AI Agent (Grep/Semantic/Math tools) -> LLM Structuring -> Evaluation Engine.
"""
import logging
import os
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.ocr_service import ocr_pdf
from app.storage.raw_store import raw_store
from app.storage.vector_store import vector_store
from app.ratio_config import get_active_ratio_configs
from app.agent.financial_agent import run_financial_agent
from app.structuring import structure_financial_ratios
from app.eval.evaluator import run_evaluation_benchmark, evaluate_extracted_ratios
from app.utils import save_json_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="Financial Ratios Extractor & Evaluation Engine",
    description=(
        "Production multi-stage agentic pipeline for financial document extraction. "
        "Converts PDF documents to Markdown, indexes semantic chunks in Qdrant Vector DB, "
        "runs a LangGraph ReAct agent with Grep, Retrieval, and Math tools to compute ratios, "
        "and formats canonical structured financial ratio outputs."
    ),
    version="2.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ocr_backend": settings.OCR_BACKEND,
        "gemini_model": settings.GEMINI_MODEL,
        "max_pages": settings.MAX_PAGES,
        "qdrant_vector_store": bool(vector_store.client),
        "qdrant_cloud_cluster": bool(settings.QDRANT_URL),
        "s3_cloud_bucket": settings.S3_BUCKET_NAME or "Not Configured (Local Fallback)",
        "raw_markdown_dir": os.path.abspath(settings.RAW_MARKDOWN_DIR),
        "output_dir": os.path.abspath(settings.OUTPUT_DIR),
    }


@app.get("/ratios/config")
def get_ratios_config():
    """Returns the list of active financial ratio configurations and formulas."""
    return {
        "status": "success",
        "ratios": get_active_ratio_configs()
    }


@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """
    Full end-to-end Agentic Pipeline:
    1. Parses PDF document into raw Markdown (Chandra/Gemini OCR).
    2. Saves raw Markdown and indexes semantic chunks into Qdrant Vector DB.
    3. Runs LangGraph AI Agent using Mirage Grep, Semantic Retrieval, and Math tools.
    4. Computes exact ratios using deterministic Python math tools.
    5. Formats structured JSON output and saves to disk.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    document_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. PDF Parser -> Markdown
    try:
        logger.info("[%s] Step 1: Parsing PDF & running OCR for %s", document_id, file.filename)
        markdown_text = ocr_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] OCR parsing failed", document_id)
        raise HTTPException(status_code=500, detail=f"PDF OCR failed: {exc}") from exc

    if not markdown_text.strip():
        raise HTTPException(status_code=500, detail="PDF parser returned empty text.")

    # 2. Storage & Qdrant Vector DB Indexing
    logger.info("[%s] Step 2: Saving Markdown & indexing in Qdrant Vector DB", document_id)
    raw_store.save_markdown(document_id, markdown_text)
    chunk_count = vector_store.index_document(document_id, markdown_text)

    # 3. LangGraph AI Agent Execution (Grep, Semantic Retrieval, Math tools)
    try:
        logger.info("[%s] Step 3: Running LangGraph AI Agent with Tooling", document_id)
        agent_result = run_financial_agent(document_id, markdown_text)
    except Exception as exc:
        logger.exception("[%s] LangGraph Agent execution failed", document_id)
        raise HTTPException(status_code=500, detail=f"AI Agent execution error: {exc}") from exc

    # 4. LLM Structuring & Ratio Normalization
    logger.info("[%s] Step 4: Formatting structured financial output", document_id)
    extracted_stmt = agent_result.get("extracted_statement") or {}
    computed_ratios = agent_result.get("computed_ratios") or {}
    unstructured_narrative = agent_result.get("unstructured_output") or ""

    structured = structure_financial_ratios(extracted_stmt, computed_ratios, unstructured_narrative)

    # 5. Save JSON Output to Disk
    company_name = structured.get("company_name") or "unknown_company"
    saved_path = save_json_output(structured, company_name)

    elapsed = round(time.time() - t0, 2)
    logger.info("[%s] End-to-end pipeline completed in %ss -> %s", document_id, elapsed, saved_path)

    return JSONResponse(
        content={
            "document_id": document_id,
            "elapsed_seconds": elapsed,
            "qdrant_chunks_indexed": chunk_count,
            "saved_path": saved_path,
            "data": structured,
        }
    )


@app.post("/evaluate")
def evaluate_benchmark():
    """Runs automated benchmark evaluations across ground-truth evaluation datasets."""
    logger.info("Running evaluation benchmark engine...")
    results = run_evaluation_benchmark()
    return JSONResponse(content=results)
