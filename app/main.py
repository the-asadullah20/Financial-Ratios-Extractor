"""
FastAPI Server - Financial Ratios Extractor Architecture Pipeline
Integrates: PDF Parsing -> Qdrant Vector Storage -> LangGraph AI Agent (Grep/Semantic/Math tools) -> LLM Structuring -> Evaluation Engine.
Supports Asynchronous RabbitMQ Queueing with Synchronous Fallback.
"""
import logging
import os
import json
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import settings
from app.ocr_service import ocr_pdf
from app.storage.raw_store import raw_store
from app.storage.vector_store import vector_store
from app.storage.queue_publisher import queue_publisher
from app.ratio_config import get_active_ratio_configs
from app.agent.financial_agent import run_financial_agent
from app.structuring import structure_financial_ratios
from app.eval.evaluator import run_evaluation_benchmark
from app.utils import save_json_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="Financial Ratios Extractor & Evaluation Engine",
    description=(
        "Production multi-stage agentic pipeline for financial document extraction.\n"
        "1. Parallel PDF & Vision OCR Engine: PyMuPDF multithreaded rendering with Gemini 2.5 Flash Vision OCR.\n"
        "2. Real-Time Cloud Storage & Vector DB: Asynchronous GCS Bucket sync & Qdrant Cloud Vector DB chunk indexing.\n"
        "3. Tooling Layer: Mirage keyword search, vector similarity search, and exact Python financial math calculator.\n"
        "4. LangGraph AI Agent: ReAct deep agent graph orchestrating tool execution and line-item extraction across all pages.\n"
        "5. LLM Structuring & Evaluation Engine: Canonical JSON schema formatting & automated ground-truth accuracy benchmarking."
    ),
    version="2.2.0",
)


@app.on_event("startup")
def startup_event():
    if settings.RABBITMQ_URL:
        import threading
        from app.worker import main as start_worker
        
        def run_worker_thread():
            try:
                start_worker()
            except Exception as exc:
                logger.error("RabbitMQ background worker thread failed to run: %s", exc)

        logger.info("Spawning background RabbitMQ task consumer thread...")
        threading.Thread(target=run_worker_thread, daemon=True).start()


@app.get("/")
def redirect_to_docs():
    """Redirects root to interactive OpenAPI Swagger documentation."""
    return RedirectResponse(url="/docs")


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
        "rabbitmq_queue_active": bool(settings.RABBITMQ_URL),
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


@app.get("/status/{document_id}")
def get_task_status(document_id: str):
    """
    Poll task status for asynchronous RabbitMQ background tasks.
    Returns status and extracted financial ratio data if completed.
    """
    status_file = os.path.join(settings.OUTPUT_DIR, f"status_{document_id}.json")
    if not os.path.exists(status_file):
        raise HTTPException(
            status_code=404, 
            detail=f"Task with ID {document_id} not found or processed synchronously."
        )
    
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed loading task status: {exc}")


@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """
    Processes a financial PDF using the 5-step Agentic Pipeline.
    If RabbitMQ is configured, task is queued and returns HTTP 202 Accepted.
    Otherwise, falls back to synchronous execution (HTTP 200 OK).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    document_id = uuid.uuid4().hex[:8]
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save PDF locally to UPLOAD_DIR for queue processing
    temp_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}.pdf")
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    # Attempt publishing to RabbitMQ Task Queue
    task_data = {"document_id": document_id, "file_path": temp_path}
    if queue_publisher.publish_task(task_data):
        # Create initial status file
        status_file = os.path.join(settings.OUTPUT_DIR, f"status_{document_id}.json")
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "document_id": document_id, 
                "status": "queued", 
                "updated_at": time.time()
            }, f)
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "queued",
                "document_id": document_id,
                "message": "Task queued in RabbitMQ. Poll /status/{document_id} for results.",
                "poll_url": f"/status/{document_id}"
            }
        )

    # --- Synchronous Fallback if RabbitMQ is missing ---
    logger.info("[%s] Bypassing RabbitMQ. Running synchronous extraction pipeline...", document_id)
    t0 = time.time()
    
    try:
        # 1. PDF Parser -> Markdown
        logger.info("[%s] Step 1: Parsing PDF & running OCR for %s", document_id, file.filename)
        markdown_text = ocr_pdf(pdf_bytes)
        
        if not markdown_text.strip():
            raise ValueError("PDF parser returned empty text.")

        # 2. Storage & Qdrant Vector DB Indexing
        logger.info("[%s] Step 2: Saving Markdown & indexing in Qdrant Vector DB", document_id)
        raw_store.save_markdown(document_id, markdown_text)
        chunk_count = vector_store.index_document(document_id, markdown_text)

        # 3. LangGraph AI Agent Execution
        logger.info("[%s] Step 3: Running LangGraph AI Agent with Tooling", document_id)
        agent_result = run_financial_agent(document_id, markdown_text)

        # 4. LLM Structuring & Ratio Normalization
        logger.info("[%s] Step 4: Formatting structured financial output", document_id)
        extracted_stmt = agent_result.get("extracted_statement") or {}
        computed_ratios = agent_result.get("computed_ratios") or {}
        unstructured_narrative = agent_result.get("unstructured_output") or ""
        structured = structure_financial_ratios(extracted_stmt, computed_ratios, unstructured_narrative)

        # 5. Save JSON Output to Disk & Cloud Storage
        company_name = structured.get("company_name") or "unknown_company"
        saved_path = save_json_output(structured, company_name)

        elapsed = round(time.time() - t0, 2)
        logger.info("[%s] End-to-end pipeline completed in %ss -> %s", document_id, elapsed, saved_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return JSONResponse(
            content={
                "document_id": document_id,
                "elapsed_seconds": elapsed,
                "qdrant_chunks_indexed": chunk_count,
                "saved_path": saved_path,
                "data": structured,
            }
        )
    except Exception as exc:
        logger.exception("[%s] Extraction failed", document_id)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate_benchmark():
    """Runs automated benchmark evaluations across ground-truth evaluation datasets."""
    logger.info("Running evaluation benchmark engine...")
    results = run_evaluation_benchmark()
    return JSONResponse(content=results)
