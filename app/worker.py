"""
RabbitMQ Worker process.
Listens to RabbitMQ queue, consumes tasks, executes the 5-step extraction pipeline, and uploads results.
Run command: python -m app.worker
"""
import json
import logging
import os
import sys
import time

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pika
from app.config import settings
from app.ocr_service import ocr_pdf
from app.storage.raw_store import raw_store
from app.storage.vector_store import vector_store
from app.agent.financial_agent import run_financial_agent
from app.structuring import structure_financial_ratios
from app.utils import save_json_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("worker")

# Thread-safe in-memory status cache for worker instances
job_status_cache = {}


def update_status(document_id: str, status: str, result: dict = None, error: str = None):
    """Updates task status in a local shared JSON file for main server polling."""
    # Write to a status file in output directory
    status_file = os.path.join(settings.OUTPUT_DIR, f"status_{document_id}.json")
    status_data = {
        "document_id": document_id,
        "status": status,
        "updated_at": time.time(),
    }
    if result:
        status_data["data"] = result
    if error:
        status_data["error"] = error
        
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
    except Exception as exc:
        logger.error("Failed writing status file for %s: %s", document_id, exc)


def process_task(ch, method, properties, body):
    """Processes task message received from RabbitMQ."""
    t0 = time.time()
    try:
        task_data = json.loads(body.decode("utf-8"))
        document_id = task_data.get("document_id")
        file_path = task_data.get("file_path")
        
        logger.info("[%s] Received task for processing: %s", document_id, file_path)
        update_status(document_id, "processing")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at local temp path: {file_path}")

        # Read PDF bytes
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # Step 1: Hybrid PDF Parser & Vision OCR
        logger.info("[%s] Step 1/5: Running Hybrid OCR Engine...", document_id)
        markdown_text = ocr_pdf(pdf_bytes)

        # Step 2: Storage & Qdrant Indexing
        logger.info("[%s] Step 2/5: Saving Markdown & Indexing vectors in Qdrant Cloud...", document_id)
        raw_store.save_markdown(document_id, markdown_text)
        chunk_count = vector_store.index_document(document_id, markdown_text)

        # Step 3: LangGraph AI Agent reasoning
        logger.info("[%s] Step 3/5: Executing LangGraph AI Agent tools...", document_id)
        agent_result = run_financial_agent(document_id, markdown_text)

        # Step 4: Normalizing Ratios & JSON Structuring
        logger.info("[%s] Step 4/5: Structuring financial schema...", document_id)
        extracted_stmt = agent_result.get("extracted_statement") or {}
        computed_ratios = agent_result.get("computed_ratios") or {}
        unstructured_narrative = agent_result.get("unstructured_output") or ""
        structured = structure_financial_ratios(extracted_stmt, computed_ratios, unstructured_narrative)

        # Step 5: Save JSON Output (triggers S3/GCS sync)
        logger.info("[%s] Step 5/5: Exporting JSON output to S3/GCS Cloud Bucket...", document_id)
        company_name = structured.get("company_name") or "unknown_company"
        saved_path = save_json_output(structured, company_name)

        elapsed = round(time.time() - t0, 2)
        logger.info("[%s] Pipeline completed in %ss. Result saved to: %s", document_id, elapsed, saved_path)

        # Update final task status
        update_status(document_id, "completed", result=structured)

        # Clean up temp upload file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as exc:
        logger.exception("Error occurred during task processing: %s", exc)
        try:
            update_status(document_id, "failed", error=str(exc))
        except Exception:
            pass
    finally:
        # Acknowledge message has been processed successfully
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    if not settings.RABBITMQ_URL:
        logger.critical("RABBITMQ_URL is not set in environment settings. Worker exiting.")
        sys.exit(1)

    logger.info("Initializing RabbitMQ Worker...")
    try:
        parameters = pika.URLParameters(settings.RABBITMQ_URL)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare durable queue matching producer
        channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True)
        
        # Configure fair dispatch (1 message at a time per worker)
        channel.basic_qos(prefetch_count=1)
        
        channel.basic_consume(
            queue=settings.RABBITMQ_QUEUE,
            on_message_callback=process_task
        )

        logger.info("Worker is online! Listening for PDF tasks on queue: %r", settings.RABBITMQ_QUEUE)
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Worker shutting down gracefully.")
    except Exception as exc:
        logger.critical("Fatal worker error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
