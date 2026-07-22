"""
RabbitMQ Task Queue Publisher.
Publishes PDF processing jobs to RabbitMQ if configured.
"""
import json
import logging
from typing import Any, Dict
from app.config import settings

logger = logging.getLogger("queue_publisher")


class QueuePublisher:
    def __init__(self):
        self.url = settings.RABBITMQ_URL
        self.queue = settings.RABBITMQ_QUEUE

    def publish_task(self, task_data: Dict[str, Any]) -> bool:
        """Publishes task payload to RabbitMQ queue. Returns True if successful, False if no RabbitMQ configured or connection failed."""
        if not self.url:
            logger.info("RabbitMQ URL not configured. Task queue bypassed (falling back to synchronous processing).")
            return False

        try:
            import pika
            
            # Connect to RabbitMQ using connection URL parameters
            parameters = pika.URLParameters(self.url)
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()

            # Declare durable queue so messages are persistent
            channel.queue_declare(queue=self.queue, durable=True)

            # Publish message with persistent properties
            message = json.dumps(task_data)
            channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                    content_type="application/json",
                )
            )
            logger.info("Successfully published task to RabbitMQ: %s", task_data.get("document_id"))
            connection.close()
            return True
        except Exception as exc:
            logger.warning("Failed publishing task to RabbitMQ (%s). Falling back to synchronous processing.", exc)
            return False


queue_publisher = QueuePublisher()
