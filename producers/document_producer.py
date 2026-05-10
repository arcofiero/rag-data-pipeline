"""
Document Producer — Day 2

Reads documents from heterogeneous sources (PDF files, web crawl output, structured records)
and publishes validated events to the Kafka `raw-documents` topic using the Confluent Kafka
producer. Invalid events are routed to the `raw-documents-dlq` Dead Letter Queue with the
original payload and a structured failure reason. Messages are serialized with Avro against
the schema in schemas/document_event.avsc via the Confluent Schema Registry.
"""

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


if __name__ == "__main__":
    logger.info("document_producer: Not implemented yet — Day 2")
