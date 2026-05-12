"""
Document Producer — simulates ingestion from three source types: PDFs, web pages,
and structured records. Approximately 5% of events are intentionally malformed
and routed to the dead-letter queue (DLQ) topic.

Design decisions:
- confluent-kafka (not kafka-python) is used for native Confluent Cloud compatibility:
  SASL/SSL auth, delivery callbacks, and production-grade librdkafka internals.
- Avro serialization uses fastavro against schemas/document_event.avsc and
  schemas/dead_letter_event.avsc. Schemas are loaded once at startup and reused
  across all messages to avoid repeated disk reads.
- Malformed events are serialized with the DeadLetterEvent schema (not dropped raw)
  so a DLQ consumer can deserialize cleanly without risking a deserialization error
  on the error path itself.
- delivery_report is registered as a per-message callback so failures surface
  asynchronously without blocking the produce loop. poll(0) is called after each
  produce to trigger pending callbacks promptly.
- linger.ms=5 allows the producer to batch messages over 5ms, improving throughput
  on the simulation workload without adding meaningful latency.
- acks=all + retries=5 ensures durability: a message must be acknowledged by all
  in-sync replicas before the producer considers it delivered.
"""

from __future__ import annotations

import io
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fastavro
from confluent_kafka import Producer, KafkaException
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = os.getenv("CONFLUENT_BOOTSTRAP_SERVERS", "")
KAFKA_API_KEY = os.getenv("CONFLUENT_API_KEY", "")
KAFKA_API_SECRET = os.getenv("CONFLUENT_API_SECRET", "")

RAW_DOCUMENTS_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw-documents")
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "raw-documents-dlq")
MALFORMED_EVENT_RATE = float(os.getenv("MALFORMED_EVENT_RATE", "0.05"))

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


# ─── Schema loading ─────────────────────────────────────────────────────────────


def _load_avro_schema(schema_name: str) -> dict:
    """Load and parse an Avro schema file from the schemas directory."""
    path = SCHEMA_DIR / f"{schema_name}.avsc"
    if not path.exists():
        raise FileNotFoundError(f"Avro schema not found: {path}")
    with path.open() as f:
        return fastavro.parse_schema(json.load(f))


# ─── Kafka producer factory ─────────────────────────────────────────────────────


def _build_producer() -> Producer:
    """
    Construct a Confluent Kafka Producer configured for Confluent Cloud.
    SASL_SSL with PLAIN mechanism is the Confluent Cloud standard auth path.
    """
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": KAFKA_API_KEY,
        "sasl.password": KAFKA_API_SECRET,
        "acks": "all",
        "retries": 5,
        "linger.ms": 5,
        "compression.type": "lz4",
    })


# ─── Delivery callback ──────────────────────────────────────────────────────────


def delivery_report(err: Any, msg: Any) -> None:
    """
    Invoked by librdkafka for every produced message.
    Logs success (topic/partition/offset) or failure so operators
    can act without polling a separate error queue.
    """
    if err is not None:
        logger.error(
            "Delivery failed | topic={} partition={} error={}",
            msg.topic(), msg.partition(), err,
        )
    else:
        logger.debug(
            "Delivered | topic={} partition={} offset={} key={}",
            msg.topic(), msg.partition(), msg.offset(),
            msg.key().decode() if msg.key() else None,
        )


# ─── Avro serialization ─────────────────────────────────────────────────────────


def _serialize(record: dict, schema: dict) -> bytes:
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, schema, record)
    return buf.getvalue()


# ─── Timestamp helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_millis() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ─── Valid event generators ─────────────────────────────────────────────────────


def _make_pdf_event(doc_id: str) -> dict:
    return {
        "document_id": doc_id,
        "source_type": "PDF",
        "source_uri": f"s3://rag-raw-docs/pdfs/sample_{random.randint(1, 1000)}.pdf",
        "content": (
            "Synthetic PDF document on machine learning, vector databases, and "
            f"retrieval-augmented generation. Document ID: {doc_id}."
        ),
        "metadata": {
            "author": random.choice(["Smith, J.", "Doe, A.", "Zhang, L."]),
            "page_count": str(random.randint(5, 120)),
            "language": "en",
        },
        "ingested_at": _now_iso(),
        "produced_at": _now_millis(),
        "schema_version": "1.0",
    }


def _make_web_event(doc_id: str) -> dict:
    domains = ["arxiv.org", "huggingface.co", "towardsdatascience.com", "openai.com"]
    return {
        "document_id": doc_id,
        "source_type": "WEB",
        "source_uri": f"https://{random.choice(domains)}/articles/{doc_id[:8]}",
        "content": (
            "Web-scraped article on large language models, fine-tuning strategies, "
            f"and deployment best practices. Crawled at {_now_iso()}."
        ),
        "metadata": {
            "crawled_at": _now_iso(),
            "word_count": str(random.randint(400, 5000)),
            "language": "en",
        },
        "ingested_at": _now_iso(),
        "produced_at": _now_millis(),
        "schema_version": "1.0",
    }


def _make_structured_event(doc_id: str) -> dict:
    return {
        "document_id": doc_id,
        "source_type": "STRUCTURED",
        "source_uri": f"postgres://warehouse/knowledge_base/{random.randint(1, 9999)}",
        "content": (
            f"Structured knowledge-base record. "
            f"Category: {random.choice(['policy', 'faq', 'procedure', 'glossary'])}. "
            f"Record ID: {doc_id}."
        ),
        "metadata": {
            "table": "knowledge_base",
            "row_id": str(random.randint(1, 9999)),
            "updated_at": _now_iso(),
        },
        "ingested_at": _now_iso(),
        "produced_at": _now_millis(),
        "schema_version": "1.0",
    }


_EVENT_GENERATORS = [_make_pdf_event, _make_web_event, _make_structured_event]


# ─── Malformed payload generators ──────────────────────────────────────────────


def _make_malformed_payload() -> str:
    """
    Returns a deliberately broken payload string. Three failure modes are
    rotated to exercise different DLQ error paths downstream:
    - missing_fields: required fields omitted
    - invalid_json: unparseable JSON
    - wrong_type: wrong Python type in a required field
    """
    mode = random.choice(["missing_fields", "invalid_json", "wrong_type"])
    if mode == "missing_fields":
        return json.dumps({
            "document_id": str(uuid.uuid4()),
            # source_type and content intentionally omitted
            "ingested_at": _now_iso(),
        })
    elif mode == "invalid_json":
        return '{"document_id": "abc", "source_type": PDF_no_quotes, }'
    else:
        return json.dumps({
            "document_id": 99999,           # should be string
            "source_type": "PDF",
            "source_uri": None,             # non-nullable
            "content": ["not", "a", "string"],
            "metadata": "not-a-map",
            "ingested_at": _now_iso(),
            "produced_at": _now_millis(),
            "schema_version": "1.0",
        })


def _build_dlq_envelope(raw_payload: str, error_reason: str) -> dict:
    return {
        "failed_at": _now_iso(),
        "source_topic": RAW_DOCUMENTS_TOPIC,
        "error_reason": error_reason,
        "original_payload": raw_payload,
        "dlq_schema_version": "1.0",
    }


# ─── Core produce logic ─────────────────────────────────────────────────────────


def _produce_valid(producer: Producer, doc_schema: dict, doc_id: str) -> None:
    generator = random.choice(_EVENT_GENERATORS)
    record = generator(doc_id)
    payload = _serialize(record, doc_schema)
    producer.produce(
        topic=RAW_DOCUMENTS_TOPIC,
        key=doc_id.encode(),
        value=payload,
        on_delivery=delivery_report,
    )
    logger.info(
        "Produced valid event | topic={} doc_id={} source_type={}",
        RAW_DOCUMENTS_TOPIC, doc_id, record["source_type"],
    )


def _produce_dlq(producer: Producer, dlq_schema: dict, doc_id: str) -> None:
    raw_payload = _make_malformed_payload()
    envelope = _build_dlq_envelope(
        raw_payload=raw_payload,
        error_reason="Schema validation failure: malformed or missing required fields",
    )
    payload = _serialize(envelope, dlq_schema)
    producer.produce(
        topic=DLQ_TOPIC,
        key=doc_id.encode(),
        value=payload,
        on_delivery=delivery_report,
    )
    logger.warning(
        "Routed malformed event to DLQ | topic={} doc_id={}",
        DLQ_TOPIC, doc_id,
    )


# ─── Public interface ───────────────────────────────────────────────────────────


def run_producer(num_events: int = 100, delay_seconds: float = 0.05) -> None:
    """
    Produce `num_events` document events to Kafka. Approximately MALFORMED_EVENT_RATE
    fraction will be intentionally broken and routed to the DLQ topic.

    Args:
        num_events: Total events to produce (valid + malformed combined).
        delay_seconds: Sleep between produces to simulate realistic ingestion rate.
    """
    logger.info(
        "Starting producer | events={} malformed_rate={:.0%} delay={}s",
        num_events, MALFORMED_EVENT_RATE, delay_seconds,
    )

    doc_schema = _load_avro_schema("document_event")
    dlq_schema = _load_avro_schema("dead_letter_event")
    producer = _build_producer()

    valid_count = 0
    dlq_count = 0

    try:
        for _ in range(num_events):
            doc_id = str(uuid.uuid4())
            if random.random() < MALFORMED_EVENT_RATE:
                _produce_dlq(producer, dlq_schema, doc_id)
                dlq_count += 1
            else:
                _produce_valid(producer, doc_schema, doc_id)
                valid_count += 1

            # Trigger delivery callbacks without blocking
            producer.poll(0)

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        logger.info("Flushing producer...")
        undelivered = producer.flush(timeout=30)
        if undelivered > 0:
            logger.error("{} messages were NOT delivered before timeout", undelivered)

    except KafkaException as exc:
        logger.exception("KafkaException during produce loop: {}", exc)
        raise
    finally:
        logger.info(
            "Producer finished | valid={} dlq={} total={}",
            valid_count, dlq_count, valid_count + dlq_count,
        )


if __name__ == "__main__":
    run_producer(num_events=200, delay_seconds=0.02)
