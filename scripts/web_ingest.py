"""
Web Content Ingestion Script

Fetches real Wikipedia articles and produces them as WEB source events
into the Kafka raw-documents topic. The full pipeline then processes them:
  Kafka → Spark Streaming → Bronze → Silver → Embedding → Pinecone

Usage:
    python scripts/web_ingest.py

Articles are chosen to cover the project's own tech stack so you can ask
meaningful questions like "What is RAG?", "How does Kafka work?", etc.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fastavro
import requests
from confluent_kafka import KafkaException, Producer
from dotenv import load_dotenv
import os

from loguru import logger

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = os.getenv("CONFLUENT_BOOTSTRAP_SERVERS", "")
KAFKA_API_KEY            = os.getenv("CONFLUENT_API_KEY", "")
KAFKA_API_SECRET         = os.getenv("CONFLUENT_API_SECRET", "")
RAW_DOCUMENTS_TOPIC      = os.getenv("KAFKA_RAW_TOPIC", "raw-documents")
SCHEMA_DIR               = Path(__file__).parent.parent / "schemas"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS       = {"User-Agent": "rag-pipeline-web-ingest/1.0 (educational project)"}

# Articles to ingest — covers the project stack and AI/ML topics for good Q&A
ARTICLES = [
    "Retrieval-augmented_generation",
    "Large_language_model",
    "Vector_database",
    "Apache_Kafka",
    "Apache_Spark",
    "Delta_Lake",
    "Transformer_(deep_learning_architecture)",
    "Pinecone_(vector_database)",
    "Prompt_engineering",
    "Word_embedding",
]


# ─── Wikipedia fetch ──────────────────────────────────────────────────────────────

def fetch_article(title: str) -> dict | None:
    """
    Fetch a Wikipedia article as plain text via the MediaWiki action API.
    Returns dict with title, url, content, word_count — or None on failure.
    """
    try:
        r = requests.get(
            WIKIPEDIA_API,
            params={
                "action":         "query",
                "prop":           "extracts|info",
                "explaintext":    True,
                "exsectionformat":"plain",
                "inprop":         "url",
                "titles":         title,
                "format":         "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        pages   = r.json()["query"]["pages"]
        page    = next(iter(pages.values()))

        if "missing" in page:
            logger.warning("Article not found: {}", title)
            return None

        content    = page.get("extract", "").strip()
        page_url   = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}")
        word_count = len(content.split())

        if word_count < 50:
            logger.warning("Article too short ({} words), skipping: {}", word_count, title)
            return None

        logger.info("Fetched '{}' — {} words", page["title"], word_count)
        return {
            "title":      page["title"],
            "url":        page_url,
            "content":    content,
            "word_count": word_count,
        }

    except Exception as exc:
        logger.error("Failed to fetch '{}': {}", title, exc)
        return None


# ─── Kafka helpers ────────────────────────────────────────────────────────────────

def _build_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms":   "PLAIN",
        "sasl.username":     KAFKA_API_KEY,
        "sasl.password":     KAFKA_API_SECRET,
        "acks":              "all",
        "retries":           5,
        "linger.ms":         5,
    })


def _serialize(record: dict, schema: dict) -> bytes:
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, schema, record)
    return buf.getvalue()


def _delivery_report(err, msg):
    if err:
        logger.error("Delivery failed | {}", err)
    else:
        logger.debug("Delivered | topic={} partition={} offset={}",
                     msg.topic(), msg.partition(), msg.offset())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_millis() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ─── Main ─────────────────────────────────────────────────────────────────────────

def run_web_ingest() -> None:
    schema_path = SCHEMA_DIR / "document_event.avsc"
    with schema_path.open() as f:
        doc_schema = fastavro.parse_schema(json.load(f))

    producer = _build_producer()

    produced = 0
    failed   = 0

    for title in ARTICLES:
        article = fetch_article(title)
        if article is None:
            failed += 1
            continue

        doc_id = str(uuid.uuid4())
        record = {
            "document_id": doc_id,
            "source_type": "WEB",
            "source_uri":  article["url"],
            "content":     article["content"],
            "metadata": {
                "crawled_at": _now_iso(),
                "word_count": str(article["word_count"]),
                "language":   "en",
                "title":      article["title"],
            },
            "ingested_at":    _now_iso(),
            "produced_at":    _now_millis(),
            "schema_version": "1.0",
        }

        payload = _serialize(record, doc_schema)
        producer.produce(
            topic=RAW_DOCUMENTS_TOPIC,
            key=doc_id.encode(),
            value=payload,
            on_delivery=_delivery_report,
        )
        producer.poll(0)

        logger.info("Produced | doc_id={} title='{}' words={}",
                    doc_id[:8], article["title"], article["word_count"])
        produced += 1
        time.sleep(0.1)

    undelivered = producer.flush(timeout=30)
    if undelivered:
        logger.error("{} messages not delivered before timeout", undelivered)

    logger.info("Web ingest complete | produced={} failed={}", produced, failed)


if __name__ == "__main__":
    run_web_ingest()
