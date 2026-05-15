"""
E2E Smoke Test — Day 9

Validates the full pipeline chain without Airflow or S3.
Uses local Delta paths and real Confluent + OpenAI + Pinecone credentials from .env.

Steps:
  1. Produce events to Confluent Cloud Kafka.
  2. Check FastAPI /health.
  3. Query RAG endpoint and assert answer + sources present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

FASTAPI_URL     = os.getenv("FASTAPI_URL", "http://localhost:8000")
PRODUCER_EVENTS = int(os.getenv("SMOKE_TEST_EVENTS", "10"))

_REQUIRED_VARS = [
    "CONFLUENT_BOOTSTRAP_SERVERS",
    "CONFLUENT_API_KEY",
    "CONFLUENT_API_SECRET",
    "KAFKA_RAW_TOPIC",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "DELTA_BRONZE_PATH",
    "DELTA_SILVER_PATH",
    "DELTA_GOLD_PATH",
]


def check_env() -> None:
    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    for var in missing:
        logger.warning("Missing required environment variable: {}", var)
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {missing}. "
            "Set them in your .env file before running the smoke test."
        )
    logger.info("Environment check passed — all required vars present")


def step_produce(n: int) -> int:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from producers.document_producer import run_producer
    logger.info("Producing {} events to Kafka topic {}", n, os.environ.get("KAFKA_RAW_TOPIC"))
    run_producer(num_events=n, delay_seconds=0.01)
    logger.info("Produced {} events", n)
    return n


def step_query_api(question: str) -> dict:
    url  = f"{FASTAPI_URL}/query"
    resp = requests.post(
        url,
        json={"query": question, "top_k": 3},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def run_smoke_test() -> None:
    logger.info("=== RAG Pipeline E2E Smoke Test ===")
    check_env()

    # Step 1: Produce events
    logger.info("Step 1/3 — Producing {} events to Kafka", PRODUCER_EVENTS)
    step_produce(PRODUCER_EVENTS)
    logger.info("Step 1 complete — events in Kafka topic")

    # Step 2: Check FastAPI health
    logger.info("Step 2/3 — Checking FastAPI /health")
    resp = requests.get(f"{FASTAPI_URL}/health", timeout=10)
    resp.raise_for_status()
    logger.info("FastAPI healthy | response={}", resp.json())

    # Step 3: Query the RAG endpoint
    logger.info("Step 3/3 — Querying RAG endpoint")
    question = "What are the key topics in the ingested documents?"
    result   = step_query_api(question)
    logger.info("RAG response received | answer_len={}", len(result.get("answer", "")))
    assert result.get("answer"), "RAG endpoint returned empty answer"
    assert isinstance(result.get("sources"), list), "RAG endpoint missing sources list"
    logger.info("=== Smoke test PASSED ===")
    logger.info("Answer preview: {}", result["answer"][:300])


if __name__ == "__main__":
    run_smoke_test()
