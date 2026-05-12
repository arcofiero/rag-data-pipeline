"""
Kafka topic administration — creates raw-documents and raw-documents-dlq
with production-appropriate configuration via the Confluent AdminClient.

Design decisions:
- retention.ms=7 days on raw-documents: gives Spark consumer enough replay
  window to recover from outages without indefinite storage costs on Confluent Cloud.
- retention.ms=30 days on DLQ: DLQ messages need human review and possible
  replay; a longer window is appropriate.
- replication_factor=3: Confluent Cloud minimum for durable topics.
- min.insync.replicas=2 pairs with acks=all on the producer: a message must
  be on at least 2 brokers before being acknowledged.
- DLQ has 1 partition: DLQ throughput is inherently low, and single-partition
  ordering is more useful for forensic inspection than parallelism.
- All operations are idempotent — TOPIC_ALREADY_EXISTS is treated as success.
"""

from __future__ import annotations

import os

from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("CONFLUENT_BOOTSTRAP_SERVERS", "")
KAFKA_API_KEY = os.getenv("CONFLUENT_API_KEY", "")
KAFKA_API_SECRET = os.getenv("CONFLUENT_API_SECRET", "")

RAW_DOCUMENTS_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw-documents")
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "raw-documents-dlq")
NUM_PARTITIONS = int(os.getenv("KAFKA_NUM_PARTITIONS", "6"))
REPLICATION_FACTOR = 3  # Confluent Cloud minimum

TOPIC_SPECS: list[dict] = [
    {
        "name": RAW_DOCUMENTS_TOPIC,
        "num_partitions": NUM_PARTITIONS,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "cleanup.policy": "delete",
            "min.insync.replicas": "2",
            "compression.type": "lz4",
        },
    },
    {
        "name": DLQ_TOPIC,
        "num_partitions": 1,
        "config": {
            "retention.ms": str(30 * 24 * 60 * 60 * 1000),  # 30 days
            "cleanup.policy": "delete",
            "min.insync.replicas": "2",
        },
    },
]


def _build_admin_client() -> AdminClient:
    return AdminClient({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": KAFKA_API_KEY,
        "sasl.password": KAFKA_API_SECRET,
    })


def create_topics(admin: AdminClient) -> None:
    """Create all pipeline topics. Already-existing topics are skipped."""
    new_topics = [
        NewTopic(
            spec["name"],
            num_partitions=spec["num_partitions"],
            replication_factor=REPLICATION_FACTOR,
            config=spec["config"],
        )
        for spec in TOPIC_SPECS
    ]

    futures = admin.create_topics(new_topics)
    for topic_name, future in futures.items():
        try:
            future.result()
            logger.info("Created topic: {}", topic_name)
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" in str(exc) or "already exists" in str(exc).lower():
                logger.info("Topic already exists (skipping): {}", topic_name)
            else:
                logger.error("Failed to create topic {}: {}", topic_name, exc)
                raise


def bootstrap_topics() -> None:
    """Entry point: create all pipeline topics."""
    logger.info("Bootstrapping Kafka topics...")
    admin = _build_admin_client()
    create_topics(admin)
    logger.info("Topic bootstrap complete.")


if __name__ == "__main__":
    bootstrap_topics()
