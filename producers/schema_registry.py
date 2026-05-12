"""
Confluent Schema Registry client — registers and manages Avro schemas
for all pipeline topics.

Design decisions:
- Subject naming follows TopicNameStrategy: <topic>-value for value schemas.
  This is the Confluent default and matches what Confluent Kafka serializers
  expect without extra configuration.
- BACKWARD compatibility on raw-documents-value: new versions may add nullable
  fields with defaults but may not remove existing fields. This allows rolling
  consumer upgrades without coordinating producer deploys.
- NONE compatibility on raw-documents-dlq-value: DLQ schemas evolve for
  operational reasons independently of compatibility constraints.
- register_all_schemas() is idempotent — re-registering an identical schema
  returns the existing schema ID without creating a new version.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from confluent_kafka.schema_registry import SchemaRegistryClient, Schema
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SCHEMA_REGISTRY_URL = os.getenv("CONFLUENT_SCHEMA_REGISTRY_URL", "")
SCHEMA_REGISTRY_API_KEY = os.getenv("CONFLUENT_SCHEMA_REGISTRY_API_KEY", "")
SCHEMA_REGISTRY_API_SECRET = os.getenv("CONFLUENT_SCHEMA_REGISTRY_API_SECRET", "")

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

RAW_DOCUMENTS_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw-documents")
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "raw-documents-dlq")

# (schema_file_stem, subject_name, compatibility_level)
SCHEMA_SUBJECTS: list[tuple[str, str, str]] = [
    ("document_event", f"{RAW_DOCUMENTS_TOPIC}-value", "BACKWARD"),
    ("dead_letter_event", f"{DLQ_TOPIC}-value", "NONE"),
]


def build_registry_client() -> SchemaRegistryClient:
    """Build a SchemaRegistryClient authenticated against Confluent Cloud."""
    return SchemaRegistryClient({
        "url": SCHEMA_REGISTRY_URL,
        "basic.auth.user.info": f"{SCHEMA_REGISTRY_API_KEY}:{SCHEMA_REGISTRY_API_SECRET}",
    })


def _load_schema_str(stem: str) -> str:
    path = SCHEMA_DIR / f"{stem}.avsc"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return path.read_text()


def register_schema(
    client: SchemaRegistryClient,
    schema_file_stem: str,
    subject: str,
    compatibility: str,
) -> int:
    """
    Register a schema under `subject`. Returns the schema ID.
    Idempotent: identical re-registration returns the existing ID.
    """
    schema_str = _load_schema_str(schema_file_stem)

    try:
        client.set_compatibility(subject_name=subject, level=compatibility)
        logger.info("Set compatibility | subject={} level={}", subject, compatibility)
    except Exception as exc:
        # Non-fatal: subject may not exist yet on first registration
        logger.warning("Could not set compatibility for {}: {}", subject, exc)

    schema_id = client.register_schema(
        subject_name=subject,
        schema=Schema(schema_str, schema_type="AVRO"),
    )
    logger.info(
        "Registered schema | subject={} schema_id={} file={}",
        subject, schema_id, schema_file_stem,
    )
    return schema_id


def register_all_schemas() -> dict[str, int]:
    """
    Register all pipeline schemas. Call during bootstrap before starting
    any producer or consumer.
    Returns mapping of subject -> schema_id.
    """
    client = build_registry_client()
    results: dict[str, int] = {}

    for stem, subject, compatibility in SCHEMA_SUBJECTS:
        try:
            schema_id = register_schema(client, stem, subject, compatibility)
            results[subject] = schema_id
        except Exception as exc:
            logger.error("Failed to register {} under {}: {}", stem, subject, exc)
            raise

    logger.info("All schemas registered | count={}", len(results))
    return results


def list_subjects() -> list[str]:
    """List all subjects registered in the Schema Registry."""
    client = build_registry_client()
    subjects = client.get_subjects()
    logger.info("Schema Registry subjects: {}", subjects)
    return subjects


if __name__ == "__main__":
    registered = register_all_schemas()
    for subject, sid in registered.items():
        logger.info("  {} → schema_id={}", subject, sid)
