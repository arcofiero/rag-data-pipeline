#!/usr/bin/env python3
"""
Pipeline bootstrap — run once before starting any producer or consumer.

Order:
1. Create Kafka topics (idempotent)
2. Register Avro schemas in Confluent Schema Registry (idempotent)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from producers.topic_admin import bootstrap_topics
from producers.schema_registry import register_all_schemas


def main() -> None:
    logger.info("=== Pipeline Bootstrap ===")

    logger.info("Step 1/2: Creating Kafka topics...")
    bootstrap_topics()

    logger.info("Step 2/2: Registering Avro schemas...")
    registered = register_all_schemas()
    for subject, schema_id in registered.items():
        logger.info("  {} -> schema_id={}", subject, schema_id)

    logger.info("=== Bootstrap complete. Pipeline is ready. ===")


if __name__ == "__main__":
    main()
