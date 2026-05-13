"""
Spark Structured Streaming consumer — Bronze ingestion layer.

Reads Avro-encoded events from the Confluent Cloud raw-documents Kafka topic,
deserializes them with fastavro, routes failures to a Delta Lake DLQ table,
chunks and cleans valid content via mapPartitions, deduplicates by chunk_id,
and MERGEs into the Delta Lake Bronze table.

Design decisions:
- schemaless_reader (not Confluent wire-format) because the producer uses
  fastavro.schemaless_writer directly. Schema is loaded from disk once per
  driver and broadcast implicitly via UDF closure.
- Avro deserialization in a Python UDF returns a JSON string (not a struct)
  to avoid declaring a StructType that mirrors the Avro schema. from_json
  then parses only the fields we need. This keeps the UDF return type simple
  (StringType) and avoids nested StructType schema drift.
- mapPartitions for chunking: one Python interpreter call per partition
  rather than one per row. Significant cost reduction on the GIL-bound
  chunking + SHA-256 work.
- MERGE with WHEN NOT MATCHED INSERT: Bronze is append-only. Re-running
  against the same Kafka offsets produces zero duplicate rows.
- DLQ is a Delta table (not a Kafka topic) because Delta gives us time
  travel, schema enforcement, and SQL-queryable forensics for free.
- SASL credentials are passed as kafka.* options on the readStream, not
  in SparkConf, so they do not appear in the Spark UI environment tab.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import fastavro
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS = os.getenv("CONFLUENT_BOOTSTRAP_SERVERS", "")
API_KEY = os.getenv("CONFLUENT_API_KEY", "")
API_SECRET = os.getenv("CONFLUENT_API_SECRET", "")
RAW_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw-documents")

DELTA_BRONZE_PATH = os.getenv("DELTA_BRONZE_PATH", "")
DELTA_BRONZE_DLQ_PATH = os.getenv("DELTA_BRONZE_DLQ_PATH", "")
SPARK_CHECKPOINT_PATH = os.getenv("SPARK_CHECKPOINT_PATH", "")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MAX_OFFSETS = int(os.getenv("MAX_OFFSETS_PER_TRIGGER", "5000"))

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"
PROJECT_ROOT = str(Path(__file__).parent.parent)


# ─── Schema ────────────────────────────────────────────────────────────────────


def _load_doc_schema() -> dict:
    with (SCHEMA_DIR / "document_event.avsc").open() as f:
        return fastavro.parse_schema(json.load(f))


def _bronze_schema() -> StructType:
    return StructType([
        StructField("chunk_id", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("content", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("ingested_at", StringType(), True),
        StructField("ingestion_date", StringType(), False),
        StructField("produced_at", TimestampType(), True),
    ])


# ─── SparkSession ───────────────────────────────────────────────────────────────


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("rag-bronze-streaming")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # NOTE: static credentials for local dev only — use IAM instance profile in
        # production by removing these three .config() calls and setting
        # spark.hadoop.fs.s3a.aws.credentials.provider to
        # com.amazonaws.auth.InstanceProfileCredentialsProvider
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config(
            "spark.jars.packages",
            ",".join([
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
                "io.delta:delta-spark_2.12:3.2.0",
                "org.apache.hadoop:hadoop-aws:3.3.4",
                "com.amazonaws:aws-java-sdk-bundle:1.12.261",
            ]),
        )
        .getOrCreate()
    )


# ─── Avro deserialization UDF ───────────────────────────────────────────────────


def _make_deserialize_udf(doc_schema: dict):
    """
    Returns a UDF that deserializes schemaless Avro bytes to a JSON string.
    Returns None on any failure so rows can be identified and routed to the DLQ.
    The schema dict is captured in the closure — no broadcast needed for a
    single small schema object.
    """
    def _deserialize(avro_bytes: bytes):
        if avro_bytes is None:
            return None
        try:
            record = fastavro.schemaless_reader(io.BytesIO(avro_bytes), doc_schema)
            return json.dumps(record, default=str)
        except Exception:
            return None

    return F.udf(_deserialize, StringType())


# ─── mapPartitions chunking ─────────────────────────────────────────────────────


def _chunk_partition(rows: Iterator) -> Iterator:
    """
    Applied via RDD.mapPartitions. Imports chunker locally so this function
    is serializable without requiring chunker to be imported at module level
    on the driver before executors are launched.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from streaming.chunker import chunk_text

    ingestion_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for row in rows:
        doc_id = row["document_id"]
        content = row["content"] or ""
        source = row["source_type"] or "UNKNOWN"
        ingested_at = row["ingested_at"] or ""

        produced_at_raw = row["produced_at"] if row["produced_at"] else None
        try:
            produced_at = datetime.fromisoformat(produced_at_raw) if produced_at_raw else None
        except (ValueError, TypeError):
            produced_at = None

        try:
            chunks = chunk_text(doc_id, content, CHUNK_SIZE, CHUNK_OVERLAP)
        except Exception as exc:
            logger.warning(
                "chunk_text failed | doc_id={} error={}", doc_id, exc
            )
            continue

        for chunk in chunks:
            yield (
                chunk.chunk_id,
                int(chunk.chunk_index),
                chunk.content,
                doc_id,
                source,
                ingested_at,
                ingestion_date,
                produced_at,
            )


# ─── foreachBatch handler ───────────────────────────────────────────────────────


def process_batch(batch_df: DataFrame, batch_id: int) -> None:
    from delta.tables import DeltaTable

    spark = batch_df.sparkSession

    batch_count = batch_df.count()
    logger.info("Batch received | batch_id={} rows={}", batch_id, batch_count)

    if batch_count == 0:
        logger.info("Empty batch, skipping | batch_id={}", batch_id)
        return

    # 1. Deserialize Avro -------------------------------------------------------
    doc_schema = _load_doc_schema()
    deserialize_udf = _make_deserialize_udf(doc_schema)

    annotated = batch_df.withColumn("json_str", deserialize_udf(F.col("value")))

    # 2. Route failures to Delta DLQ --------------------------------------------
    dlq_df = annotated.filter(F.col("json_str").isNull()).select(
        F.col("value"),
        F.current_timestamp().alias("failed_at"),
        F.lit("Avro deserialization failure").alias("error_reason"),
        F.lit(batch_id).cast(IntegerType()).alias("batch_id"),
    )
    dlq_count = dlq_df.count()
    if dlq_count > 0:
        dlq_df.write.format("delta").mode("append").option("mergeSchema", "true").save(DELTA_BRONZE_DLQ_PATH)
        logger.warning(
            "Routed rows to Bronze DLQ | batch_id={} count={}",
            batch_id, dlq_count,
        )

    # 3. Parse valid JSON --------------------------------------------------------
    parsed_schema = StructType([
        StructField("document_id", StringType(), True),
        StructField("source_type", StringType(), True),
        StructField("content", StringType(), True),
        StructField("ingested_at", StringType(), True),
        StructField("produced_at", StringType(), True),
    ])
    valid_df = (
        annotated
        .filter(F.col("json_str").isNotNull())
        .select(F.from_json(F.col("json_str"), parsed_schema).alias("doc"))
        .select("doc.*")
        .filter(
            F.col("document_id").isNotNull() & F.col("content").isNotNull()
        )
    )
    valid_count = valid_df.count()
    if valid_count == 0:
        logger.info("No valid rows after parsing | batch_id={}", batch_id)
        return

    # 4. Chunk via mapPartitions -------------------------------------------------
    chunked_rdd = valid_df.rdd.mapPartitions(_chunk_partition)
    chunked_df = spark.createDataFrame(chunked_rdd, schema=_bronze_schema())

    # 5. Deduplicate within batch ------------------------------------------------
    chunks_df = chunked_df.dropDuplicates(["chunk_id"])
    chunks_df = chunks_df.cache()
    chunks_written = chunks_df.count()
    logger.info(
        "Chunks after dedup | batch_id={} count={}", batch_id, chunks_written
    )

    if chunks_written == 0:
        chunks_df.unpersist()
        return

    # 6. MERGE into Bronze (or create on first run) -----------------------------
    if DeltaTable.isDeltaTable(spark, DELTA_BRONZE_PATH):
        bronze = DeltaTable.forPath(spark, DELTA_BRONZE_PATH)
        (
            bronze.alias("tgt")
            .merge(chunks_df.alias("src"), "tgt.chunk_id = src.chunk_id")
            .whenNotMatchedInsertAll()
            .execute()
        )
        logger.info(
            "Merged into Bronze | batch_id={} chunks={}", batch_id, chunks_written
        )
    else:
        (
            chunks_df.write
            .format("delta")
            .mode("append")
            .partitionBy("source", "ingestion_date")
            .save(DELTA_BRONZE_PATH)
        )
        logger.info(
            "Created Bronze Delta table | batch_id={} chunks={}",
            batch_id, chunks_written,
        )

    chunks_df.unpersist()
    logger.info(
        "Batch complete | batch_id={} valid_count={} dlq_count={} chunks_written={}",
        batch_id, valid_count, dlq_count, chunks_written,
    )


# ─── Streaming entry point ──────────────────────────────────────────────────────


def run_streaming() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(
        "Starting streaming job | topic={} bronze={}",
        RAW_TOPIC, DELTA_BRONZE_PATH,
    )

    sasl_jaas = (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="{API_KEY}" password="{API_SECRET}";'
    )

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("kafka.security.protocol", "SASL_SSL")
        .option("kafka.sasl.mechanism", "PLAIN")
        .option("kafka.sasl.jaas.config", sasl_jaas)
        .option("subscribe", RAW_TOPIC)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS)
        .option("failOnDataLoss", "false")
        .load()
    )

    query = (
        raw_stream
        .writeStream
        .foreachBatch(process_batch)
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", SPARK_CHECKPOINT_PATH)
        .start()
    )

    logger.info("Streaming query active | awaiting termination...")
    query.awaitTermination()


if __name__ == "__main__":
    run_streaming()
