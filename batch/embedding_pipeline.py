"""
Embedding Pipeline — Day 6

PySpark batch job that reads unembedded chunks from Delta Lake Silver,
calls the OpenAI embeddings API in batches via mapPartitions, upserts
vectors into Pinecone, and writes an audit record to Delta Lake Gold.

Pipeline:
  1. Soda Core Silver quality gate — aborts if Silver fails checks.
  2. Incremental Silver read — left anti-join Silver on Gold by chunk_id.
     Already-embedded chunks skipped. Fully idempotent.
  3. Embed via mapPartitions — one batched OpenAI request per partition
     sub-batch of EMBED_BATCH_SIZE chunks. Never row-by-row.
  4. Upsert to Pinecone — vector_id == chunk_id (idempotency guarantee).
     Full metadata attached: source, document_id, ingested_at, embedded_at.
  5. Append to Gold audit trail — only after Pinecone upsert succeeds.
     Gold is append-only, never updated.
  6. Soda Core Gold freshness gate after write.

Design decisions:
- Left anti-join not watermark: correct for monotonically growing Gold;
  handles retry of failed chunks regardless of age.
- Pinecone upsert before Gold write: if Gold write fails, chunk stays
  unembedded and is retried next run via anti-join. Reverse order causes
  permanent data loss from Pinecone.
- Gold append-only: preserves full embedding history as audit trail.
- Content truncated at 8000 chars (conservative proxy for token limit).
- OpenAI client max_retries=3 (exponential backoff built in).
- Failure isolation: OpenAI failure skips batch (retried next run);
  Pinecone failure excludes chunk from Gold write.
- All Spark imports deferred — module importable without Spark installed.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DELTA_SILVER_PATH      = os.environ.get("DELTA_SILVER_PATH", "")
DELTA_GOLD_PATH        = os.environ.get("DELTA_GOLD_PATH", "")
AWS_ACCESS_KEY_ID      = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
OPENAI_API_KEY         = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY       = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME    = os.environ.get("PINECONE_INDEX_NAME", "rag-pipeline")
EMBED_BATCH_SIZE       = int(os.getenv("EMBED_BATCH_SIZE", "512"))
PINECONE_BATCH_SIZE    = int(os.getenv("PINECONE_BATCH_SIZE", "100"))
FRESHNESS_HOURS        = int(os.getenv("EMBEDDING_FRESHNESS_HOURS", "24"))

QUALITY_DIR        = Path(__file__).parent.parent / "quality"
SILVER_CHECKS_PATH = QUALITY_DIR / "silver_checks.yml"
GOLD_CHECKS_PATH   = QUALITY_DIR / "gold_checks.yml"


def _gold_schema():
    from pyspark.sql.types import StringType, StructField, StructType
    return StructType([
        StructField("chunk_id",      StringType(), nullable=False),
        StructField("vector_id",     StringType(), nullable=False),
        StructField("document_id",   StringType(), nullable=False),
        StructField("source",        StringType(), nullable=False),
        StructField("model",         StringType(), nullable=False),
        StructField("ingested_at",   StringType(), nullable=True),
        StructField("embedded_at",   StringType(), nullable=False),
        StructField("embedded_date", StringType(), nullable=False),
    ])


def _build_spark():
    """
    NOTE: static credentials for local dev only — use IAM instance profile
    in production by removing credential .config() calls and setting
    spark.hadoop.fs.s3a.aws.credentials.provider to
    com.amazonaws.auth.InstanceProfileCredentialsProvider
    """
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    builder = (
        SparkSession.builder
        .appName("rag-gold-embedding")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.jars.packages", ",".join([
            "io.delta:delta-spark_2.12:3.2.0",
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.261",
        ]))
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def _read_unembedded_silver(spark):
    """
    Left anti-join Silver on Gold by chunk_id.
    Returns Silver rows with no corresponding Gold record.
    First run (no Gold table): returns all Silver rows.
    """
    from delta import DeltaTable
    silver_df = spark.read.format("delta").load(DELTA_SILVER_PATH)
    if DeltaTable.isDeltaTable(spark, DELTA_GOLD_PATH):
        gold_df   = spark.read.format("delta").load(DELTA_GOLD_PATH).select("chunk_id")
        unembedded = silver_df.join(gold_df, on="chunk_id", how="left_anti")
        logger.info("Anti-join complete — reading Silver rows not yet in Gold")
    else:
        unembedded = silver_df
        logger.info("Gold table not found — embedding all Silver rows (first run)")
    count = unembedded.count()
    logger.info("Unembedded Silver chunks: {}", count)
    return unembedded


def _embed_and_upsert_partition(rows: Iterator) -> Iterator:
    """
    Runs on executors. Self-contained — all imports local.

    Per partition:
    1. Collect all rows into memory.
    2. Split into sub-batches of EMBED_BATCH_SIZE.
    3. Call OpenAI embeddings API once per sub-batch.
    4. Upsert vectors to Pinecone in sub-batches of PINECONE_BATCH_SIZE.
    5. Yield Gold audit dict for each successfully upserted chunk.

    Failure isolation:
    - OpenAI failure: log and skip batch — chunks retried next run.
    - Pinecone failure: mark vectors _failed=True — excluded from Gold yield.
    """
    import os as _os
    from datetime import datetime as _dt, timezone as _tz
    from openai import OpenAI
    from pinecone import Pinecone

    api_key     = _os.environ.get("OPENAI_API_KEY", "")
    embed_model = _os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    pc_api_key  = _os.environ.get("PINECONE_API_KEY", "")
    pc_index    = _os.environ.get("PINECONE_INDEX_NAME", "rag-pipeline")
    batch_size  = int(_os.environ.get("EMBED_BATCH_SIZE", "512"))
    pc_batch    = int(_os.environ.get("PINECONE_BATCH_SIZE", "100"))

    openai_client   = OpenAI(api_key=api_key, max_retries=3)
    pinecone_client = Pinecone(api_key=pc_api_key)
    index           = pinecone_client.Index(pc_index)

    def _g(row, field):
        try:
            return row[field]
        except TypeError:
            return getattr(row, field, None)

    partition_rows = list(rows)
    if not partition_rows:
        return

    for batch_start in range(0, len(partition_rows), batch_size):
        batch      = partition_rows[batch_start:batch_start + batch_size]
        texts      = [(_g(r, "content") or "")[:8000] for r in batch]
        chunk_ids  = [_g(r, "chunk_id")    for r in batch]
        doc_ids    = [_g(r, "document_id") for r in batch]
        sources    = [_g(r, "source")      for r in batch]
        ingested   = [_g(r, "ingested_at") for r in batch]

        try:
            response   = openai_client.embeddings.create(input=texts, model=embed_model)
            embeddings = [item.embedding for item in response.data]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "OpenAI embedding call failed for batch of %d chunks: %s", len(batch), exc
            )
            continue

        embedded_at   = _dt.now(_tz.utc).isoformat()
        embedded_date = _dt.now(_tz.utc).strftime("%Y-%m-%d")

        vectors = []
        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
            vectors.append({
                "id":     chunk_id,
                "values": embedding,
                "metadata": {
                    "chunk_id":    chunk_id,
                    "document_id": doc_ids[i],
                    "source":      sources[i],
                    "ingested_at": ingested[i] or "",
                    "embedded_at": embedded_at,
                    "model":       embed_model,
                },
            })

        for pc_start in range(0, len(vectors), pc_batch):
            pc_batch_vecs = vectors[pc_start:pc_start + pc_batch]
            try:
                index.upsert(vectors=pc_batch_vecs)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    "Pinecone upsert failed for %d vectors: %s", len(pc_batch_vecs), exc
                )
                for v in pc_batch_vecs:
                    v["_failed"] = True

        failed_ids = {v["id"] for v in vectors if v.get("_failed")}
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id in failed_ids:
                continue
            yield {
                "chunk_id":      chunk_id,
                "vector_id":     chunk_id,
                "document_id":   doc_ids[i],
                "source":        sources[i],
                "model":         embed_model,
                "ingested_at":   ingested[i],
                "embedded_at":   embedded_at,
                "embedded_date": embedded_date,
            }


def _append_to_gold(gold_df, spark) -> int:
    """
    Append Gold audit records to Delta Lake Gold table.
    Append-only — never MERGE or UPDATE. Preserves full embedding history.
    First run creates the table with partitioning.
    """
    from delta import DeltaTable
    row_count = gold_df.count()
    if row_count == 0:
        logger.info("No Gold records to write — skipping")
        return 0
    if DeltaTable.isDeltaTable(spark, DELTA_GOLD_PATH):
        (
            gold_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .save(DELTA_GOLD_PATH)
        )
        logger.info("Appended {} records → Gold at {}", row_count, DELTA_GOLD_PATH)
    else:
        logger.info("Gold table not found — creating at {}", DELTA_GOLD_PATH)
        (
            gold_df.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("source", "embedded_date")
            .save(DELTA_GOLD_PATH)
        )
        logger.info("Gold table created at {}", DELTA_GOLD_PATH)
    return row_count


def _run_soda_scan(spark, df, checks_path: Path, dataset_name: str) -> None:
    """Programmatic Soda Core scan. Raises RuntimeError on failure."""
    try:
        from soda.scan import Scan
    except ImportError:
        logger.warning("soda-core not installed — skipping quality gate for {}", dataset_name)
        return
    logger.info("Running Soda Core checks | dataset={} checks={}", dataset_name, checks_path)
    scan = Scan()
    scan.set_scan_definition_name(f"rag-pipeline-{dataset_name}")
    scan.set_data_source_name(dataset_name)
    scan.add_spark_session(spark, data_source_name=dataset_name)
    scan.add_dataframe_datasets(dataset_name, [(dataset_name, df)])
    with checks_path.open() as fh:
        scan.add_sodacl_yaml_str(fh.read())
    scan.execute()
    if scan.has_check_failures():
        failed = [c.name for c in scan.get_checks() if c.outcome and c.outcome.value == "fail"]
        logger.error("Soda Core check failures | dataset={} failed={}", dataset_name, failed)
        raise RuntimeError(
            f"Soda Core quality gate failed for {dataset_name}. "
            f"Failed checks: {failed}. Blocking downstream job."
        )
    logger.info("Soda Core checks passed | dataset={}", dataset_name)


def run_embedding_pipeline() -> None:
    """
    Full Gold embedding pipeline.
    Step order: Silver gate → anti-join read → embed+upsert → Gold write → Gold gate.
    """
    logger.info("Starting embedding pipeline | silver={} gold={}", DELTA_SILVER_PATH, DELTA_GOLD_PATH)
    spark = _build_spark()
    spark.sparkContext.setLogLevel("WARN")

    silver_all = spark.read.format("delta").load(DELTA_SILVER_PATH)
    _run_soda_scan(spark, silver_all, SILVER_CHECKS_PATH, "silver_chunks")

    unembedded_df = _read_unembedded_silver(spark)
    if unembedded_df.rdd.isEmpty():
        logger.info("No unembedded Silver chunks — embedding pipeline exiting early")
        spark.stop()
        return

    gold_rdd  = unembedded_df.rdd.mapPartitions(_embed_and_upsert_partition)
    gold_df   = spark.createDataFrame(gold_rdd, schema=_gold_schema())
    gold_df   = gold_df.cache()

    embedded_count = gold_df.count()
    logger.info("Chunks embedded and upserted to Pinecone: {}", embedded_count)

    written_count = _append_to_gold(gold_df, spark)
    gold_df.unpersist()

    if written_count > 0:
        gold_for_check = spark.read.format("delta").load(DELTA_GOLD_PATH)
        _run_soda_scan(spark, gold_for_check, GOLD_CHECKS_PATH, "gold_embeddings")

    logger.info(
        "Embedding pipeline complete | embedded={} gold_written={}",
        embedded_count, written_count,
    )
    spark.stop()


if __name__ == "__main__":
    run_embedding_pipeline()
