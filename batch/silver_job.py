"""
Silver Transform Job — Day 5

PySpark batch job that reads from Delta Lake Bronze and writes clean,
normalized, enriched chunks to Delta Lake Silver via idempotent MERGE.

Pipeline:
  1. Soda Core Bronze quality gate — aborts if Bronze fails checks.
  2. Incremental Bronze read via high-watermark on ingestion_date.
  3. Normalize — parse metadata JSON per source type, enrich with derived fields.
  4. Filter — drop boilerplate chunks (too short, too few words, punctuation soup).
  5. Soda Core Silver quality gate on in-memory DataFrame before write.
  6. MERGE into Silver Delta keyed on chunk_id (idempotent upsert).

Design decisions:
- Batch job not streaming — Silver transforms are heavier, benefit from
  full-partition visibility for dedup and filtering.
- High-watermark incremental read avoids full Bronze scan on every run.
- Soda Core programmatic API (not CLI) — gates MERGE in same process,
  failures raise RuntimeError that Airflow catches as task failure.
- Bronze gate before transform, Silver gate after transform before MERGE.
- All Spark imports deferred inside functions so module is importable
  in pure-Python test environments without Spark installed.
- cache/unpersist around MERGE to prevent double RDD evaluation.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DELTA_BRONZE_PATH     = os.environ.get("DELTA_BRONZE_PATH", "")
DELTA_SILVER_PATH     = os.environ.get("DELTA_SILVER_PATH", "")

MIN_WORD_COUNT = int(os.getenv("SILVER_MIN_WORD_COUNT", "10"))
MIN_CHAR_COUNT = int(os.getenv("SILVER_MIN_CHAR_COUNT", "50"))

QUALITY_DIR        = Path(__file__).parent.parent / "quality"
BRONZE_CHECKS_PATH = QUALITY_DIR / "bronze_checks.yml"
SILVER_CHECKS_PATH = QUALITY_DIR / "silver_checks.yml"


def _silver_schema():
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType
    return StructType([
        StructField("chunk_id",        StringType(),  nullable=False),
        StructField("document_id",     StringType(),  nullable=False),
        StructField("chunk_index",     IntegerType(), nullable=False),
        StructField("content",         StringType(),  nullable=False),
        StructField("source",          StringType(),  nullable=False),
        StructField("source_uri",      StringType(),  nullable=True),
        StructField("author",          StringType(),  nullable=True),
        StructField("page_count",      IntegerType(), nullable=True),
        StructField("crawled_at",      StringType(),  nullable=True),
        StructField("word_count_src",  StringType(),  nullable=True),
        StructField("table_name",      StringType(),  nullable=True),
        StructField("row_id",          StringType(),  nullable=True),
        StructField("word_count",      IntegerType(), nullable=False),
        StructField("char_count",      IntegerType(), nullable=False),
        StructField("language",        StringType(),  nullable=False),
        StructField("schema_version",  StringType(),  nullable=True),
        StructField("ingested_at",     StringType(),  nullable=False),
        StructField("processed_at",    StringType(),  nullable=False),
        StructField("processed_date",  StringType(),  nullable=False),
    ])


def _build_spark():
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    builder = (
        SparkSession.builder
        .appName("rag-silver-batch")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "true")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def _read_incremental_bronze(spark):
    """
    High-watermark incremental read.
    If Silver exists: read Bronze rows where ingestion_date >= max(processed_date) in Silver.
    First run: read all Bronze rows.
    >= not > to catch late-arriving rows on the boundary date.
    MERGE handles any duplicates introduced by the overlap.
    """
    from delta import DeltaTable
    from pyspark.sql import functions as F

    bronze_df = spark.read.format("delta").load(DELTA_BRONZE_PATH)

    if DeltaTable.isDeltaTable(spark, DELTA_SILVER_PATH):
        silver_df = spark.read.format("delta").load(DELTA_SILVER_PATH)
        watermark = silver_df.agg(F.max("processed_date")).collect()[0][0]
        if watermark:
            logger.info("Incremental read: Bronze ingestion_date >= {}", watermark)
            bronze_df = bronze_df.filter(F.col("ingestion_date") >= watermark)
        else:
            logger.info("Silver watermark is null — reading all Bronze rows")
    else:
        logger.info("Silver table not found — full Bronze read (first run)")

    row_count = bronze_df.count()
    logger.info("Bronze rows to process: {}", row_count)
    return bronze_df


def _normalize_partition(rows: Iterator) -> Iterator:
    """
    mapPartitions transform. Runs on executors — all imports local.
    Directly testable without SparkSession.

    _g() provides uniform field access across Spark Row, namedtuple, and dict.
    namedtuples raise TypeError on string indexing, caught by try/except.
    """
    import json as _json
    import re as _re
    from datetime import datetime as _dt, timezone as _tz

    min_words = MIN_WORD_COUNT
    min_chars = MIN_CHAR_COUNT

    def _g(row, field):
        try:
            return row[field]
        except (TypeError, ValueError, KeyError):
            return getattr(row, field, None)

    def _parse_meta(meta_str):
        if not meta_str:
            return {}
        try:
            return _json.loads(meta_str)
        except Exception:
            return {}

    def _word_count(text):
        return len(_re.findall(r"\w+", text))

    def _is_boilerplate(text):
        if len(text) < min_chars:
            return True
        if _word_count(text) < min_words:
            return True
        alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
        if alnum_ratio < 0.5:
            return True
        return False

    now_str = _dt.now(_tz.utc).isoformat()
    today   = _dt.now(_tz.utc).strftime("%Y-%m-%d")

    for row in rows:
        meta    = _parse_meta(_g(row, "metadata"))
        source  = (_g(row, "source") or "UNKNOWN").upper()
        content = _g(row, "content") or ""

        author = page_count = crawled_at = None
        word_count_s = table_name = row_id_val = None

        if source == "PDF":
            author    = meta.get("author")
            raw_pages = meta.get("page_count")
            try:
                page_count = int(raw_pages) if raw_pages else None
            except (ValueError, TypeError):
                page_count = None
        elif source == "WEB":
            crawled_at   = meta.get("crawled_at")
            word_count_s = meta.get("word_count")
        elif source == "STRUCTURED":
            table_name = meta.get("table")
            row_id_val = meta.get("row_id")

        wc       = _word_count(content)
        cc       = len(content)
        filtered = _is_boilerplate(content)

        ci_raw = _g(row, "chunk_index")
        yield {
            "chunk_id":       _g(row, "chunk_id"),
            "document_id":    _g(row, "document_id"),
            "chunk_index":    int(ci_raw) if ci_raw is not None else 0,
            "content":        content,
            "source":         source,
            "source_uri":     _g(row, "source_uri"),
            "author":         author,
            "page_count":     page_count,
            "crawled_at":     crawled_at,
            "word_count_src": word_count_s,
            "table_name":     table_name,
            "row_id":         row_id_val,
            "word_count":     wc,
            "char_count":     cc,
            "language":       "en",
            "schema_version": _g(row, "schema_version"),
            "ingested_at":    _g(row, "ingested_at"),
            "processed_at":   now_str,
            "processed_date": today,
            "_filtered":      filtered,
        }


def _merge_into_silver(silver_df, spark) -> int:
    """
    Idempotent WHEN NOT MATCHED INSERT only.
    Silver is append-only: first clean version of a chunk wins.
    Returns row count for metrics.
    """
    from delta import DeltaTable

    row_count = silver_df.count()

    if DeltaTable.isDeltaTable(spark, DELTA_SILVER_PATH):
        silver_table = DeltaTable.forPath(spark, DELTA_SILVER_PATH)
        (
            silver_table.alias("existing")
            .merge(silver_df.alias("incoming"), "existing.chunk_id = incoming.chunk_id")
            .whenNotMatchedInsertAll()
            .execute()
        )
        logger.info("MERGE complete → Silver at {}", DELTA_SILVER_PATH)
    else:
        logger.info("Silver table not found — creating at {}", DELTA_SILVER_PATH)
        (
            silver_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("source", "processed_date")
            .save(DELTA_SILVER_PATH)
        )
        logger.info("Silver table created at {}", DELTA_SILVER_PATH)

    return row_count


def _run_soda_scan(spark, df, checks_path: Path, dataset_name: str) -> None:
    """
    Programmatic Soda Core scan against a Spark DataFrame.
    No disk write needed — DataFrame passed directly to soda-core-spark-df.
    Raises RuntimeError on check failure — Airflow catches as task failure.
    """
    try:
        from soda.scan import Scan
    except ImportError:
        logger.warning("soda-core not installed — skipping quality gate for {}", dataset_name)
        return

    logger.info("Running Soda Core checks | dataset={} checks={}", dataset_name, checks_path)
    df.createOrReplaceTempView(dataset_name)
    scan = Scan()
    scan.set_scan_definition_name(f"rag-pipeline-{dataset_name}")
    scan.set_data_source_name(dataset_name)
    scan.add_spark_session(spark, data_source_name=dataset_name)
    with checks_path.open() as fh:
        scan.add_sodacl_yaml_str(fh.read())
    scan.execute()

    if scan.has_check_fails():
        failed_text = scan.get_checks_fail_text()
        logger.error("Soda Core check failures | dataset={} failed={}", dataset_name, failed_text)
        raise RuntimeError(
            f"Soda Core quality gate failed for {dataset_name}. "
            f"Failed checks: {failed_text}. Blocking downstream job."
        )
    logger.info("Soda Core checks passed | dataset={}", dataset_name)


def run_silver_job() -> None:
    from lineage.emitter import silver_emitter
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    emitter = silver_emitter()
    logger.info("Starting Silver batch job | bronze={} silver={}", DELTA_BRONZE_PATH, DELTA_SILVER_PATH)
    emitter.emit_start()

    try:
        spark = _build_spark()
        spark.sparkContext.setLogLevel("WARN")

        bronze_df = _read_incremental_bronze(spark)
        if bronze_df.rdd.isEmpty():
            logger.info("No new Bronze rows — Silver job exiting early")
            spark.stop()
            return

        _run_soda_scan(spark, bronze_df, BRONZE_CHECKS_PATH, "bronze_documents")

        base_schema    = _silver_schema()
        interim_schema = StructType(base_schema.fields + [StructField("_filtered", StringType(), nullable=True)])
        interim_df     = spark.createDataFrame(bronze_df.rdd.mapPartitions(_normalize_partition), schema=interim_schema)
        interim_df     = interim_df.cache()

        total_rows    = interim_df.count()
        filtered_rows = interim_df.filter(F.col("_filtered") == True).count()
        passed_rows   = total_rows - filtered_rows

        logger.info("Normalization complete | total={} filtered={} passed={}", total_rows, filtered_rows, passed_rows)

        silver_df = (
            interim_df
            .filter(F.col("_filtered") == False)
            .drop("_filtered")
            .dropDuplicates(["chunk_id"])
        )

        if silver_df.rdd.isEmpty():
            logger.warning("All Bronze rows filtered — nothing to write to Silver")
            interim_df.unpersist()
            spark.stop()
            return

        _run_soda_scan(spark, silver_df, SILVER_CHECKS_PATH, "silver_chunks")

        silver_df    = silver_df.cache()
        merged_count = _merge_into_silver(silver_df, spark)

        interim_df.unpersist()
        silver_df.unpersist()

        logger.info(
            "Silver job complete | bronze_read={} filtered={} passed={} silver_merged={}",
            total_rows, filtered_rows, passed_rows, merged_count,
        )
        spark.stop()
        emitter.emit_complete(row_count=merged_count)
    except Exception as exc:
        emitter.emit_fail(error=exc)
        raise


if __name__ == "__main__":
    run_silver_job()
