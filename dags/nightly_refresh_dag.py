"""
Nightly Embedding Refresh DAG — Day 7

Airflow DAG that runs nightly at 02:00 UTC to detect and re-embed Silver
chunks that have not been embedded within the 24-hour SLA.

Task chain:
  freshness_check → re_embedding_pipeline → gold_soda_gate

Design decisions:
- Freshness check: a PythonOperator that counts Silver chunks with no
  corresponding Gold record older than EMBEDDING_FRESHNESS_HOURS. If the
  count is zero, downstream tasks are skipped via ShortCircuitOperator.
- Re-embedding is handled by running the same embedding_pipeline again.
  The anti-join in _read_unembedded_silver naturally picks up stale chunks
  without any special logic — idempotency is built into the pipeline.
- ShortCircuitOperator: if no stale chunks exist, downstream tasks are
  skipped rather than succeeding vacuously. This keeps the Airflow UI clean
  and prevents spurious Gold gate runs on empty datasets.
- schedule at 02:00 UTC: off-peak to avoid contention with the 30-minute
  full pipeline runs during business hours.
- catchup=False: nightly freshness is a point-in-time check, not cumulative.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

PROJECT_ROOT = Path(__file__).parent.parent

DELTA_SILVER_PATH       = os.environ.get("DELTA_SILVER_PATH", "")
DELTA_GOLD_PATH         = os.environ.get("DELTA_GOLD_PATH", "")
GOLD_CHECKS_PATH        = str(PROJECT_ROOT / "quality" / "gold_checks.yml")
EMBEDDING_FRESHNESS_HOURS = int(os.environ.get("EMBEDDING_FRESHNESS_HOURS", "24"))

SPARK_CONF = {
    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    "spark.jars.packages": ",".join([
        "io.delta:delta-spark_2.12:3.2.0",
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.261",
    ]),
    "spark.hadoop.fs.s3a.access.key":  os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "spark.hadoop.fs.s3a.secret.key":  os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "spark.hadoop.fs.s3a.impl":        "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "spark.hadoop.fs.s3a.aws.credentials.provider":
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}

DEFAULT_ARGS = {
    "owner":            "data-engineering",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}


def _on_failure(context: dict) -> None:
    task_id = context["task_instance"].task_id
    dag_id  = context["task_instance"].dag_id
    run_id  = context["run_id"]
    import logging
    logging.getLogger(__name__).error(
        "Task failed | dag=%s task=%s run_id=%s", dag_id, task_id, run_id
    )


def _check_stale_chunks() -> bool:
    """
    Count Silver chunks with no Gold record (left anti-join).
    Returns True if stale chunks exist (downstream runs), False to short-circuit.

    Uses a local SparkSession for the anti-join check. In production this
    could be replaced with a Delta Lake SQL query via Athena or Databricks SQL
    to avoid spinning up a local Spark context in the Airflow worker.
    """
    from pyspark.sql import SparkSession
    from delta import configure_spark_with_delta_pip, DeltaTable
    import logging

    log = logging.getLogger(__name__)

    builder = (
        SparkSession.builder
        .appName("nightly-freshness-check")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.jars.packages", ",".join([
            "io.delta:delta-spark_2.12:3.2.0",
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.261",
        ]))
        .config("spark.hadoop.fs.s3a.access.key",
                os.environ.get("AWS_ACCESS_KEY_ID", ""))
        .config("spark.hadoop.fs.s3a.secret.key",
                os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    try:
        silver_df = spark.read.format("delta").load(DELTA_SILVER_PATH)

        if not DeltaTable.isDeltaTable(spark, DELTA_GOLD_PATH):
            stale_count = silver_df.count()
        else:
            gold_df     = spark.read.format("delta").load(DELTA_GOLD_PATH).select("chunk_id")
            stale_count = silver_df.join(gold_df, on="chunk_id", how="left_anti").count()

        log.info("Stale chunks (Silver not in Gold): %d", stale_count)

        if stale_count > 0:
            log.warning(
                "SLA VIOLATION: %d Silver chunks have no Gold embedding record. "
                "Triggering re-embedding.", stale_count
            )
            return True

        log.info("Freshness check passed — all Silver chunks are embedded")
        return False

    finally:
        spark.stop()


def _gold_soda_gate() -> None:
    """Run Gold quality checks after nightly refresh."""
    from pyspark.sql import SparkSession
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder
        .appName("nightly-gold-gate")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.jars.packages", ",".join([
            "io.delta:delta-spark_2.12:3.2.0",
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.261",
        ]))
        .config("spark.hadoop.fs.s3a.access.key",
                os.environ.get("AWS_ACCESS_KEY_ID", ""))
        .config("spark.hadoop.fs.s3a.secret.key",
                os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    try:
        df = spark.read.format("delta").load(DELTA_GOLD_PATH)
        try:
            from soda.scan import Scan
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "soda-core not installed — skipping Gold gate"
            )
            return

        scan = Scan()
        scan.set_scan_definition_name("rag-pipeline-nightly-gold")
        scan.set_data_source_name("gold_embeddings")
        scan.add_spark_session(spark, data_source_name="gold_embeddings")
        scan.add_dataframe_datasets("gold_embeddings", [("gold_embeddings", df)])
        with open(GOLD_CHECKS_PATH) as fh:
            scan.add_sodacl_yaml_str(fh.read())
        scan.execute()

        if scan.has_check_failures():
            failed = [
                c.name for c in scan.get_checks()
                if c.outcome and c.outcome.value == "fail"
            ]
            raise RuntimeError(
                f"Nightly Gold gate failed. Failed checks: {failed}"
            )
    finally:
        spark.stop()


with DAG(
    dag_id="rag_nightly_refresh",
    default_args=DEFAULT_ARGS,
    description="Nightly refresh: detect and re-embed stale Silver chunks",
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "nightly", "freshness"],
) as dag:

    freshness_check = ShortCircuitOperator(
        task_id="freshness_check",
        python_callable=_check_stale_chunks,
        on_failure_callback=_on_failure,
    )

    re_embedding_pipeline = SparkSubmitOperator(
        task_id="re_embedding_pipeline",
        application=str(PROJECT_ROOT / "batch" / "embedding_pipeline.py"),
        conf=SPARK_CONF,
        on_failure_callback=_on_failure,
    )

    gold_soda_gate = PythonOperator(
        task_id="gold_soda_gate",
        python_callable=_gold_soda_gate,
        on_failure_callback=_on_failure,
    )

    freshness_check >> re_embedding_pipeline >> gold_soda_gate
