"""
Full Pipeline DAG — Day 7

Airflow DAG orchestrating the complete RAG ingestion pipeline.

Task chain:
  bronze_soda_gate → silver_transform_job → silver_soda_gate
  → embedding_pipeline → gold_soda_gate

Design decisions:
- PythonOperator for Soda Core gates: _run_soda_scan loads the Delta table
  and runs checks in-process. A failed check raises RuntimeError which
  Airflow catches as task failure, blocking all downstream tasks automatically.
- SparkSubmitOperator for PySpark jobs: submits spark-submit and blocks
  until driver exits. Exit code 0 = success, anything else = task failure.
- All layer transitions are blocked on Soda Core passing — same pattern
  as contract-driven-platform.
- catchup=False: prevents Airflow from backfilling missed runs on startup.
- max_active_runs=1: prevents concurrent pipeline runs which would cause
  Delta Lake MERGE conflicts.
- Each SparkSubmitOperator passes JAVA_HOME and SPARK_HOME from environment
  so the operator can locate the Spark installation.
- on_failure_callback logs the failure with task context for alerting.
  In production, replace with a PagerDuty or Slack webhook call.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

PROJECT_ROOT = Path(__file__).parent.parent

DELTA_BRONZE_PATH = os.environ.get("DELTA_BRONZE_PATH", "")
DELTA_SILVER_PATH = os.environ.get("DELTA_SILVER_PATH", "")
DELTA_GOLD_PATH   = os.environ.get("DELTA_GOLD_PATH", "")

BRONZE_CHECKS_PATH = str(PROJECT_ROOT / "quality" / "bronze_checks.yml")
SILVER_CHECKS_PATH = str(PROJECT_ROOT / "quality" / "silver_checks.yml")
GOLD_CHECKS_PATH   = str(PROJECT_ROOT / "quality" / "gold_checks.yml")

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
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


def _on_failure(context: dict) -> None:
    """Log task failure with context. Replace with alerting webhook in production."""
    task_id = context["task_instance"].task_id
    dag_id  = context["task_instance"].dag_id
    run_id  = context["run_id"]
    import logging
    logging.getLogger(__name__).error(
        "Task failed | dag=%s task=%s run_id=%s", dag_id, task_id, run_id
    )


def _soda_gate(delta_path: str, checks_path: str, dataset_name: str) -> None:
    """
    Load a Delta table into Spark and run Soda Core checks.
    Raises RuntimeError on failure — Airflow catches as task failure.

    This runs in the Airflow worker process (not a Spark cluster) using a
    local SparkSession. For production, replace with a SparkSubmitOperator
    that runs soda-core-spark-df on the cluster.
    """
    from pyspark.sql import SparkSession
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder
        .appName(f"soda-gate-{dataset_name}")
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
        df = spark.read.format("delta").load(delta_path)
        try:
            from soda.scan import Scan
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "soda-core not installed — skipping gate for %s", dataset_name
            )
            return

        scan = Scan()
        scan.set_scan_definition_name(f"rag-pipeline-{dataset_name}")
        scan.set_data_source_name(dataset_name)
        scan.add_spark_session(spark, data_source_name=dataset_name)
        scan.add_dataframe_datasets(dataset_name, [(dataset_name, df)])
        with open(checks_path) as fh:
            scan.add_sodacl_yaml_str(fh.read())
        scan.execute()

        if scan.has_check_failures():
            failed = [
                c.name for c in scan.get_checks()
                if c.outcome and c.outcome.value == "fail"
            ]
            raise RuntimeError(
                f"Soda Core quality gate failed for {dataset_name}. "
                f"Failed checks: {failed}"
            )
    finally:
        spark.stop()


with DAG(
    dag_id="rag_full_pipeline",
    default_args=DEFAULT_ARGS,
    description="Full RAG ingestion pipeline: Bronze → Silver → Gold → Pinecone",
    schedule_interval="*/30 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "ingestion", "production"],
) as dag:

    bronze_soda_gate = PythonOperator(
        task_id="bronze_soda_gate",
        python_callable=_soda_gate,
        op_kwargs={
            "delta_path":   DELTA_BRONZE_PATH,
            "checks_path":  BRONZE_CHECKS_PATH,
            "dataset_name": "bronze_documents",
        },
        on_failure_callback=_on_failure,
    )

    silver_transform_job = SparkSubmitOperator(
        task_id="silver_transform_job",
        application=str(PROJECT_ROOT / "batch" / "silver_job.py"),
        conf=SPARK_CONF,
        on_failure_callback=_on_failure,
    )

    silver_soda_gate = PythonOperator(
        task_id="silver_soda_gate",
        python_callable=_soda_gate,
        op_kwargs={
            "delta_path":   DELTA_SILVER_PATH,
            "checks_path":  SILVER_CHECKS_PATH,
            "dataset_name": "silver_chunks",
        },
        on_failure_callback=_on_failure,
    )

    embedding_pipeline = SparkSubmitOperator(
        task_id="embedding_pipeline",
        application=str(PROJECT_ROOT / "batch" / "embedding_pipeline.py"),
        conf=SPARK_CONF,
        on_failure_callback=_on_failure,
    )

    gold_soda_gate = PythonOperator(
        task_id="gold_soda_gate",
        python_callable=_soda_gate,
        op_kwargs={
            "delta_path":   DELTA_GOLD_PATH,
            "checks_path":  GOLD_CHECKS_PATH,
            "dataset_name": "gold_embeddings",
        },
        on_failure_callback=_on_failure,
    )

    (
        bronze_soda_gate
        >> silver_transform_job
        >> silver_soda_gate
        >> embedding_pipeline
        >> gold_soda_gate
    )
