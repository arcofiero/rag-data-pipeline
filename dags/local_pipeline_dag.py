"""
Local Pipeline DAG — Day 10 validation

PythonOperator version of the full pipeline DAG for local testing.
Calls the pipeline Python functions directly instead of spark-submit,
so it runs without a Spark cluster or Airflow workers.

Test with:
    airflow dags test rag_local_pipeline 2024-01-01

Production DAG (SparkSubmitOperator) is in full_pipeline_dag.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_ARGS = {
    "owner":            "data-engineering",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}


def _run_silver(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    from batch.silver_job import run_silver_job
    run_silver_job()


def _run_embedding(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    from batch.embedding_pipeline import run_embedding_pipeline
    run_embedding_pipeline()


def _run_bronze_soda(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    from batch.silver_job import _build_spark, _run_soda_scan
    spark = _build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    bronze_path   = os.environ.get("DELTA_BRONZE_PATH", "/tmp/rag-delta/bronze")
    checks_path   = PROJECT_ROOT / "quality" / "bronze_checks.yml"
    df = spark.read.format("delta").load(bronze_path)
    _run_soda_scan(spark, df, checks_path, "bronze_documents")
    spark.stop()


def _run_silver_soda(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    from batch.silver_job import _build_spark, _run_soda_scan
    spark = _build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    silver_path = os.environ.get("DELTA_SILVER_PATH", "/tmp/rag-delta/silver")
    checks_path = PROJECT_ROOT / "quality" / "silver_checks.yml"
    df = spark.read.format("delta").load(silver_path)
    _run_soda_scan(spark, df, checks_path, "silver_chunks")
    spark.stop()


def _run_gold_soda(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    from batch.embedding_pipeline import _build_spark, _run_soda_scan
    spark = _build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    gold_path   = os.environ.get("DELTA_GOLD_PATH", "/tmp/rag-delta/gold")
    checks_path = PROJECT_ROOT / "quality" / "gold_checks.yml"
    df = spark.read.format("delta").load(gold_path)
    _run_soda_scan(spark, df, checks_path, "gold_embeddings")
    spark.stop()


with DAG(
    dag_id="rag_local_pipeline",
    default_args=DEFAULT_ARGS,
    description="Local PythonOperator version of the full RAG pipeline (no Spark cluster needed)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "local", "testing"],
) as dag:

    bronze_gate = PythonOperator(
        task_id="bronze_soda_gate",
        python_callable=_run_bronze_soda,
    )

    silver_job = PythonOperator(
        task_id="silver_transform_job",
        python_callable=_run_silver,
    )

    silver_gate = PythonOperator(
        task_id="silver_soda_gate",
        python_callable=_run_silver_soda,
    )

    embed_job = PythonOperator(
        task_id="embedding_pipeline",
        python_callable=_run_embedding,
    )

    gold_gate = PythonOperator(
        task_id="gold_soda_gate",
        python_callable=_run_gold_soda,
    )

    bronze_gate >> silver_job >> silver_gate >> embed_job >> gold_gate
