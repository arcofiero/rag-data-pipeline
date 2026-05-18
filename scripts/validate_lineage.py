"""
Lineage Graph Validation Script — Day 9

Connects to a running Marquez instance and validates that all expected
pipeline jobs and dataset edges are present in the lineage graph.

Usage:
    # Start Marquez first:
    #   docker compose up marquez marquez-web
    #
    # Run the full pipeline once to emit events:
    #   python batch/silver_job.py
    #   python batch/embedding_pipeline.py
    #
    # Then validate:
    python scripts/validate_lineage.py

Exit codes:
    0 — all lineage assertions passed
    1 — one or more assertions failed or Marquez unreachable
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

MARQUEZ_URL = os.getenv("OPENLINEAGE_URL", "http://localhost:5000")
NAMESPACE   = os.getenv("OPENLINEAGE_NAMESPACE", "rag-pipeline")

# Every job that must appear in the lineage graph
EXPECTED_JOBS = [
    "spark.bronze_write",
    "spark.silver_transform",
    "spark.gold_embed",
]

# Every dataset edge that must exist: (input_namespace, input_name, output_job)
EXPECTED_EDGES = [
    ("kafka",   None,   "spark.bronze_write"),    # Kafka → bronze
    ("delta",   None,   "spark.silver_transform"), # bronze → silver
    ("delta",   None,   "spark.gold_embed"),       # silver → gold
    ("pinecone", None,  "spark.gold_embed"),       # gold → pinecone (output)
]


def _get(path: str) -> dict | list | None:
    url = f"{MARQUEZ_URL.rstrip('/')}{path}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Marquez at {} — is it running?", MARQUEZ_URL)
        return None
    except requests.exceptions.HTTPError as exc:
        logger.error("Marquez HTTP error: {}", exc)
        return None


def validate_marquez_reachable() -> bool:
    data = _get("/api/v1/namespaces")
    if data is None:
        return False
    logger.info("Marquez reachable at {} | namespaces: {}", MARQUEZ_URL, len(data.get("namespaces", [])))
    return True


def validate_jobs() -> list[str]:
    """Return list of missing job names."""
    data = _get(f"/api/v1/namespaces/{NAMESPACE}/jobs")
    if data is None:
        return EXPECTED_JOBS

    found = {job["name"] for job in data.get("jobs", [])}
    missing = [j for j in EXPECTED_JOBS if j not in found]

    for job in EXPECTED_JOBS:
        if job in found:
            logger.info("  [PASS] Job found: {}", job)
        else:
            logger.error("  [FAIL] Job missing: {}", job)

    return missing


def validate_job_inputs_outputs() -> list[str]:
    """Return list of failures for input/output dataset namespaces."""
    failures = []
    for job_name in EXPECTED_JOBS:
        data = _get(f"/api/v1/namespaces/{NAMESPACE}/jobs/{job_name}")
        if data is None:
            failures.append(f"{job_name}: not found")
            continue

        input_ns  = {d.get("namespace") for d in data.get("inputs", [])}
        output_ns = {d.get("namespace") for d in data.get("outputs", [])}

        logger.info(
            "  Job '{}' | inputs={} outputs={}",
            job_name, sorted(input_ns), sorted(output_ns),
        )

        if job_name == "spark.bronze_write" and "kafka" not in input_ns:
            failures.append(f"{job_name}: missing kafka input")
        if job_name == "spark.silver_transform" and "delta" not in input_ns:
            failures.append(f"{job_name}: missing delta input")
        if job_name == "spark.gold_embed":
            if "delta" not in input_ns:
                failures.append(f"{job_name}: missing delta input")
            if "pinecone" not in output_ns:
                failures.append(f"{job_name}: missing pinecone output")

    return failures


def validate_latest_run_states() -> list[str]:
    """Return list of jobs whose latest run did not COMPLETE successfully."""
    failures = []
    for job_name in EXPECTED_JOBS:
        data = _get(f"/api/v1/namespaces/{NAMESPACE}/jobs/{job_name}/runs?limit=1")
        if data is None:
            continue
        runs = data.get("runs", [])
        if not runs:
            logger.warning("  [WARN] No runs found for job: {}", job_name)
            continue
        latest = runs[0]
        state  = latest.get("currentState", "UNKNOWN")
        if state == "COMPLETE":
            logger.info("  [PASS] Latest run COMPLETE | job={}", job_name)
        else:
            logger.error("  [FAIL] Latest run state={} | job={}", state, job_name)
            failures.append(f"{job_name}: latest run state={state}")
    return failures


def run_validation() -> int:
    logger.info("── Lineage Graph Validation ──────────────────────────")
    logger.info("Marquez: {} | namespace: {}", MARQUEZ_URL, NAMESPACE)

    if not validate_marquez_reachable():
        logger.error("Marquez unreachable. Start it with: docker compose up marquez")
        return 1

    all_failures: list[str] = []

    logger.info("\n[1] Validating expected jobs exist...")
    all_failures.extend(validate_jobs())

    logger.info("\n[2] Validating job input/output dataset edges...")
    all_failures.extend(validate_job_inputs_outputs())

    logger.info("\n[3] Validating latest run states...")
    all_failures.extend(validate_latest_run_states())

    logger.info("\n── Results ────────────────────────────────────────────")
    if all_failures:
        logger.error("FAILED — {} assertion(s) failed:", len(all_failures))
        for f in all_failures:
            logger.error("  ✗ {}", f)
        return 1

    logger.info("PASSED — all lineage assertions verified ({} jobs, graph complete)", len(EXPECTED_JOBS))
    return 0


if __name__ == "__main__":
    sys.exit(run_validation())
