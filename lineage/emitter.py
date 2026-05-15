"""
OpenLineage emitter for the RAG pipeline — Day 9.

Emits START / COMPLETE / FAIL run events to a Marquez-compatible
OpenLineage backend. No-ops silently when OPENLINEAGE_URL is unset so
the pipeline runs unmodified in environments without a lineage server.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

OPENLINEAGE_URL       = os.environ.get("OPENLINEAGE_URL", "")
OPENLINEAGE_API_KEY   = os.environ.get("OPENLINEAGE_API_KEY", "")
OPENLINEAGE_NAMESPACE = os.environ.get("OPENLINEAGE_NAMESPACE", "rag-pipeline")
PIPELINE_VERSION      = os.environ.get("PIPELINE_VERSION", "1.0.0")

NS_DELTA   = "delta"
NS_KAFKA   = "kafka"
NS_PINECONE = "pinecone"

_PRODUCER   = "https://github.com/arcofiero/rag-data-pipeline"
_SCHEMA_URL = "https://openlineage.io/spec/1-0-5/OpenLineage.json"


def _build_dataset(namespace: str, name: str) -> dict:
    return {
        "_producer":   _PRODUCER,
        "_schemaURL":  _SCHEMA_URL,
        "namespace":   namespace,
        "name":        name,
    }


class LineageEmitter:
    def __init__(
        self,
        job_name: str,
        inputs:   list[tuple[str, str]],
        outputs:  list[tuple[str, str]],
        run_id:   str | None = None,
    ) -> None:
        self.job_name  = job_name
        self.run_id    = run_id or str(uuid.uuid4())
        self._enabled  = bool(OPENLINEAGE_URL)

        if not self._enabled:
            logger.warning(
                "OPENLINEAGE_URL not set — lineage events will not be emitted"
            )

        self._inputs  = [_build_dataset(ns, name) for ns, name in inputs]
        self._outputs = [_build_dataset(ns, name) for ns, name in outputs]

    def emit_start(self) -> None:
        self._emit("START")
        logger.info(
            "Lineage START emitted | job={} run_id={}", self.job_name, self.run_id
        )

    def emit_complete(self, row_count: int | None = None) -> None:
        run_facets: dict = {}
        if row_count is not None:
            run_facets["outputStatistics"] = {
                "_producer":  _PRODUCER,
                "_schemaURL": _SCHEMA_URL,
                "rowCount":   row_count,
            }
        self._emit("COMPLETE", run_facets=run_facets)
        logger.info(
            "Lineage COMPLETE emitted | job={} run_id={} rows={}",
            self.job_name, self.run_id, row_count,
        )

    def emit_fail(self, error: BaseException | None = None) -> None:
        error_str = str(error)[:2000] if error is not None else ""
        run_facets = {
            "errorMessage": {
                "_producer":          _PRODUCER,
                "_schemaURL":         _SCHEMA_URL,
                "message":            error_str,
                "programmingLanguage": "Python",
            }
        }
        self._emit("FAIL", run_facets=run_facets)
        logger.warning(
            "Lineage FAIL emitted | job={} run_id={} error={}",
            self.job_name, self.run_id, error_str[:200],
        )

    def _build_event(self, state: str, run_facets: dict | None = None) -> dict:
        facets: dict = {
            "pipeline": {
                "_producer":  _PRODUCER,
                "_schemaURL": _SCHEMA_URL,
                "version":    PIPELINE_VERSION,
                "host":       socket.gethostname(),
            }
        }
        if run_facets:
            facets.update(run_facets)

        return {
            "_producer":  _PRODUCER,
            "_schemaURL": _SCHEMA_URL,
            "eventTime":  datetime.now(timezone.utc).isoformat(),
            "eventType":  state,
            "run": {
                "runId":  self.run_id,
                "facets": facets,
            },
            "job": {
                "namespace": OPENLINEAGE_NAMESPACE,
                "name":      self.job_name,
                "facets": {
                    "jobType": {
                        "_producer":  _PRODUCER,
                        "_schemaURL": _SCHEMA_URL,
                        "jobType":    "BATCH",
                    }
                },
            },
            "inputs":  self._inputs,
            "outputs": self._outputs,
        }

    def _emit(self, state: str, run_facets: dict | None = None) -> None:
        if not self._enabled:
            return

        event = self._build_event(state, run_facets)
        url   = f"{OPENLINEAGE_URL.rstrip('/')}/api/v1/lineage"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if OPENLINEAGE_API_KEY:
            headers["Authorization"] = f"Bearer {OPENLINEAGE_API_KEY}"

        try:
            resp = requests.post(url, json=event, headers=headers, timeout=10)
            resp.raise_for_status()
            logger.debug(
                "Lineage event posted | state={} job={} status={}",
                state, self.job_name, resp.status_code,
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Lineage emit failed (connection error) | job={} error={}", self.job_name, exc
            )
        except requests.exceptions.Timeout as exc:
            logger.warning(
                "Lineage emit failed (timeout) | job={} error={}", self.job_name, exc
            )
        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "Lineage emit failed (HTTP error) | job={} error={}", self.job_name, exc
            )
        except Exception as exc:
            logger.warning(
                "Lineage emit failed (unexpected) | job={} error={}", self.job_name, exc
            )


# ─── Factory functions ────────────────────────────────────────────────────────

def bronze_emitter(run_id: str | None = None) -> LineageEmitter:
    bootstrap  = os.environ.get("CONFLUENT_BOOTSTRAP_SERVERS", "")
    raw_topic  = os.environ.get("KAFKA_RAW_TOPIC", "raw-documents")
    bronze_path = os.environ.get("DELTA_BRONZE_PATH", "")
    return LineageEmitter(
        job_name="spark.bronze_write",
        inputs=[(NS_KAFKA, f"{bootstrap}/{raw_topic}")],
        outputs=[(NS_DELTA, bronze_path)],
        run_id=run_id,
    )


def silver_emitter(run_id: str | None = None) -> LineageEmitter:
    bronze_path = os.environ.get("DELTA_BRONZE_PATH", "")
    silver_path = os.environ.get("DELTA_SILVER_PATH", "")
    return LineageEmitter(
        job_name="spark.silver_transform",
        inputs=[(NS_DELTA, bronze_path)],
        outputs=[(NS_DELTA, silver_path)],
        run_id=run_id,
    )


def gold_emitter(run_id: str | None = None) -> LineageEmitter:
    silver_path  = os.environ.get("DELTA_SILVER_PATH", "")
    gold_path    = os.environ.get("DELTA_GOLD_PATH", "")
    index_name   = os.environ.get("PINECONE_INDEX_NAME", "")
    return LineageEmitter(
        job_name="spark.gold_embed",
        inputs=[(NS_DELTA, silver_path)],
        outputs=[(NS_DELTA, gold_path), (NS_PINECONE, index_name)],
        run_id=run_id,
    )
