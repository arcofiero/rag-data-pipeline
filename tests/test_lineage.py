"""
Unit tests for lineage/emitter.py — Day 9.
No Spark, no real HTTP calls.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_emitter(url="http://localhost:5000", job="test.job", run_id="run-001"):
    os.environ["OPENLINEAGE_URL"] = url
    from lineage.emitter import LineageEmitter
    return LineageEmitter(
        job_name=job,
        inputs=[("delta", "/tmp/bronze")],
        outputs=[("delta", "/tmp/silver")],
        run_id=run_id,
    )


def _posted_payload(mock_post) -> dict:
    return mock_post.call_args[1]["json"]


# ─── tests ───────────────────────────────────────────────────────────────────

def test_emitter_noop_when_no_url(monkeypatch):
    monkeypatch.delenv("OPENLINEAGE_URL", raising=False)
    # reload module so constant is re-read
    import importlib
    import lineage.emitter as mod
    monkeypatch.setattr(mod, "OPENLINEAGE_URL", "")

    emitter = mod.LineageEmitter(
        job_name="test.job",
        inputs=[("delta", "/tmp/bronze")],
        outputs=[("delta", "/tmp/silver")],
        run_id="run-noop",
    )
    with patch("requests.post") as mock_post:
        emitter.emit_start()
        mock_post.assert_not_called()


def test_emit_start_posts_correct_event(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setattr(mod, "OPENLINEAGE_URL", "http://localhost:5000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    emitter = mod.LineageEmitter(
        job_name="spark.bronze_write",
        inputs=[("kafka", "broker/raw-documents")],
        outputs=[("delta", "/tmp/bronze")],
        run_id="run-abc",
    )
    with patch("requests.post", return_value=mock_resp) as mock_post:
        emitter.emit_start()
        mock_post.assert_called_once()
        payload = _posted_payload(mock_post)

    assert payload["eventType"] == "START"
    assert payload["job"]["name"] == "spark.bronze_write"
    assert payload["run"]["runId"] == "run-abc"


def test_emit_complete_carries_row_count(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setattr(mod, "OPENLINEAGE_URL", "http://localhost:5000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    emitter = mod.LineageEmitter(
        job_name="spark.silver_transform",
        inputs=[("delta", "/tmp/bronze")],
        outputs=[("delta", "/tmp/silver")],
        run_id="run-def",
    )
    with patch("requests.post", return_value=mock_resp) as mock_post:
        emitter.emit_complete(row_count=42)
        payload = _posted_payload(mock_post)

    assert payload["eventType"] == "COMPLETE"
    assert payload["run"]["facets"]["outputStatistics"]["rowCount"] == 42


def test_emit_fail_carries_error_message(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setattr(mod, "OPENLINEAGE_URL", "http://localhost:5000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    emitter = mod.LineageEmitter(
        job_name="spark.gold_embed",
        inputs=[("delta", "/tmp/silver")],
        outputs=[("delta", "/tmp/gold")],
        run_id="run-fail",
    )
    with patch("requests.post", return_value=mock_resp) as mock_post:
        emitter.emit_fail(error=ValueError("something broke"))
        payload = _posted_payload(mock_post)

    assert payload["eventType"] == "FAIL"
    assert "something broke" in payload["run"]["facets"]["errorMessage"]["message"]


def test_emit_does_not_raise_on_connection_error(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setattr(mod, "OPENLINEAGE_URL", "http://localhost:5000")

    emitter = mod.LineageEmitter(
        job_name="spark.bronze_write",
        inputs=[("kafka", "broker/topic")],
        outputs=[("delta", "/tmp/bronze")],
        run_id="run-conn",
    )
    with patch("requests.post", side_effect=ConnectionError("refused")):
        emitter.emit_start()   # must not raise


def test_bronze_emitter_factory(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setenv("DELTA_BRONZE_PATH",          "/tmp/bronze")
    monkeypatch.setenv("KAFKA_RAW_TOPIC",            "raw-documents")
    monkeypatch.setenv("CONFLUENT_BOOTSTRAP_SERVERS", "pkc-abc.confluent.cloud:9092")

    emitter = mod.bronze_emitter()
    assert emitter.job_name        == "spark.bronze_write"
    assert emitter._inputs[0]["namespace"]  == "kafka"
    assert emitter._outputs[0]["namespace"] == "delta"


def test_silver_emitter_factory(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setenv("DELTA_BRONZE_PATH", "/tmp/bronze")
    monkeypatch.setenv("DELTA_SILVER_PATH", "/tmp/silver")

    emitter = mod.silver_emitter()
    assert emitter.job_name        == "spark.silver_transform"
    assert emitter._inputs[0]["namespace"]  == "delta"
    assert emitter._inputs[0]["name"]       == "/tmp/bronze"
    assert emitter._outputs[0]["namespace"] == "delta"
    assert emitter._outputs[0]["name"]      == "/tmp/silver"


def test_gold_emitter_factory(monkeypatch):
    import lineage.emitter as mod
    monkeypatch.setenv("DELTA_SILVER_PATH",   "/tmp/silver")
    monkeypatch.setenv("DELTA_GOLD_PATH",     "/tmp/gold")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-index")

    emitter = mod.gold_emitter()
    assert emitter.job_name == "spark.gold_embed"
    assert len(emitter._outputs) == 2
    namespaces = {o["namespace"] for o in emitter._outputs}
    assert "delta"   in namespaces
    assert "pinecone" in namespaces
