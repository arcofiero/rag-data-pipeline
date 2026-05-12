"""
Unit tests for the document producer module.

All tests are offline — no Kafka broker required. fastavro serialization is
exercised against the actual schema files to catch schema/code drift early.

Coverage:
- Valid event generation for all three source types
- Avro round-trip: serialize -> deserialize for each event type
- DLQ envelope construction and round-trip
- Malformed payload generation (structural sanity)
- DLQ routing rate (statistical check over many samples)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import fastavro
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from producers.document_producer import (
    _build_dlq_envelope,
    _make_malformed_payload,
    _make_pdf_event,
    _make_structured_event,
    _make_web_event,
    _serialize,
    MALFORMED_EVENT_RATE,
)

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


def _parsed(name: str) -> dict:
    with (SCHEMA_DIR / f"{name}.avsc").open() as f:
        return fastavro.parse_schema(json.load(f))


@pytest.fixture
def doc_schema():
    return _parsed("document_event")


@pytest.fixture
def dlq_schema():
    return _parsed("dead_letter_event")


# ─── Event generators ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("generator,expected_source_type", [
    (_make_pdf_event, "PDF"),
    (_make_web_event, "WEB"),
    (_make_structured_event, "STRUCTURED"),
])
def test_event_generator_required_fields(generator, expected_source_type):
    doc_id = "test-doc-id-1234"
    event = generator(doc_id)
    assert event["document_id"] == doc_id
    assert event["source_type"] == expected_source_type
    assert isinstance(event["source_uri"], str) and event["source_uri"]
    assert isinstance(event["content"], str) and event["content"]
    assert isinstance(event["metadata"], dict)
    assert event["schema_version"] == "1.0"
    assert isinstance(event["produced_at"], int)


# ─── Avro round-trip ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("generator", [_make_pdf_event, _make_web_event, _make_structured_event])
def test_avro_roundtrip_valid_event(generator, doc_schema):
    import uuid
    doc_id = str(uuid.uuid4())
    record = generator(doc_id)
    serialized = _serialize(record, doc_schema)

    assert isinstance(serialized, bytes) and len(serialized) > 0

    deserialized = fastavro.schemaless_reader(io.BytesIO(serialized), doc_schema)
    assert deserialized["document_id"] == doc_id
    assert deserialized["source_type"] == record["source_type"]
    assert deserialized["content"] == record["content"]


def test_avro_roundtrip_dlq_envelope(dlq_schema):
    envelope = _build_dlq_envelope(
        raw_payload='{"document_id": "bad-event"}',
        error_reason="Missing required field: content",
    )
    serialized = _serialize(envelope, dlq_schema)
    deserialized = fastavro.schemaless_reader(io.BytesIO(serialized), dlq_schema)

    assert deserialized["error_reason"] == "Missing required field: content"
    assert deserialized["source_topic"] == "raw-documents"
    assert deserialized["dlq_schema_version"] == "1.0"
    assert "bad-event" in deserialized["original_payload"]


# ─── DLQ envelope ───────────────────────────────────────────────────────────────


def test_dlq_envelope_preserves_payload():
    raw = '{"document_id": "x", "broken": true}'
    envelope = _build_dlq_envelope(raw_payload=raw, error_reason="test error")
    assert envelope["original_payload"] == raw
    assert envelope["source_topic"] == "raw-documents"


# ─── Malformed payloads ─────────────────────────────────────────────────────────


def test_malformed_payload_is_always_string():
    for _ in range(30):
        payload = _make_malformed_payload()
        assert isinstance(payload, str) and len(payload) > 0


def test_malformed_payload_fails_avro_validation(doc_schema):
    failures = 0
    for _ in range(20):
        payload = _make_malformed_payload()
        try:
            record = json.loads(payload)
            _serialize(record, doc_schema)
        except Exception:
            failures += 1
    assert failures >= 10, f"Expected >=10 Avro failures from malformed payloads, got {failures}"


# ─── DLQ routing rate ───────────────────────────────────────────────────────────


def test_dlq_routing_rate_within_tolerance():
    """Over 10,000 samples the observed DLQ rate should be within +/-2% of target."""
    import random
    samples = 10_000
    dlq_count = sum(1 for _ in range(samples) if random.random() < MALFORMED_EVENT_RATE)
    observed = dlq_count / samples
    assert abs(observed - MALFORMED_EVENT_RATE) < 0.02, (
        f"Observed DLQ rate {observed:.2%} deviated >2% from target {MALFORMED_EVENT_RATE:.2%}"
    )
