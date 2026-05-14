"""
Unit tests for the Gold embedding pipeline — Day 6.
Tests cover _embed_and_upsert_partition with mocked OpenAI and Pinecone.
No SparkSession required.
"""

from __future__ import annotations

import os
from collections import namedtuple
from typing import List
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("PINECONE_API_KEY", "test-pc-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("EMBED_BATCH_SIZE", "512")
os.environ.setdefault("PINECONE_BATCH_SIZE", "100")
os.environ.setdefault("DELTA_SILVER_PATH", "s3a://test/silver")
os.environ.setdefault("DELTA_GOLD_PATH", "s3a://test/gold")

from batch.embedding_pipeline import EMBED_BATCH_SIZE, _embed_and_upsert_partition

SilverRow = namedtuple(
    "SilverRow",
    ["chunk_id", "document_id", "source", "content", "ingested_at",
     "chunk_index", "word_count", "char_count", "language", "processed_at"],
)


def _make_row(chunk_id="chunk-abc", content="This is a meaningful test chunk with enough content.", source="PDF", document_id="doc-001", ingested_at="2024-01-15T10:00:00+00:00"):
    return SilverRow(chunk_id=chunk_id, document_id=document_id, source=source, content=content, ingested_at=ingested_at, chunk_index=0, word_count=10, char_count=len(content), language="en", processed_at="2024-01-15T10:05:00+00:00")


def _fake_embedding(dim=1536):
    return [0.01] * dim


def _mock_openai_response(n):
    response = MagicMock()
    response.data = [MagicMock(embedding=_fake_embedding()) for _ in range(n)]
    return response


def _run_partition(rows, openai_mock=None, pinecone_mock=None):
    if openai_mock is None:
        openai_mock = MagicMock()
        openai_mock.return_value.embeddings.create.return_value = _mock_openai_response(len(rows))
    if pinecone_mock is None:
        pinecone_mock = MagicMock()
        pinecone_mock.return_value.Index.return_value.upsert.return_value = None
    with patch("openai.OpenAI", openai_mock), patch("pinecone.Pinecone", pinecone_mock):
        return list(_embed_and_upsert_partition(iter(rows)))


class TestBasicCorrectness:
    def test_empty_partition_yields_nothing(self):
        assert _run_partition([]) == []

    def test_single_row_yields_one_gold_record(self):
        assert len(_run_partition([_make_row()])) == 1

    def test_multiple_rows_yield_correct_count(self):
        rows = [_make_row(chunk_id=f"chunk-{i}") for i in range(5)]
        assert len(_run_partition(rows)) == 5

    def test_vector_id_equals_chunk_id(self):
        result = _run_partition([_make_row(chunk_id="my-chunk-id")])
        assert result[0]["chunk_id"] == "my-chunk-id"
        assert result[0]["vector_id"] == "my-chunk-id"

    def test_chunk_id_equals_vector_id_for_all_rows(self):
        rows = [_make_row(chunk_id=f"chunk-{i}") for i in range(10)]
        for rec in _run_partition(rows):
            assert rec["chunk_id"] == rec["vector_id"]


class TestMetadataFields:
    def test_document_id_populated(self):
        assert _run_partition([_make_row(document_id="doc-xyz")])[0]["document_id"] == "doc-xyz"

    def test_source_populated(self):
        assert _run_partition([_make_row(source="WEB")])[0]["source"] == "WEB"

    def test_model_populated(self):
        assert _run_partition([_make_row()])[0]["model"] == "text-embedding-3-small"

    def test_ingested_at_preserved(self):
        result = _run_partition([_make_row(ingested_at="2024-06-01T00:00:00+00:00")])
        assert result[0]["ingested_at"] == "2024-06-01T00:00:00+00:00"

    def test_embedded_at_populated(self):
        result = _run_partition([_make_row()])
        assert result[0]["embedded_at"] is not None and len(result[0]["embedded_at"]) > 0

    def test_embedded_date_is_yyyy_mm_dd(self):
        parts = _run_partition([_make_row()])[0]["embedded_date"].split("-")
        assert len(parts) == 3 and len(parts[0]) == 4


class TestOpenAICallMechanics:
    def test_openai_called_once_for_small_batch(self):
        rows = [_make_row(chunk_id=f"c-{i}") for i in range(3)]
        openai_mock = MagicMock()
        openai_mock.return_value.embeddings.create.return_value = _mock_openai_response(3)
        _run_partition(rows, openai_mock=openai_mock)
        assert openai_mock.return_value.embeddings.create.call_count == 1

    def test_content_truncated_at_8000_chars(self):
        row = _make_row(content="x" * 10000)
        openai_mock = MagicMock()
        openai_mock.return_value.embeddings.create.return_value = _mock_openai_response(1)
        with patch("openai.OpenAI", openai_mock), patch("pinecone.Pinecone", MagicMock()):
            list(_embed_and_upsert_partition(iter([row])))
        call_args = openai_mock.return_value.embeddings.create.call_args
        sent_texts = call_args[1].get("input") or call_args[0][0]
        assert len(sent_texts[0]) == 8000

    def test_openai_failure_yields_no_gold_records(self):
        openai_mock = MagicMock()
        openai_mock.return_value.embeddings.create.side_effect = Exception("API error")
        assert _run_partition([_make_row()], openai_mock=openai_mock) == []

    def test_openai_max_retries_set(self):
        openai_mock = MagicMock()
        openai_mock.return_value.embeddings.create.return_value = _mock_openai_response(1)
        _run_partition([_make_row()], openai_mock=openai_mock)
        openai_mock.assert_called_once_with(api_key="test-key", max_retries=3)


class TestPineconeUpsertMechanics:
    def test_pinecone_upsert_called(self):
        pc_mock = MagicMock()
        _run_partition([_make_row()], pinecone_mock=pc_mock)
        assert pc_mock.return_value.Index.return_value.upsert.called

    def test_pinecone_vector_id_equals_chunk_id(self):
        pc_mock = MagicMock()
        upserted = []
        pc_mock.return_value.Index.return_value.upsert.side_effect = lambda vectors: upserted.extend(vectors)
        _run_partition([_make_row(chunk_id="target-chunk")], pinecone_mock=pc_mock)
        assert upserted[0]["id"] == "target-chunk"

    def test_pinecone_metadata_contains_required_fields(self):
        pc_mock = MagicMock()
        upserted = []
        pc_mock.return_value.Index.return_value.upsert.side_effect = lambda vectors: upserted.extend(vectors)
        _run_partition([_make_row()], pinecone_mock=pc_mock)
        meta = upserted[0]["metadata"]
        for field in ["chunk_id", "document_id", "source", "ingested_at", "embedded_at", "model"]:
            assert field in meta

    def test_pinecone_failure_excludes_chunk_from_gold(self):
        pc_mock = MagicMock()
        pc_mock.return_value.Index.return_value.upsert.side_effect = Exception("Pinecone unavailable")
        assert _run_partition([_make_row()], pinecone_mock=pc_mock) == []


class TestBatchHandling:
    def test_large_partition_triggers_multiple_api_calls(self):
        n = EMBED_BATCH_SIZE + 10
        rows = [_make_row(chunk_id=f"c-{i}") for i in range(n)]
        openai_mock = MagicMock()
        call_count = [0]
        def create_response(**kwargs):
            call_count[0] += 1
            return _mock_openai_response(len(kwargs["input"]))
        openai_mock.return_value.embeddings.create.side_effect = create_response
        result = _run_partition(rows, openai_mock=openai_mock)
        assert call_count[0] == 2
        assert len(result) == n

    def test_gold_record_count_matches_successful_embeddings(self):
        rows = [_make_row(chunk_id=f"c-{i}") for i in range(7)]
        assert len(_run_partition(rows)) == 7
