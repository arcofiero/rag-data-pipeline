"""
Unit tests for batch/silver_job.py — _normalize_partition.

All tests run offline — no SparkSession required. _normalize_partition
is a pure Python generator that accepts any iterable of row-like objects.
BronzeRow namedtuple simulates a Spark Row for executor-side access.
"""

from __future__ import annotations

import json
import re
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch.silver_job import MIN_CHAR_COUNT, MIN_WORD_COUNT, _normalize_partition

BronzeRow = namedtuple(
    "BronzeRow",
    [
        "chunk_id", "document_id", "chunk_index", "content",
        "source", "source_uri", "metadata", "schema_version", "ingested_at",
    ],
)


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _make_row(**overrides) -> BronzeRow:
    defaults = {
        "chunk_id": "a" * 64,
        "document_id": "doc-001",
        "chunk_index": 0,
        "content": "This is a meaningful text chunk for testing with enough words and characters.",
        "source": "PDF",
        "source_uri": "s3://bucket/test.pdf",
        "metadata": json.dumps({"author": "Smith, J.", "page_count": "42"}),
        "schema_version": "1.0",
        "ingested_at": "2026-05-14T10:00:00+00:00",
    }
    defaults.update(overrides)
    return BronzeRow(**defaults)


def _normalize_one(row: BronzeRow) -> dict:
    return list(_normalize_partition([row]))[0]


# ─── TestPDFMetadataNormalization ───────────────────────────────────────────────


class TestPDFMetadataNormalization:
    def test_author_extracted(self):
        row = _make_row(metadata=json.dumps({"author": "Doe, A.", "page_count": "5"}))
        result = _normalize_one(row)
        assert result["author"] == "Doe, A."

    def test_page_count_as_int(self):
        row = _make_row(metadata=json.dumps({"author": "Smith", "page_count": "42"}))
        result = _normalize_one(row)
        assert result["page_count"] == 42
        assert isinstance(result["page_count"], int)

    def test_invalid_page_count_yields_none(self):
        row = _make_row(metadata=json.dumps({"author": "Smith", "page_count": "not-a-number"}))
        result = _normalize_one(row)
        assert result["page_count"] is None

    def test_missing_author_yields_none(self):
        row = _make_row(metadata=json.dumps({"page_count": "10"}))
        result = _normalize_one(row)
        assert result["author"] is None

    def test_web_fields_null_for_pdf(self):
        result = _normalize_one(_make_row(source="PDF"))
        assert result["crawled_at"] is None
        assert result["word_count_src"] is None

    def test_structured_fields_null_for_pdf(self):
        result = _normalize_one(_make_row(source="PDF"))
        assert result["table_name"] is None
        assert result["row_id"] is None


# ─── TestWEBMetadataNormalization ────────────────────────────────────────────────


class TestWEBMetadataNormalization:
    def _web_row(self, **overrides) -> BronzeRow:
        defaults = {
            "source": "WEB",
            "source_uri": "https://example.com/article",
            "metadata": json.dumps({"crawled_at": "2026-05-14T09:00:00Z", "word_count": "1500"}),
        }
        defaults.update(overrides)
        return _make_row(**defaults)

    def test_crawled_at_extracted(self):
        result = _normalize_one(self._web_row())
        assert result["crawled_at"] == "2026-05-14T09:00:00Z"

    def test_word_count_src_extracted(self):
        result = _normalize_one(self._web_row())
        assert result["word_count_src"] == "1500"

    def test_pdf_fields_null_for_web(self):
        result = _normalize_one(self._web_row())
        assert result["author"] is None
        assert result["page_count"] is None


# ─── TestStructuredMetadataNormalization ─────────────────────────────────────────


class TestStructuredMetadataNormalization:
    def _struct_row(self, **overrides) -> BronzeRow:
        defaults = {
            "source": "STRUCTURED",
            "metadata": json.dumps({"table": "knowledge_base", "row_id": "7890"}),
        }
        defaults.update(overrides)
        return _make_row(**defaults)

    def test_table_name_extracted(self):
        result = _normalize_one(self._struct_row())
        assert result["table_name"] == "knowledge_base"

    def test_row_id_extracted(self):
        result = _normalize_one(self._struct_row())
        assert result["row_id"] == "7890"


# ─── TestEnrichment ──────────────────────────────────────────────────────────────


class TestEnrichment:
    def test_word_count_computed(self):
        content = "one two three four five six seven eight nine ten"
        result = _normalize_one(_make_row(content=content))
        assert result["word_count"] == 10

    def test_char_count_computed(self):
        content = "x" * 80
        result = _normalize_one(_make_row(content=content))
        assert result["char_count"] == 80

    def test_language_is_en(self):
        result = _normalize_one(_make_row())
        assert result["language"] == "en"

    def test_processed_at_populated(self):
        result = _normalize_one(_make_row())
        assert result["processed_at"] is not None
        assert len(result["processed_at"]) > 0

    def test_processed_date_is_yyyy_mm_dd_format(self):
        result = _normalize_one(_make_row())
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result["processed_date"]), (
            f"processed_date '{result['processed_date']}' is not YYYY-MM-DD format"
        )

    def test_chunk_index_is_int(self):
        result = _normalize_one(_make_row(chunk_index="3"))
        assert isinstance(result["chunk_index"], int)
        assert result["chunk_index"] == 3

    def test_ingested_at_preserved(self):
        ingested = "2026-05-14T10:00:00+00:00"
        result = _normalize_one(_make_row(ingested_at=ingested))
        assert result["ingested_at"] == ingested


# ─── TestBoilerplateFilter ───────────────────────────────────────────────────────


class TestBoilerplateFilter:
    def test_meaningful_content_not_filtered(self):
        content = "This is a meaningful sentence about machine learning and RAG pipelines for testing."
        result = _normalize_one(_make_row(content=content))
        assert result["_filtered"] is False

    def test_short_content_filtered(self):
        # length < MIN_CHAR_COUNT
        content = "too short"
        result = _normalize_one(_make_row(content=content))
        assert result["_filtered"] is True

    def test_punctuation_soup_filtered(self):
        # length >= MIN_CHAR_COUNT, but alnum ratio = 0 < 0.5
        content = "!@#$%^&*" * 20
        result = _normalize_one(_make_row(content=content))
        assert result["_filtered"] is True

    def test_few_words_filtered(self):
        # length >= MIN_CHAR_COUNT, high alnum ratio, but word count < MIN_WORD_COUNT
        content = "word " * 5 + "x" * 30  # 55 chars, 6 tokens, 91% alnum
        result = _normalize_one(_make_row(content=content))
        assert result["_filtered"] is True

    def test_empty_content_filtered(self):
        result = _normalize_one(_make_row(content=""))
        assert result["_filtered"] is True


# ─── TestResilience ──────────────────────────────────────────────────────────────


class TestResilience:
    def test_null_metadata_does_not_raise(self):
        row = _make_row(metadata=None)
        result = _normalize_one(row)
        assert result["chunk_id"] is not None

    def test_malformed_json_does_not_raise(self):
        row = _make_row(metadata='{"key": broken json ,,,')
        result = _normalize_one(row)
        assert result["chunk_id"] is not None

    def test_empty_metadata_does_not_raise(self):
        row = _make_row(metadata="{}")
        result = _normalize_one(row)
        assert result["chunk_id"] is not None

    def test_unknown_source_passes_through(self):
        row = _make_row(source="UNKNOWN", metadata=json.dumps({}))
        result = _normalize_one(row)
        assert result["source"] == "UNKNOWN"

    def test_multiple_rows_processed(self):
        rows = [
            _make_row(chunk_id=str(i) * 64, document_id=f"doc-{i}")
            for i in range(5)
        ]
        results = list(_normalize_partition(rows))
        assert len(results) == 5
