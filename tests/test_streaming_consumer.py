"""
Unit tests for streaming/chunker.py.

All tests run offline — no SparkSession required. The chunker module has no
Spark dependencies and can be imported and exercised in plain pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming.chunker import Chunk, _make_chunk_id, chunk_text, normalize_text


# ─── normalize_text ─────────────────────────────────────────────────────────────


def test_normalize_strips_control_chars():
    text = "hello\x00world\x01end"
    result = normalize_text(text)
    assert "\x00" not in result
    assert "\x01" not in result
    assert "helloworld" in result or "hello" in result


def test_normalize_collapses_multiple_spaces():
    assert normalize_text("too   many   spaces") == "too many spaces"


def test_normalize_nfc():
    # é as NFD (e + combining acute) should become NFC single codepoint
    nfd_e = "é"  # e + combining acute accent
    nfc_e = "\xe9"     # é as single NFC codepoint
    result = normalize_text(nfd_e)
    assert result == nfc_e


def test_normalize_preserves_newlines():
    text = "line one\nline two"
    assert "\n" in normalize_text(text)


def test_normalize_preserves_tabs():
    text = "col1\tcol2"
    assert "\t" in normalize_text(text)


def test_normalize_empty_string():
    assert normalize_text("") == ""


# ─── chunk_text ─────────────────────────────────────────────────────────────────


def test_chunk_empty_returns_empty():
    assert chunk_text("doc-1", "") == []


def test_chunk_whitespace_only_returns_empty():
    assert chunk_text("doc-1", "   ") == []


def test_chunk_short_text_is_single_chunk():
    chunks = chunk_text("doc-1", "short text", chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].content == "short text"


def test_chunk_long_text_produces_multiple_chunks():
    text = "a" * 1000
    chunks = chunk_text("doc-1", text, chunk_size=512, overlap=64)
    assert len(chunks) > 1


def test_chunk_size_respected():
    text = "x" * 2000
    chunk_size = 256
    chunks = chunk_text("doc-1", text, chunk_size=chunk_size, overlap=32)
    for chunk in chunks[:-1]:
        assert len(chunk.content) == chunk_size
    assert len(chunks[-1].content) <= chunk_size


def test_chunk_ids_deterministic_across_calls():
    text = "deterministic content " * 50
    chunks_a = chunk_text("doc-stable", text, chunk_size=512, overlap=64)
    chunks_b = chunk_text("doc-stable", text, chunk_size=512, overlap=64)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_ids_unique_within_document():
    text = "unique " * 200
    chunks = chunk_text("doc-unique", text, chunk_size=512, overlap=64)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_differ_across_documents():
    text = "same content " * 100
    chunks_a = chunk_text("doc-aaa", text, chunk_size=512, overlap=64)
    chunks_b = chunk_text("doc-bbb", text, chunk_size=512, overlap=64)
    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id


def test_chunk_index_sequential():
    text = "index test " * 200
    chunks = chunk_text("doc-idx", text, chunk_size=256, overlap=32)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ─── _make_chunk_id ─────────────────────────────────────────────────────────────


def test_make_chunk_id_is_64_char_hex():
    chunk_id = _make_chunk_id("doc-1", 0, "some content")
    assert len(chunk_id) == 64
    assert all(c in "0123456789abcdef" for c in chunk_id)


def test_make_chunk_id_stable_across_calls():
    a = _make_chunk_id("doc-x", 3, "stable content")
    b = _make_chunk_id("doc-x", 3, "stable content")
    assert a == b


def test_make_chunk_id_changes_with_content():
    a = _make_chunk_id("doc-x", 0, "content A")
    b = _make_chunk_id("doc-x", 0, "content B")
    assert a != b


def test_make_chunk_id_changes_with_document_id():
    a = _make_chunk_id("doc-aaa", 0, "same content")
    b = _make_chunk_id("doc-bbb", 0, "same content")
    assert a != b
