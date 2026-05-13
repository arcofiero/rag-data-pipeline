"""
Pure-Python text normalization and chunking utilities.

No Spark imports — this module is safe to import on both the driver and
executors, and is independently testable without a SparkSession.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import List, NamedTuple


class Chunk(NamedTuple):
    chunk_id: str
    chunk_index: int
    content: str


def normalize_text(text: str) -> str:
    """
    NFC unicode normalization, control-char stripping, whitespace collapse.

    Preserves \\t and \\n (structural whitespace). Strips all other ASCII
    control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F) which
    appear in PDF extraction output and raw web scrapes.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _make_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """
    Stable, deterministic chunk identifier.

    chunk_id = SHA-256( "{document_id}::{chunk_index}::{content_hash[:16]}" )
    where content_hash is the full SHA-256 hex digest of the chunk content.

    Using the first 16 hex chars of the content hash (64 bits of entropy)
    in the payload keeps the input short while making accidental collisions
    astronomically unlikely. The outer SHA-256 produces a fixed-length 64-char
    hex ID used as the Pinecone vector_id.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    payload = f"{document_id}::{chunk_index}::{content_hash[:16]}"
    return hashlib.sha256(payload.encode()).hexdigest()


def chunk_text(
    document_id: str,
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """
    Split normalized text into overlapping fixed-size character windows.

    Each window is chunk_size chars wide; consecutive windows advance by
    (chunk_size - overlap) chars so adjacent chunks share overlap chars.
    The final window may be shorter than chunk_size.

    Returns an empty list for empty or whitespace-only input.
    """
    text = normalize_text(text)
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        content = text[start : start + chunk_size]
        chunk_id = _make_chunk_id(document_id, chunk_index, content)
        chunks.append(Chunk(chunk_id=chunk_id, chunk_index=chunk_index, content=content))
        chunk_index += 1
        start += step

    return chunks
