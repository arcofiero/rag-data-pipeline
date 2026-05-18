"""
Unit tests for the RAG endpoint — Day 8.
Tests cover embed_query, retrieve_chunks, build_context, generate_answer,
and the /query and /health endpoints using mocked OpenAI and Pinecone clients.
No live API calls required.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY",         "test-key")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# Force OpenAI provider — prevents .env Gemini values leaking into these tests
os.environ["CHAT_PROVIDER"]      = "openai"
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ.setdefault("OPENAI_CHAT_MODEL",      "gpt-4o")
os.environ.setdefault("PINECONE_API_KEY",        "test-pc-key")
os.environ.setdefault("PINECONE_INDEX_NAME",     "test-index")
os.environ.setdefault("RAG_TOP_K",               "5")
os.environ.setdefault("RAG_MAX_CONTEXT_CHARS",   "20000")


def _make_pinecone_match(chunk_id="chunk-1", score=0.92, source="PDF", doc_id="doc-1", content="This is chunk content."):
    match = MagicMock()
    match.id    = chunk_id
    match.score = score
    match.metadata = {
        "chunk_id":    chunk_id,
        "document_id": doc_id,
        "source":      source,
        "content":     content,
    }
    return match


def _fake_embedding(dim=1536):
    return [0.01] * dim


def _mock_openai():
    mock = MagicMock()
    mock.embeddings.create.return_value.data = [MagicMock(embedding=_fake_embedding())]
    mock.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="This is the answer."))
    ]
    return mock


@pytest.fixture
def mock_clients(monkeypatch):
    openai_mock   = _mock_openai()
    pinecone_mock = MagicMock()
    pinecone_mock.query.return_value.matches = [_make_pinecone_match()]
    pinecone_mock.describe_index_stats.return_value = {}

    import api.rag_endpoint as ep
    monkeypatch.setattr(ep, "_openai_client",  openai_mock)
    monkeypatch.setattr(ep, "_chat_client",    openai_mock)  # same mock — default provider is openai
    monkeypatch.setattr(ep, "_pinecone_index", pinecone_mock)
    return openai_mock, pinecone_mock


@pytest.fixture
def client(mock_clients):
    from api.rag_endpoint import app
    return TestClient(app)


# ─── embed_query ─────────────────────────────────────────────────────────────────

class TestEmbedQuery:
    def test_returns_embedding_list(self, mock_clients):
        from api.rag_endpoint import embed_query
        result = embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 1536

    def test_calls_openai_with_correct_model(self, mock_clients):
        from api.rag_endpoint import embed_query, OPENAI_EMBEDDING_MODEL
        openai_mock, _ = mock_clients
        embed_query("test query")
        call_kwargs = openai_mock.embeddings.create.call_args[1]
        assert call_kwargs["model"] == OPENAI_EMBEDDING_MODEL

    def test_openai_failure_raises_http_502(self, mock_clients):
        from api.rag_endpoint import embed_query
        from fastapi import HTTPException
        openai_mock, _ = mock_clients
        openai_mock.embeddings.create.side_effect = Exception("API down")
        with pytest.raises(HTTPException) as exc_info:
            embed_query("test query")
        assert exc_info.value.status_code == 502


# ─── retrieve_chunks ─────────────────────────────────────────────────────────────

class TestRetrieveChunks:
    def test_returns_matches(self, mock_clients):
        from api.rag_endpoint import retrieve_chunks
        result = retrieve_chunks(_fake_embedding(), top_k=5, filter_source=None)
        assert len(result) == 1

    def test_applies_source_filter(self, mock_clients):
        from api.rag_endpoint import retrieve_chunks
        _, pinecone_mock = mock_clients
        retrieve_chunks(_fake_embedding(), top_k=5, filter_source="PDF")
        call_kwargs = pinecone_mock.query.call_args[1]
        assert call_kwargs["filter"] == {"source": "PDF"}

    def test_no_filter_when_source_is_none(self, mock_clients):
        from api.rag_endpoint import retrieve_chunks
        _, pinecone_mock = mock_clients
        retrieve_chunks(_fake_embedding(), top_k=5, filter_source=None)
        call_kwargs = pinecone_mock.query.call_args[1]
        assert call_kwargs["filter"] is None

    def test_pinecone_failure_raises_http_502(self, mock_clients):
        from api.rag_endpoint import retrieve_chunks
        from fastapi import HTTPException
        _, pinecone_mock = mock_clients
        pinecone_mock.query.side_effect = Exception("Pinecone down")
        with pytest.raises(HTTPException) as exc_info:
            retrieve_chunks(_fake_embedding(), top_k=5, filter_source=None)
        assert exc_info.value.status_code == 502


# ─── build_context ────────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_returns_context_string_and_citations(self):
        from api.rag_endpoint import build_context
        matches  = [_make_pinecone_match()]
        ctx, cits = build_context(matches, max_chars=20000)
        assert isinstance(ctx, str)
        assert len(cits) == 1

    def test_citation_fields_populated(self):
        from api.rag_endpoint import build_context
        matches   = [_make_pinecone_match(chunk_id="c1", score=0.95, source="PDF", doc_id="doc-1")]
        _, cits   = build_context(matches, max_chars=20000)
        assert cits[0].chunk_id    == "c1"
        assert cits[0].document_id == "doc-1"
        assert cits[0].source      == "PDF"
        assert cits[0].score       == 0.95

    def test_context_budget_respected(self):
        from api.rag_endpoint import build_context
        long_content = "x" * 10000
        matches = [
            _make_pinecone_match(chunk_id="c1", content=long_content),
            _make_pinecone_match(chunk_id="c2", content=long_content),
            _make_pinecone_match(chunk_id="c3", content=long_content),
        ]
        ctx, cits = build_context(matches, max_chars=15000)
        assert len(cits) < 3

    def test_content_preview_truncated_at_200_chars(self):
        from api.rag_endpoint import build_context
        long_content = "a" * 500
        matches      = [_make_pinecone_match(content=long_content)]
        _, cits      = build_context(matches, max_chars=20000)
        assert len(cits[0].content_preview) <= 203  # 200 chars + "..."

    def test_chunk_id_in_context_string(self):
        from api.rag_endpoint import build_context
        matches   = [_make_pinecone_match(chunk_id="my-chunk-id")]
        ctx, _    = build_context(matches, max_chars=20000)
        assert "my-chunk-id" in ctx

    def test_empty_matches_returns_empty(self):
        from api.rag_endpoint import build_context
        ctx, cits = build_context([], max_chars=20000)
        assert ctx  == ""
        assert cits == []


# ─── generate_answer ─────────────────────────────────────────────────────────────

class TestGenerateAnswer:
    def test_returns_answer_string(self, mock_clients):
        from api.rag_endpoint import generate_answer
        result = generate_answer("What is X?", "Context about X.")
        assert result == "This is the answer."

    def test_openai_failure_raises_http_502(self, mock_clients):
        from api.rag_endpoint import generate_answer
        from fastapi import HTTPException
        openai_mock, _ = mock_clients
        openai_mock.chat.completions.create.side_effect = Exception("Chat API down")
        with pytest.raises(HTTPException) as exc_info:
            generate_answer("query", "context")
        assert exc_info.value.status_code == 502

    def test_temperature_is_low(self, mock_clients):
        from api.rag_endpoint import generate_answer
        openai_mock, _ = mock_clients
        generate_answer("query", "context")
        call_kwargs = openai_mock.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] <= 0.3


# ─── /query endpoint ─────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_returns_200(self, client):
        response = client.post("/query", json={"query": "What is the data retention policy?"})
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        response = client.post("/query", json={"query": "test question"})
        data     = response.json()
        for field in ["answer", "sources", "query", "model", "chunks_retrieved"]:
            assert field in data

    def test_answer_is_string(self, client):
        response = client.post("/query", json={"query": "test"})
        assert isinstance(response.json()["answer"], str)

    def test_sources_contain_citation_fields(self, client):
        response = client.post("/query", json={"query": "test"})
        source   = response.json()["sources"][0]
        for field in ["chunk_id", "document_id", "source", "score", "content_preview"]:
            assert field in source

    def test_chunks_retrieved_matches_sources_count(self, client):
        response = client.post("/query", json={"query": "test"})
        data     = response.json()
        assert data["chunks_retrieved"] == len(data["sources"])

    def test_empty_query_returns_422(self, client):
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_too_long_returns_422(self, client):
        response = client.post("/query", json={"query": "x" * 1001})
        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self, client):
        response = client.post("/query", json={"query": "test", "top_k": 999})
        assert response.status_code == 422

    def test_filter_source_passed_to_pinecone(self, client, mock_clients):
        _, pinecone_mock = mock_clients
        client.post("/query", json={"query": "test", "filter_source": "PDF"})
        call_kwargs = pinecone_mock.query.call_args[1]
        assert call_kwargs["filter"] == {"source": "PDF"}

    def test_no_matches_returns_graceful_response(self, client, mock_clients):
        _, pinecone_mock = mock_clients
        pinecone_mock.query.return_value.matches = []
        response = client.post("/query", json={"query": "obscure question"})
        assert response.status_code == 200
        assert response.json()["chunks_retrieved"] == 0


# ─── /health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_pinecone_ready_true_when_reachable(self, client):
        assert client.get("/health").json()["pinecone_ready"] is True

    def test_openai_key_set_true_when_key_present(self, client):
        assert client.get("/health").json()["openai_key_set"] is True

    def test_status_ok_when_pinecone_ready(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_status_degraded_when_pinecone_fails(self, mock_clients):
        _, pinecone_mock = mock_clients
        pinecone_mock.describe_index_stats.side_effect = Exception("unreachable")
        from api.rag_endpoint import app
        test_client = TestClient(app)
        response    = test_client.get("/health")
        assert response.json()["status"] == "degraded"
        assert response.json()["pinecone_ready"] is False
