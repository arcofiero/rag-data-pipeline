"""
RAG Serving Endpoint — Day 8 (multi-provider update)

FastAPI application exposing a POST /query endpoint for retrieval-augmented
generation.

Request flow:
  1. Validate incoming query (non-empty, max 1000 chars).
  2. Embed the query (provider selected via EMBEDDING_PROVIDER env var).
  3. Query Pinecone for top-k most similar vectors (default k=5).
  4. Assemble retrieved chunks into a context block.
  5. Generate answer via chat model (provider selected via CHAT_PROVIDER).
  6. Return answer, source citations, and retrieval metadata.

Supported providers:
  EMBEDDING_PROVIDER: openai (default) | gemini
  CHAT_PROVIDER:      openai (default) | groq | gemini

  Groq uses the OpenAI SDK with a custom base_url — no extra package needed.
  Gemini uses google-genai SDK.

  WARNING: Gemini embeddings are 768-dimensional (vs 1536 for OpenAI). If you
  switch EMBEDDING_PROVIDER, recreate the Pinecone index with matching dims.

Design decisions:
- _openai_client handles OpenAI embeddings; _chat_client handles chat
  (may be a Groq-pointed OpenAI client, None for Gemini).
- Gemini helpers (_embed_query_gemini, _generate_answer_gemini) defer
  google.genai import so the module loads cleanly when Gemini is inactive.
- Synchronous FastAPI handlers: the embedding + Pinecone + completion round
  trip is ~1-2s; async adds complexity without meaningful throughput gain
  for a single-user RAG endpoint. Revisit if concurrency > 50 rps.
- All config from environment — no hardcoded values.
- Clients initialised once at module level to reuse HTTP connection pools.
"""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────────

OPENAI_API_KEY         = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY         = os.environ.get("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL      = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

CHAT_PROVIDER      = os.getenv("CHAT_PROVIDER", "openai").lower()       # openai | groq | gemini
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # openai | gemini

PINECONE_API_KEY    = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-pipeline")

DEFAULT_TOP_K       = int(os.getenv("RAG_TOP_K", "5"))
MAX_CONTEXT_CHARS   = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "20000"))
MAX_QUERY_CHARS     = int(os.getenv("RAG_MAX_QUERY_CHARS", "1000"))

SYSTEM_PROMPT = """You are a precise, helpful assistant. Answer the user's question
using only the context provided below. If the context does not contain enough
information to answer the question, say so clearly — do not hallucinate.
Cite the source of each claim using the chunk_id provided in the context."""

# ─── Clients (initialised once, reused across requests) ─────────────────────────

from openai import OpenAI
from pinecone import Pinecone

_openai_client   = OpenAI(api_key=OPENAI_API_KEY, max_retries=3)
_pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
try:
    # newer Pinecone SDK resolves the index host eagerly via an HTTP call;
    # wrap in try/except so the module imports cleanly in test environments
    # where the API key is a placeholder — monkeypatch replaces _pinecone_index
    # in fixtures before any test calls retrieve_chunks or health.
    _pinecone_index: object | None = (
        _pinecone_client.Index(PINECONE_INDEX_NAME) if PINECONE_API_KEY else None
    )
except Exception:
    _pinecone_index = None

if CHAT_PROVIDER == "groq":
    _chat_client: object | None = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        max_retries=3,
    )
    _active_chat_model = GROQ_CHAT_MODEL
elif CHAT_PROVIDER == "gemini":
    _chat_client = None   # Gemini uses its own SDK; see _generate_answer_gemini
    _active_chat_model = GEMINI_CHAT_MODEL
else:  # openai (default)
    _chat_client = OpenAI(api_key=OPENAI_API_KEY, max_retries=3)
    _active_chat_model = OPENAI_CHAT_MODEL


# ─── Pydantic models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filter_source: Optional[str] = Field(
        default=None,
        description="Optional source type filter: PDF, WEB, or STRUCTURED",
    )


class SourceCitation(BaseModel):
    chunk_id:    str
    document_id: str
    source:      str
    score:       float
    content_preview: str = Field(description="First 200 chars of the chunk content")


class QueryResponse(BaseModel):
    answer:   str
    sources:  List[SourceCitation]
    query:    str
    model:    str
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status:          str
    pinecone_ready:  bool
    openai_key_set:  bool


# ─── FastAPI app ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Data Pipeline — Query API",
    description="Retrieval-Augmented Generation endpoint backed by Pinecone + OpenAI",
    version="1.0.0",
)


# ─── Core RAG logic (pure functions for testability) ────────────────────────────

def _embed_query_gemini(query: str) -> List[float]:
    """Gemini embedding via google-genai SDK. Produces 768-dim vectors."""
    try:
        from google import genai as _genai
        _client = _genai.Client(api_key=GEMINI_API_KEY)
        result  = _client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=query)
        return result.embeddings[0].values
    except Exception as exc:
        logger.error("Gemini embedding call failed: {}", exc)
        raise HTTPException(status_code=502, detail=f"Embedding service unavailable: {exc}")


def embed_query(query: str) -> List[float]:
    """
    Embed a query string. Provider selected via EMBEDDING_PROVIDER env var.
    Must use the same model/provider as the ingestion pipeline — model mismatch
    makes cosine similarity scores meaningless.
    Raises HTTPException 502 on failure.
    """
    if EMBEDDING_PROVIDER == "gemini":
        return _embed_query_gemini(query)
    try:
        response = _openai_client.embeddings.create(
            input=[query],
            model=OPENAI_EMBEDDING_MODEL,
        )
        return response.data[0].embedding
    except Exception as exc:
        logger.error("OpenAI embedding call failed: {}", exc)
        raise HTTPException(status_code=502, detail=f"Embedding service unavailable: {exc}")


def retrieve_chunks(
    embedding: List[float],
    top_k: int,
    filter_source: Optional[str],
) -> list:
    """
    Query Pinecone for the top-k most similar vectors.

    filter_source applies a Pinecone metadata filter so only chunks from
    the specified source type are returned. This lets callers restrict
    answers to PDF documents only, web content only, etc.

    Returns Pinecone match objects with id, score, and metadata fields.
    Raises HTTPException 502 on Pinecone failure.
    """
    if _pinecone_index is None:
        raise HTTPException(status_code=503, detail="Pinecone index not initialised")

    pinecone_filter = {"source": filter_source} if filter_source else None

    try:
        results = _pinecone_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
        return results.matches
    except Exception as exc:
        logger.error("Pinecone query failed: {}", exc)
        raise HTTPException(status_code=502, detail=f"Vector store unavailable: {exc}")


def build_context(matches: list, max_chars: int) -> tuple[str, List[SourceCitation]]:
    """
    Assemble retrieved chunks into a context string for the chat prompt
    and build the source citation list for the response.

    Context format:
        [chunk_id: abc123 | source: PDF | doc: doc-001]
        <chunk content>

        [chunk_id: def456 | source: WEB | doc: doc-002]
        <chunk content>

    Chunks are added in score order (Pinecone returns them ranked) until
    max_chars is reached. This ensures the highest-relevance content is
    always included even if the full context would exceed the budget.
    """
    context_parts: List[str] = []
    citations:     List[SourceCitation] = []
    total_chars = 0

    for match in matches:
        meta    = match.metadata or {}
        content = meta.get("content", meta.get("text", ""))

        header  = (
            f"[chunk_id: {match.id} | "
            f"source: {meta.get('source', 'UNKNOWN')} | "
            f"doc: {meta.get('document_id', 'unknown')}]"
        )
        block   = f"{header}\n{content}"
        block_chars = len(block)

        if total_chars + block_chars > max_chars:
            logger.debug(
                "Context budget reached at {} chars — truncating retrieval",
                total_chars,
            )
            break

        context_parts.append(block)
        total_chars += block_chars

        citations.append(SourceCitation(
            chunk_id=match.id,
            document_id=meta.get("document_id", "unknown"),
            source=meta.get("source", "UNKNOWN"),
            score=round(float(match.score), 4),
            content_preview=(content[:200] + "...") if len(content) > 200 else content,
        ))

    return "\n\n".join(context_parts), citations


def _generate_answer_gemini(query: str, context: str) -> str:
    """Gemini chat via google-genai SDK."""
    try:
        from google import genai as _genai
        _client = _genai.Client(api_key=GEMINI_API_KEY)
        prompt  = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
        result  = _client.models.generate_content(model=GEMINI_CHAT_MODEL, contents=prompt)
        return result.text or ""
    except Exception as exc:
        logger.error("Gemini chat completion failed: {}", exc)
        raise HTTPException(status_code=502, detail=f"Chat completion unavailable: {exc}")


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the active chat provider (_active_chat_model).
    Provider is selected via CHAT_PROVIDER env var (openai | groq | gemini).

    The system prompt instructs the model to answer only from context and
    cite chunk_ids — this is the grounding mechanism that prevents
    hallucination on out-of-context questions.

    Raises HTTPException 502 on failure.
    """
    if CHAT_PROVIDER == "gemini":
        return _generate_answer_gemini(query, context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
    try:
        response = _chat_client.chat.completions.create(
            model=_active_chat_model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Chat completion failed (provider={}): {}", CHAT_PROVIDER, exc)
        raise HTTPException(status_code=502, detail=f"Chat completion unavailable: {exc}")


# ─── Endpoints ────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Main RAG query endpoint.

    POST /query
    {
        "query": "What is the retention policy for customer data?",
        "top_k": 5,
        "filter_source": "PDF"
    }

    Returns the answer, source citations, and retrieval metadata.
    """
    logger.info(
        "Query received | query_len={} top_k={} filter_source={}",
        len(request.query), request.top_k, request.filter_source,
    )

    # Step 1: embed query
    embedding = embed_query(request.query)

    # Step 2: retrieve from Pinecone
    matches = retrieve_chunks(embedding, request.top_k, request.filter_source)

    if not matches:
        logger.warning("No chunks retrieved for query — returning empty answer")
        return QueryResponse(
            answer="I could not find any relevant information in the knowledge base for your query.",
            sources=[],
            query=request.query,
            model=_active_chat_model,
            chunks_retrieved=0,
        )

    # Step 3: build context + citations
    context, citations = build_context(matches, MAX_CONTEXT_CHARS)

    # Step 4: generate answer
    answer = generate_answer(request.query, context)

    logger.info(
        "Query complete | chunks_retrieved={} answer_len={}",
        len(citations), len(answer),
    )

    return QueryResponse(
        answer=answer,
        sources=citations,
        query=request.query,
        model=OPENAI_CHAT_MODEL,
        chunks_retrieved=len(citations),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check endpoint for load balancer probes.
    Returns 200 with dependency status.
    Pinecone connectivity is checked by calling describe_index_stats()
    which is a lightweight metadata call (no vector scan).
    """
    pinecone_ready = False
    if _pinecone_index is not None:
        try:
            _pinecone_index.describe_index_stats()
            pinecone_ready = True
        except Exception as exc:
            logger.warning("Pinecone health check failed: {}", exc)

    return HealthResponse(
        status="ok" if pinecone_ready else "degraded",
        pinecone_ready=pinecone_ready,
        openai_key_set=bool(OPENAI_API_KEY),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
