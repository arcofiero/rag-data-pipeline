# RAG Data Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Apache Kafka](https://img.shields.io/badge/Kafka-Confluent_Cloud-231F20?logo=apachekafka&logoColor=white)](https://confluent.io)
[![Apache Spark](https://img.shields.io/badge/Spark-3.5-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-S3-003366)](https://delta.io)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_Store-000000)](https://pinecone.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings_+_Chat-412991?logo=openai&logoColor=white)](https://openai.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-F55036)](https://groq.com)
[![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Soda Core](https://img.shields.io/badge/Soda_Core-Data_Quality-1DB954)](https://soda.io)
[![OpenLineage](https://img.shields.io/badge/OpenLineage-Lineage-FF6B35)](https://openlineage.io)

Production-grade RAG data pipeline: document ingestion through Kafka → Spark → Delta Lake (Bronze/Silver/Gold) → multi-provider embeddings → Pinecone → FastAPI + Streamlit query interface. Orchestrated by Airflow, validated by Soda Core at every layer transition, and tracked end-to-end by OpenLineage.

---

## Architecture

```
Document sources (PDF · web crawl · structured records)
    │
    ▼
Kafka (Confluent Cloud)                     ← raw-documents topic · Avro · Schema Registry
    │                    │
    │                    └──► Dead Letter Queue (~5% malformed · dead_letter_event.avsc)
    ▼
Spark Structured Streaming                  ← chunk · clean · deduplicate · micro-batch
    │
    ▼
Delta Lake Bronze                           ← raw chunks · partitioned by source + ingestion_date
    │  Soda Core check (nulls · length · dedup)
    ▼
PySpark Silver batch job                    ← normalise · filter · enrich metadata
    │
    ▼
Delta Lake Silver                           ← cleaned chunks · normalised metadata
    │  Soda Core check (schema · word/char counts · source type)
    ▼
PySpark embedding job (mapPartitions)       ← multi-provider embeddings · idempotent
    │                                         chunk_id = Pinecone vector ID
    ├──► Delta Lake Gold                    ← audit trail: chunk_id · vector_id · model · embedded_at
    │      Soda Core freshness check (24h SLA)
    └──► Pinecone                           ← upserted from Gold · full metadata payload
    │
    ▼
Airflow DAGs                                ← full pipeline · local dev · nightly refresh
    │
    ▼
OpenLineage                                 ← source doc → chunk → embedding lineage
    │
    ▼
FastAPI RAG endpoint                        ← Pinecone retrieval · multi-provider chat completion
    │
    ▼
Streamlit UI                                ← query interface · source citations · pipeline health
```

**Design invariants:**
- Delta Lake Gold is the source of truth — Pinecone is derived from it, never the reverse
- Embedding pipeline is fully idempotent — re-running never creates duplicate vectors
- All layer transitions are blocked on Soda Core passing — quality gates are not optional
- Every Pinecone upsert carries full metadata: `source`, `chunk_id`, `document_id`, `content`, `ingested_at`, `embedded_at`
- OpenLineage tracks which source documents influenced which embeddings

---

## Stack

| Layer | Technology |
|-------|------------|
| Streaming ingest | Confluent Cloud Kafka · Avro · Schema Registry |
| Stream processing | Apache Spark Structured Streaming 3.5 |
| Batch processing | PySpark |
| Storage | Delta Lake on AWS S3 |
| Embeddings | OpenAI `text-embedding-3-small` · Gemini `text-embedding-004` (via `EMBEDDING_PROVIDER`) |
| Chat completion | OpenAI GPT-4o · Gemini · Groq (via `CHAT_PROVIDER`) |
| Vector store | Pinecone |
| Data quality | Soda Core |
| Orchestration | Apache Airflow 2.9 |
| Lineage | OpenLineage |
| Query API | FastAPI |
| UI | Streamlit |
| Language | Python 3.11+ |

---

## Project Structure

```
rag-data-pipeline/
├── api/
│   ├── rag_endpoint.py            # FastAPI app: /query, /health, embed, retrieve, generate
│   └── main.py                    # ASGI entry point
├── batch/
│   ├── silver_job.py              # PySpark Silver: normalise, filter, enrich, MERGE
│   └── embedding_pipeline.py      # PySpark Gold: mapPartitions → embed → Pinecone upsert → audit
├── dags/
│   ├── full_pipeline_dag.py       # Production SparkSubmitOperator DAG
│   ├── local_pipeline_dag.py      # Local dev PythonOperator DAG (no Spark cluster needed)
│   └── nightly_refresh_dag.py     # Nightly embedding refresh (02:00 UTC)
├── lineage/
│   ├── emitter.py                 # OpenLineage START/COMPLETE/FAIL event emitter
│   └── openlineage_config.yml     # Namespace, transport, job naming conventions
├── producers/
│   ├── document_producer.py       # Kafka producer (PDF · web · structured · ~5% malformed → DLQ)
│   ├── schema_registry.py         # Confluent Schema Registry Avro serialiser wrapper
│   └── topic_admin.py             # Topic + DLQ creation utility
├── quality/
│   ├── bronze_checks.yml          # Soda Core: nulls, chunk length ≥ 50 chars, dedup
│   ├── silver_checks.yml          # Soda Core: schema, word/char counts, source type
│   └── gold_checks.yml            # Soda Core: freshness SLA (24h), vector_id nulls
├── schemas/
│   ├── document_event.avsc        # Avro schema: raw document event
│   └── dead_letter_event.avsc     # Avro schema: malformed event envelope
├── scripts/
│   ├── bootstrap.py               # First-run setup: topics, registry, Delta table init
│   ├── web_ingest.py              # Wikipedia crawler → Kafka producer
│   ├── e2e_smoke_test.py          # End-to-end pipeline smoke test
│   ├── load_test.py               # RAG endpoint load test (p50/p95/p99 latency, RPS)
│   └── validate_lineage.py        # OpenLineage graph validator
├── streaming/
│   ├── spark_streaming_consumer.py # Spark Structured Streaming: Kafka → chunk → Delta Bronze
│   └── chunker.py                 # Document chunking (512 tokens, 64 overlap, SHA-256 chunk_id)
├── tests/                         # 165 tests, all passing
│   ├── test_dags.py
│   ├── test_embedding_pipeline.py
│   ├── test_lineage.py
│   ├── test_producer.py
│   ├── test_rag_endpoint.py
│   ├── test_silver_job.py
│   ├── test_streaming_consumer.py
│   └── test_web_ingest.py
├── ui/
│   └── app.py                     # Streamlit: query interface, source citations, health sidebar
├── conftest.py                    # Pytest fixtures (Spark session, mock Kafka, Delta tables)
├── .env.example                   # All required environment variables with comments
├── .gitignore
├── architecture.svg               # Architecture diagram
├── docker-compose.yml             # Kafka · Spark · Airflow · Marquez
├── Dockerfile
└── requirements.txt               # Pinned dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Confluent Cloud account (free tier sufficient)
- AWS S3 bucket (or use local `/tmp` paths — already configured in `.env.example`)
- Pinecone account (free tier sufficient)
- At least one of: OpenAI API key · Google AI API key · Groq API key

### 1. Clone and configure

```bash
git clone https://github.com/arcofiero/rag-data-pipeline.git
cd rag-data-pipeline
cp .env.example .env
# Edit .env — fill in Kafka, S3, Pinecone, and LLM credentials
```

### 2. Set your providers

```bash
# In .env — choose your LLM and embedding providers:
CHAT_PROVIDER=openai        # openai | groq | gemini
EMBEDDING_PROVIDER=openai   # openai | gemini
```

### 3. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Bootstrap infrastructure

```bash
# Creates Kafka topics, registers Avro schemas, initialises Delta tables
python scripts/bootstrap.py
```

### 5. Start services

```bash
docker-compose up -d
# Airflow UI:    http://localhost:8080
# Marquez UI:    http://localhost:3000
# FastAPI docs:  http://localhost:8000/docs
```

### 6. Run the pipeline

```bash
# Produce document events to Kafka (web crawl)
python scripts/web_ingest.py

# Start Spark Structured Streaming (Bronze layer)
spark-submit streaming/spark_streaming_consumer.py

# Run Silver + Gold batch jobs
spark-submit batch/silver_job.py
spark-submit batch/embedding_pipeline.py
```

### 7. Query

```bash
# FastAPI endpoint
uvicorn api.rag_endpoint:app --port 8000
# → POST http://localhost:8000/query

# Streamlit UI (query interface + pipeline health)
streamlit run ui/app.py
```

### 8. Run tests

```bash
pytest tests/ -v
# 165 tests, all passing
```

### 9. Load test

```bash
python scripts/load_test.py --requests 50 --concurrency 5
# Reports p50 / p95 / p99 latency and RPS
```

---

## Multi-Provider Support

The pipeline supports multiple LLM and embedding providers via environment variables — no code changes required when switching.

| Variable | Accepted values | Default |
|----------|----------------|---------|
| `CHAT_PROVIDER` | `openai` \| `groq` \| `gemini` | `openai` |
| `EMBEDDING_PROVIDER` | `openai` \| `gemini` | `openai` |

> **Note:** OpenAI embeddings are 1536-dim; Gemini `text-embedding-004` embeddings are 768-dim. Switching embedding providers requires recreating the Pinecone index at the matching dimension.

---

## Low-Level Design

> Component inventory, schema definitions, interface specifications, quality contracts, error handling matrix, and known limitations.

### Component inventory

| Component | Layer | Technology | Responsibility |
|-----------|-------|------------|----------------|
| Kafka producer | Ingestion | `confluent-kafka` · Confluent Cloud | Simulate PDF, web, and structured document events. Serialise with Avro. Route ~5% malformed to DLQ topic. |
| Schema Registry | Ingestion | Confluent Schema Registry | Enforce `document_event.avsc` and `dead_letter_event.avsc`. Reject non-conforming messages at produce time. |
| Spark Streaming | Bronze | Spark Structured Streaming 3.5 | Consume from Kafka micro-batches. Chunk documents (512 tokens, 64 overlap). Deduplicate on `chunk_id`. Write to Delta Bronze. |
| Delta Lake Bronze | Bronze | Delta Lake on S3 | Raw chunks partitioned by `source` + `ingestion_date`. Append-only. Immutable record of all ingest. |
| Silver job | Silver | PySpark batch | Normalise metadata. Filter short chunks (<50 chars, <10 words). Enrich with word/char counts. Delta MERGE on `chunk_id`. |
| Delta Lake Silver | Silver | Delta Lake on S3 | Cleaned, enriched chunks. Input to embedding pipeline. |
| Soda Core checks | Quality | Soda Core · YAML contracts | Validate Bronze, Silver, Gold. Block layer transition on failure. |
| Embedding job | Gold | PySpark · `mapPartitions` · multi-provider | Anti-join Silver on Gold by `chunk_id` — skip already-embedded. Call embedding API. Upsert to Pinecone. Write audit row to Delta Gold. |
| Delta Lake Gold | Gold | Delta Lake on S3 | Embedding audit trail. Source of truth for Pinecone state. Partitioned by `source` + `embedded_date`. |
| Pinecone | Gold | Pinecone serverless | Vector store. Receives upserts from Gold job. Full metadata payload per vector. |
| Airflow DAGs | Orchestration | Airflow 2.9 | Three DAGs: full pipeline (SparkSubmitOperator), local dev (PythonOperator), nightly refresh. Soda checks are tasks — failure halts downstream. |
| OpenLineage emitter | Observability | OpenLineage | Emit `START`/`COMPLETE`/`FAIL` events at each layer transition. No-ops gracefully if `OPENLINEAGE_URL` is unset. |
| FastAPI endpoint | Serving | FastAPI · Python 3.11 | Accept query. Embed with configured provider. Query Pinecone top-k. Compose prompt. Return completion + source citations. |
| Streamlit UI | Serving | Streamlit | Query interface with answer card, source type badges, similarity score bars, query history, and pipeline health sidebar. |

---

### Schema definitions

**Delta Lake Bronze / Silver — chunk record**

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | `sha256(document_id + ":" + chunk_index + ":" + content_hash)` — deterministic, system-wide PK |
| `document_id` | string | UUID assigned at produce time |
| `source` | string | enum: `PDF` · `WEB` · `STRUCTURED` |
| `source_uri` | string | URL or file path. Nullable for PDFs |
| `content` | string | Chunk text, 512 token max, 64 token overlap |
| `chunk_index` | integer | Zero-based position within parent document |
| `ingested_at` | timestamp | Kafka message timestamp (UTC) |
| `ingestion_date` | date | Partition column, derived from `ingested_at` |
| `word_count` | integer | Silver: added during normalisation |
| `char_count` | integer | Silver: added during normalisation |
| `language` | string | Silver: `en` (detected or defaulted) |
| `processed_at` | timestamp | Silver: timestamp of Silver job write |

**Delta Lake Gold — embedding audit record**

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | FK → Silver.chunk_id · also Pinecone vector ID |
| `vector_id` | string | Pinecone upsert ID (== `chunk_id`) |
| `document_id` | string | Denormalised from Silver |
| `source` | string | Denormalised from Silver |
| `model` | string | Embedding model name, e.g. `text-embedding-3-small` |
| `ingested_at` | string | Copied from Silver — used in freshness check |
| `embedded_at` | string | Timestamp of embedding API call (ISO 8601, UTC) |
| `embedded_date` | string | Partition column, derived from `embedded_at` |

**Pinecone vector metadata**

| Field | Notes |
|-------|-------|
| `chunk_id` | Matches `vector_id` — joins back to Gold |
| `document_id` | Source document identifier |
| `source` | `PDF` · `WEB` · `STRUCTURED` |
| `content` | Full chunk text — returned in RAG citations |
| `ingested_at` | Bronze ingestion timestamp |
| `embedded_at` | Gold embedding timestamp |

---

### Interface specifications

**Producer → Kafka**

| Property | Value |
|----------|-------|
| Topic | `raw-documents` |
| DLQ topic | `raw-documents-dlq` |
| Schema | `document_event.avsc` |
| Message key | `document_id` (string) |
| Acks | `all` — no data loss on produce |
| Registry strategy | topic-value subject naming |
| Malformed rate | ~5% routed to DLQ as `dead_letter_event.avsc` |

**Kafka → Spark Streaming**

| Property | Value |
|----------|-------|
| Mode | Micro-batch, trigger every 30s |
| Offset management | Committed to Kafka after Delta write |
| Failure behaviour | Restart from last committed offset |
| Checkpointing | S3 / local checkpoint location |
| Output | Delta Bronze, append mode |

**Embedding job → Pinecone**

| Property | Value |
|----------|-------|
| Vector ID | `chunk_id` — upsert is idempotent |
| Metadata payload | `chunk_id`, `document_id`, `source`, `content`, `ingested_at`, `embedded_at` |
| Batch size | 100 vectors per upsert call |
| Pre-check | Gold anti-joined against Silver — already-embedded `chunk_id`s skipped |
| Retry logic | Exponential backoff with `retryDelay` from error body; sub-batching for Gemini (≤100 items) |

**FastAPI RAG endpoint**

| Property | Value |
|----------|-------|
| Route | `POST /query` |
| Request | `{ "query": string (1–1000 chars), "top_k": int (1–20), "filter_source": string? }` |
| Response | `{ "answer": string, "sources": [Citation], "query": string, "model": string, "chunks_retrieved": int }` |
| Citation fields | `chunk_id`, `document_id`, `source`, `score`, `content_preview` |
| Flow | Embed query → Pinecone top-k → build context (budget: 20,000 chars) → chat completion |
| Health route | `GET /health` → `{ "status": "ok"|"degraded", "pinecone_ready": bool, "openai_key_set": bool }` |

---

### Quality contracts

All layer promotions are blocked if any Soda Core check fails.

**Bronze (6 checks)**

| Check | Condition |
|-------|-----------|
| Schema conformance | All required fields present with correct types |
| No null content | `missing_count(content) = 0` |
| No null chunk IDs | `missing_count(chunk_id) = 0` |
| No duplicate chunk IDs | `duplicate_count(chunk_id) = 0` |
| No null document IDs | `missing_count(document_id) = 0` |
| Valid source type | `invalid_count(source) = 0` (enum: PDF, WEB, STRUCTURED) |

**Silver (12 checks)**

| Check | Condition |
|-------|-----------|
| Schema conformance | All required fields present |
| No null content | `missing_count(content) = 0` |
| No null / duplicate chunk IDs | `missing_count = 0`, `duplicate_count = 0` |
| No null document IDs | `missing_count(document_id) = 0` |
| Word count minimum | `min(word_count) >= 10` |
| Char count minimum | `min(char_count) >= 50` |
| No null processed_at | `missing_count(processed_at) = 0` |
| No null language | `missing_count(language) = 0` |
| Valid source type | `invalid_count(source) = 0` |

**Gold (8 checks)**

| Check | Condition |
|-------|-----------|
| Schema conformance | All required fields present |
| No null chunk IDs | `missing_count(chunk_id) = 0` |
| No null / duplicate vector IDs | `missing_count = 0`, `duplicate_count = 0` |
| No null document IDs | `missing_count(document_id) = 0` |
| No null embedded_at | `missing_count(embedded_at) = 0` |
| No null model | `missing_count(model) = 0` |
| Valid embedding model | `invalid_count(model) = 0` (known model names) |

---

### Error handling matrix

| Failure scenario | Detection | Handling | Recovery |
|-----------------|-----------|----------|----------|
| Malformed Kafka message | Producer — Avro serialise | Re-serialise as `dead_letter_event.avsc`, produce to DLQ topic | Manual DLQ replay after source fix |
| Schema Registry reject | Confluent broker on produce | Exception raised; message not committed | Fix schema, re-register, re-produce |
| Spark Streaming driver crash | Spark driver | Kafka offsets not committed; checkpoint preserved | Restart job; replay from last checkpoint |
| Soda check failure | Airflow task | Task FAILED; downstream tasks skipped; OpenLineage FAIL event emitted | Fix data quality issue; re-trigger DAG |
| Gemini embedding 503 / transient | Embedding job — `mapPartitions` | Exponential backoff [1s, 2s, 4s]; `retryDelay` from error body respected | Automatic — no intervention needed |
| Gemini embedding 429 quota | Embedding job | `retryDelay` from error body used as wait; raised after exhausting retries | Wait for quota reset; re-run job |
| Gemini batch > 100 items | Embedding job | Sub-batched automatically into ≤100 item chunks | Transparent — no intervention needed |
| Pinecone upsert failure | Embedding job | Gold record NOT written (Gold write follows successful upsert) | Re-run job; chunk re-embedded and re-upserted |
| Embedding API timeout | Embedding job | `retries=3` with backoff; partition logged on exhaustion | Re-run job; already-embedded chunks skipped |
| FastAPI upstream failure (OpenAI/Pinecone) | RAG endpoint | `HTTPException(502)` with descriptive detail | Client retries; no state corruption |

---

### Idempotency

Re-running any part of the pipeline produces identical state:

1. `chunk_id` is deterministic: `sha256(document_id + ":" + chunk_index + ":" + content_hash)`
2. `chunk_id` is the Pinecone `vector_id` — upserts are idempotent by API contract
3. Embedding job anti-joins Silver against Gold before calling the embedding API — already-embedded chunks are skipped without an API call
4. Silver MERGE uses `chunk_id` as the match key — re-runs update in place, no duplicates

---

### Known limitations

| Limitation | Detail | Path to fix |
|------------|--------|-------------|
| Static Avro schema in consumer | Spark Streaming reads schema from local `.avsc` file, not from Schema Registry at runtime. Schema evolution requires a consumer restart. | Confluent Spark Avro connector with registry integration |
| Single Pinecone namespace | All vectors go into the default namespace — no per-source isolation. | Gold carries `source` — one parameter change in the upsert call |
| No embedding model migration DAG | Switching providers requires re-embedding all Silver chunks. `model` in Gold enables detection of mixed-model vectors but no migration DAG exists. | New Airflow DAG parameterised on `model` value |
| Gemini embedding dimension mismatch | Gemini `text-embedding-004` produces 768-dim vectors; OpenAI produces 1536-dim. Switching providers without recreating the Pinecone index causes silent retrieval failures. | Documented in `.env.example`; index must be recreated |
| Airflow on local Docker only | Not suitable for production scale. | Managed Airflow (Astronomer, MWAA, or Cloud Composer) with S3-backed DAG storage |
| Groq chat only | Groq is supported as a chat provider but not as an embedding provider (Groq does not offer an embeddings API). | N/A — by design |

---

## Lineage

OpenLineage emits `START`, `COMPLETE`, and `FAIL` events at every job boundary:

```
kafka://raw-documents
  → delta://bronze/documents
  → delta://silver/chunks
  → delta://gold/embeddings
  → pinecone://rag-documents
```

Validate the lineage graph after a run:

```bash
python scripts/validate_lineage.py
```

---

## Build Plan

| Day | Focus | What landed |
|-----|-------|-------------|
| 0 | Repo scaffold | Architecture diagram, stack, design principles, README |
| 1 | Infrastructure | Confluent Cloud Kafka, S3 Delta tables, Pinecone index |
| 2 | Kafka producer | Document event producer, ~5% malformed → DLQ, topic admin |
| 3 | Avro + Schema Registry | `document_event.avsc`, `dead_letter_event.avsc`, registry client |
| 4 | Spark Streaming → Bronze | Micro-batch consumer, chunker, SHA-256 `chunk_id`, Delta Bronze write |
| 5 | Silver job + Soda gates | PySpark normalisation, Bronze + Silver Soda Core contracts |
| 6 | Embedding pipeline | `mapPartitions`, OpenAI API, Pinecone upsert, Delta Gold audit trail |
| 7 | Soda Gold + Airflow | Freshness SLA contract, full/local/nightly DAGs, end-to-end DAG test |
| 8 | FastAPI RAG endpoint | Pinecone retrieval, chat completion, source citations, `/health` |
| 9 | OpenLineage | Lineage emitter at every job boundary, Marquez integration, lineage validator |
| 10 | Hardening + portfolio | Multi-provider support (Gemini, Groq), Streamlit UI, web ingest, load test, 165 tests, LLD |

---

## License

MIT
