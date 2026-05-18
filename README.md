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
    │  Soda Core check (schema · word/char count · source type)
    ▼
PySpark embedding job (mapPartitions)       ← multi-provider embeddings · idempotent
    │                                         chunk_id = Pinecone vector ID
    ├──► Delta Lake Gold                    ← audit trail: chunk_id · vector_id · embedded_at · model_version
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
- Every Pinecone upsert carries full metadata: `source`, `chunk_id`, `document_id`, `ingested_at`, `embedded_at`
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
- AWS S3 bucket (or use local `/tmp` paths for dev — already configured in `.env.example`)
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

## Data Quality Contracts

Soda Core validates quality at every layer transition. **All layer promotions are blocked if any check fails.**

| Layer | Contract | Checks |
|-------|----------|--------|
| Bronze → Silver | `quality/bronze_checks.yml` | Null content · chunk length ≥ 50 chars · duplicate `chunk_id` · schema conformance · valid source type |
| Silver → Gold | `quality/silver_checks.yml` | Null content · null `chunk_id` · word count ≥ 10 · char count ≥ 50 · null `processed_at` · valid source type |
| Gold freshness | `quality/gold_checks.yml` | Null `vector_id` · null `embedded_at` · null model name · valid embedding model |

In Airflow the full chain is:

```
bronze_soda_gate >> silver_transform_job >> silver_soda_gate >> embedding_pipeline >> gold_soda_gate
```

---

## Idempotency

Re-running any part of the pipeline produces identical state:

1. `chunk_id` is deterministic: `sha256(document_id + ":" + chunk_index + ":" + content_hash)`
2. `chunk_id` is the Pinecone `vector_id` — upserts are idempotent by API contract
3. Embedding job anti-joins Silver against Gold before calling the embedding API — already-embedded chunks are skipped without an API call

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
| 10 | Hardening + portfolio | Multi-provider support (Gemini, Groq), Streamlit UI, web ingest, load test, 165 tests |

---

## License

MIT
