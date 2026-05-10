# RAG Data Pipeline

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)
![Apache Spark 3.5](https://img.shields.io/badge/Apache%20Spark-3.5-orange)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.2-00ADD8)
![Kafka](https://img.shields.io/badge/Kafka-Confluent%20Cloud-black)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20Store-green)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings%20%2B%20Chat-412991)
![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE)

---

## Architecture

```
Document Sources
  (PDFs · Web Crawl · Structured Records)
          │
          │ Avro-serialized events
          ▼
  ┌───────────────────────────────────────┐
  │  Kafka — raw-documents topic          │
  │  (Confluent Cloud)                    │
  │                                       │
  │  ~5% malformed → raw-documents-dlq   │
  └──────────────┬────────────────────────┘
                 │ micro-batch
                 ▼
  ┌───────────────────────────────────────┐
  │  Spark Structured Streaming           │
  │  chunk · clean · deduplicate          │
  │  chunk_id = SHA-256(doc+idx+hash)     │
  └──────────────┬────────────────────────┘
                 │ Delta write (partitioned by source + ingestion_date)
                 ▼
  ┌───────────────────────────────────────┐
  │  Delta Lake Bronze (S3)               │
  │  raw documents, full fidelity         │
  └──────────────┬────────────────────────┘
                 │ Soda Core quality gate (null · length · dedup)
                 ▼
  ┌───────────────────────────────────────┐
  │  PySpark Silver Job                   │
  │  normalize metadata · filter · enrich │
  └──────────────┬────────────────────────┘
                 │ Delta MERGE on chunk_id
                 ▼
  ┌───────────────────────────────────────┐
  │  Delta Lake Silver (S3)               │
  │  cleaned chunks, normalized metadata  │
  └──────────────┬────────────────────────┘
                 │ Soda Core quality gate (schema · freshness contract)
                 ▼
  ┌───────────────────────────────────────┐
  │  PySpark Embedding Pipeline           │
  │  mapPartitions → OpenAI API (batched) │
  └──────────────┬────────────────────────┘
                 │ Delta write
                 ▼
  ┌───────────────────────────────────────┐
  │  Delta Lake Gold (S3)                 │
  │  chunk_id · vector_id · embedded_at   │
  │  model_version · source               │
  └──────────────┬────────────────────────┘
                 │ Pinecone upsert (vector_id = chunk_id)
                 ▼
  ┌───────────────────────────────────────┐
  │  Pinecone Vector Store                │
  │  1536-dim cosine · full metadata      │
  └──────────────┬────────────────────────┘
                 │ Soda Core freshness check (24h SLA on Silver→Gold gap)
                 │
  ┌──────────────┴────────────────────────┐
  │  Apache Airflow DAGs                  │
  │  full_pipeline_dag · nightly_refresh  │
  └──────────────┬────────────────────────┘
                 │ OpenLineage events at every job boundary
                 │
                 ▼
  ┌───────────────────────────────────────┐
  │  FastAPI RAG Endpoint                 │
  │  POST /query                          │
  │  Pinecone retrieval + OpenAI chat     │
  └───────────────────────────────────────┘
```

---

## Stack

| Technology | Role |
|---|---|
| **Kafka (Confluent Cloud)** | Durable, ordered event bus for all document source events |
| **Spark Structured Streaming** | Micro-batch ingestion — chunking, cleaning, deduplication with exactly-once semantics |
| **PySpark (batch)** | Bronze→Silver normalization and Silver→Gold parallel embedding via `mapPartitions` |
| **Delta Lake on S3** | ACID lakehouse — Bronze / Silver / Gold medallion layers with time travel and CDF |
| **OpenAI `text-embedding-3-small`** | 1536-dimension dense vector generation for semantic search |
| **Pinecone** | Managed ANN vector store — idempotent upsert via `chunk_id` as `vector_id` |
| **Soda Core** | Contract-driven quality gates between every lakehouse layer transition |
| **Apache Airflow** | DAG-based orchestration of the full pipeline and nightly refresh |
| **OpenLineage** | End-to-end dataset lineage graph from source event to Pinecone vector |
| **FastAPI** | Low-latency RAG serving — retrieval + OpenAI chat completion |
| **Python 3.11+** | Consistent runtime across streaming, batch, quality, serving layers |

---

## Design Principles

### 1. Idempotency — `chunk_id` is the system-wide primary key

Every chunk is assigned a deterministic ID:

```python
chunk_id = hashlib.sha256(
    f"{document_id}:{chunk_index}:{content_hash}".encode()
).hexdigest()
```

This `chunk_id` is used directly as the Pinecone `vector_id`. Because Pinecone upsert is idempotent on `vector_id`, any re-run — due to a bug fix, reprocessing, or infrastructure failure — produces exactly the same index state. No duplicate vectors accumulate.

### 2. Delta Lake is the source of truth

Pinecone is a **derived projection** of the Gold Delta table. It can be dropped and rebuilt at any time by replaying Gold through the embedding job. Audit queries, debugging, and reprocessing all target Delta — never Pinecone.

### 3. Contract-driven layer transitions

Soda Core quality gates block pipeline progression on failure. In Airflow:

```
bronze_write >> bronze_soda_gate >> silver_transform
silver_write >> silver_soda_gate >> gold_embed
```

A failed gate raises `SodaScanError` and blocks all downstream tasks. No silent data quality degradation.

### 4. Dead Letter Queue for malformed events

Kafka events failing schema validation are routed to `raw-documents-dlq` with the original payload and a structured failure reason. The main pipeline is never blocked by malformed events. Operators can inspect, fix, and re-publish from the DLQ independently.

### 5. OpenLineage for full data lineage

OpenLineage events are emitted at every job boundary, building a queryable lineage graph:

```
kafka://raw-documents
  → delta://bronze/documents
  → delta://silver/chunks
  → delta://gold/embeddings
  → pinecone://rag-documents
```

When a RAG response is wrong, full provenance of any vector is one lineage query away.

---

## Project Structure

```
rag-data-pipeline/
├── producers/
│   └── document_producer.py        # Kafka producer for PDF, web, structured sources (Day 2)
├── streaming/
│   └── spark_streaming_consumer.py # Spark Structured Streaming → Bronze Delta (Day 4)
├── batch/
│   ├── silver_job.py               # Bronze → Silver PySpark normalize + enrich (Day 5)
│   └── embedding_pipeline.py       # Silver → Gold embedding via mapPartitions (Day 6)
├── quality/
│   ├── bronze_checks.yml           # Soda Core Bronze→Silver gate (Day 5)
│   ├── silver_checks.yml           # Soda Core Silver→Gold gate (Day 5)
│   └── gold_checks.yml             # Soda Core 24h freshness contract (Day 7)
├── dags/
│   ├── full_pipeline_dag.py        # Airflow full pipeline DAG (Day 7)
│   └── nightly_refresh_dag.py      # Airflow nightly embedding refresh DAG (Day 7)
├── api/
│   └── rag_endpoint.py             # FastAPI RAG serving endpoint (Day 8)
├── schemas/
│   └── document_event.avsc         # Avro schema for raw-documents topic (Day 3)
├── lineage/
│   └── openlineage_config.yml      # OpenLineage client configuration (Day 9)
├── tests/
│   └── .gitkeep
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/arcofiero/rag-data-pipeline.git
cd rag-data-pipeline

# 2. Configure credentials
cp .env.example .env
# Edit .env — fill in Kafka, AWS, OpenAI, Pinecone, Soda, OpenLineage values

# 3. Install Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Start local infrastructure
docker compose up -d
# Spark Master UI:  http://localhost:8080
# Airflow UI:       http://localhost:8081
# FastAPI docs:     http://localhost:8000/docs
```

---

## Build Plan

| Day | Milestone | Description |
|---|---|---|
| **0** | Scaffold | Project structure, docker-compose, requirements, .env.example, README |
| **1** | Infrastructure | Confluent Cloud Kafka setup, S3 bucket + Delta table init, Pinecone index creation |
| **2** | Producers | PDF, web crawl, structured record producers with DLQ routing |
| **3** | Avro schemas | `document_event.avsc`, Schema Registry setup, producer integration |
| **4** | Spark Streaming | Micro-batch consumer, chunking, SHA-256 chunk_id, Bronze Delta write |
| **5** | Silver + Soda | PySpark Silver transform, Bronze→Silver and Silver→Gold Soda Core gates |
| **6** | Embeddings | `mapPartitions` OpenAI embedding job, Gold Delta write, idempotency verification |
| **7** | Airflow + Soda | Full pipeline DAG, nightly refresh DAG, Soda freshness contract |
| **8** | FastAPI | `/query` endpoint, Pinecone retrieval, OpenAI chat completion, attribution metadata |
| **9** | OpenLineage | Emitter helpers at each job boundary, Marquez lineage graph validation |
| **10** | Testing + Hardening | Unit tests (chunking, chunk_id, embedding batch), integration tests, load test |

---

## Data Quality Contracts

Soda Core checks are defined as YAML contracts and run as Airflow tasks. A failed scan blocks all downstream tasks.

| Layer | Contract File | Checks |
|---|---|---|
| **Bronze → Silver** | `quality/bronze_checks.yml` | Null content, min chunk length (50 chars), duplicate chunk_id, schema conformance |
| **Silver → Gold** | `quality/silver_checks.yml` | Null content, null chunk_id, min chunk length, schema conformance |
| **Gold freshness** | `quality/gold_checks.yml` | Silver chunks not embedded within 24h, null vector_id, null embedded_at |

Soda scan results are persisted to Soda Cloud for historical audit and alerting.

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/your-feature`
2. Follow Conventional Commits for all commit messages: `feat:`, `fix:`, `docs:`, `chore:`
3. Ensure all Soda Core checks pass locally before opening a PR
4. Add or update tests in `tests/` for any new logic
5. Open a pull request against `main` with a clear description of the change and its motivation

---

## License

MIT
