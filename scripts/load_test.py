"""
Load Test Script — Day 10

Measures RAG endpoint throughput and latency under concurrent load.
Sends N parallel queries and reports p50, p95, p99 latency and RPS.

Usage:
    # Ensure API is running first:
    #   uvicorn api.rag_endpoint:app --port 8000
    #
    python scripts/load_test.py [--url URL] [--requests N] [--concurrency C]

Defaults:
    --url         http://localhost:8000
    --requests    50
    --concurrency 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

QUERIES = [
    "What is retrieval-augmented generation?",
    "How does Apache Kafka work?",
    "What are word embeddings?",
    "What is prompt engineering?",
    "How does Apache Spark process data?",
    "What is a vector database?",
    "What are large language models?",
    "How does semantic search work?",
    "What is fine-tuning in machine learning?",
    "How are transformer models trained?",
]


def _single_request(url: str, query: str, top_k: int = 3) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{url}/query",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        return {
            "status":   r.status_code,
            "elapsed":  elapsed,
            "ok":       r.status_code == 200,
            "error":    None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status":  0,
            "elapsed": elapsed,
            "ok":      False,
            "error":   str(exc),
        }


def run_load_test(url: str, n_requests: int, concurrency: int) -> None:
    print(f"\nLoad Test: {n_requests} requests | concurrency={concurrency} | target={url}")
    print("-" * 60)

    # Warm-up
    print("Warming up...")
    _single_request(url, QUERIES[0])

    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_single_request, url, QUERIES[i % len(QUERIES)])
            for i in range(n_requests)
        ]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = "OK" if result["ok"] else f"ERR {result['status']}"
            print(f"  [{i:3d}/{n_requests}] {status:10s} {result['elapsed']:.3f}s")

    wall_elapsed = time.perf_counter() - wall_start

    # ── Stats ─────────────────────────────────────────────────────────
    latencies  = [r["elapsed"] for r in results]
    successes  = [r for r in results if r["ok"]]
    failures   = [r for r in results if not r["ok"]]

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(data, p):
        idx = max(0, int(len(data) * p / 100) - 1)
        return data[idx]

    print("\n── Results ─────────────────────────────────────────────")
    print(f"  Total requests : {n_requests}")
    print(f"  Successes      : {len(successes)}")
    print(f"  Failures       : {len(failures)}")
    print(f"  Success rate   : {len(successes)/n_requests*100:.1f}%")
    print(f"  Wall time      : {wall_elapsed:.2f}s")
    print(f"  RPS            : {n_requests/wall_elapsed:.2f}")
    print()
    print(f"  Latency p50    : {percentile(latencies_sorted, 50):.3f}s")
    print(f"  Latency p95    : {percentile(latencies_sorted, 95):.3f}s")
    print(f"  Latency p99    : {percentile(latencies_sorted, 99):.3f}s")
    print(f"  Latency mean   : {statistics.mean(latencies):.3f}s")
    print(f"  Latency min    : {min(latencies):.3f}s")
    print(f"  Latency max    : {max(latencies):.3f}s")

    if failures:
        print(f"\n  Failure details:")
        for f in failures[:5]:
            print(f"    status={f['status']} error={f['error']}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG endpoint load test")
    parser.add_argument("--url",         default="http://localhost:8000",
                        help="API base URL")
    parser.add_argument("--requests",    type=int, default=50,
                        help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent workers")
    args = parser.parse_args()

    run_load_test(args.url, args.requests, args.concurrency)
