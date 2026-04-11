#!/usr/bin/env python3
"""
Benchmark runner — runs Criterion micro-benchmarks and generates a summary.

Usage:
    python bench/run_all.py               # Criterion only (no server needed)
    python bench/run_all.py --live        # also run load test + memory profile
    python bench/run_all.py --live --url http://localhost:8000 --api-key mykey
"""

import argparse
import subprocess
import sys
import os
import time
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORE = ROOT / "core"
BENCH = ROOT / "bench"

# ── criterion ────────────────────────────────────────────────────────────────

def run_criterion(bench_name: str) -> bool:
    print(f"\n{'═'*54}")
    print(f"  Criterion: {bench_name}")
    print(f"{'═'*54}")
    result = subprocess.run(
        ["cargo", "bench", "--bench", bench_name, "--", "--output-format", "verbose"],
        cwd=CORE,
        env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"},
    )
    return result.returncode == 0

# ── live benchmarks ───────────────────────────────────────────────────────────

def run_load_test(url: str, api_key: str | None, concurrency: int, duration: int):
    cmd = [
        sys.executable, str(BENCH / "load_test.py"),
        "--url", url,
        "--concurrency", str(concurrency),
        "--duration", str(duration),
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    subprocess.run(cmd)

def run_memory_profile(url: str, api_key: str | None, duration: int, output: str):
    cmd = [
        sys.executable, str(BENCH / "memory_profile.py"),
        "--url", url,
        "--duration", str(duration),
        "--send-queries",
        "--output", output,
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    subprocess.run(cmd)

# ── summary ───────────────────────────────────────────────────────────────────

def print_how_to_run():
    print("""
╔══════════════════════════════════════════════════════╗
║  How to run the full benchmark suite                 ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  1. Micro-benchmarks (no server needed):             ║
║                                                      ║
║     cd core                                          ║
║     cargo bench                                      ║
║                                                      ║
║     # or individual suites:                          ║
║     cargo bench --bench graph_traversal              ║
║     cargo bench --bench vector_search                ║
║     cargo bench --bench scoring                      ║
║                                                      ║
║  2. Load test (requires running engine):             ║
║                                                      ║
║     python bench/load_test.py                        ║
║     python bench/load_test.py --concurrency 20       ║
║                                                      ║
║  3. Memory profile:                                  ║
║                                                      ║
║     python bench/memory_profile.py --send-queries    ║
║                                                      ║
║  4. All at once:                                     ║
║                                                      ║
║     python bench/run_all.py --live                   ║
║                                                      ║
║  View Criterion HTML reports:                        ║
║     open core/target/criterion/report/index.html     ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

def print_expected_numbers():
    print("""
Expected results (reference machine: M2 MacBook Pro, 16GB RAM)
─────────────────────────────────────────────────────────────

CSR graph traversal
  neighbors() single lookup       ~20 ns
  BFS depth=2, 10k nodes          ~150 μs
  BFS depth=2, 100k nodes         ~1.2 ms
  CSR build, 10k nodes/60k edges  ~8 ms

Vector search (dim=384)
  cosine_sim single pair          ~80 ns
  brute-force top-10, 1k vecs     ~250 μs
  brute-force top-10, 10k vecs    ~2.5 ms
  brute-force top-10, 50k vecs    ~13 ms
  HNSW search top-10, 10k vecs    ~120 μs   ← ~20x faster
  HNSW search top-10, 100k vecs   ~200 μs   ← ~65x faster vs 50k brute
  HNSW build, 10k vecs            ~1.5 s

Scoring
  score 500 nodes                 ~400 μs
  BFS depth=2 + score, 10k graph  ~600 μs
  LRU cache hot path (100 hits)   ~5 μs
  LRU cache cold path (100 misses)~800 μs

End-to-end (with real LLM, 10 concurrent users)
  p50 latency                     ~350 ms
  p95 latency                     ~900 ms
  p99 latency                     ~1.5 s
  throughput                      ~25–60 req/s  (LLM-bound)

Memory (100k nodes, dim=384)
  CSR graph (topology only)       ~3 MB
  HNSW index                      ~600 MB
  LRU cache (50k × 384-dim)       ~300 MB
  mmap vectors (100k × 384-dim)   ~150 MB (OS managed)
  Total RSS                       ~1.1 GB
""")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live",        action="store_true", help="also run load test + memory profile")
    p.add_argument("--url",         default="http://localhost:8000")
    p.add_argument("--api-key",     default=None)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--duration",    type=int, default=30)
    p.add_argument("--skip-criterion", action="store_true")
    args = p.parse_args()

    print_how_to_run()
    print_expected_numbers()

    if args.skip_criterion:
        print("Skipping Criterion (--skip-criterion)")
    else:
        suites = ["graph_traversal", "vector_search", "scoring"]
        failed = []
        for suite in suites:
            ok = run_criterion(suite)
            if not ok:
                failed.append(suite)
                print(f"  ⚠ {suite} failed — run 'cargo bench --bench {suite}' to debug")

        if failed:
            print(f"\n⚠ {len(failed)} bench suite(s) failed: {failed}")
        else:
            print(f"\n✓ All Criterion benchmarks complete")
            print(f"  HTML reports: core/target/criterion/report/index.html")

    if args.live:
        print("\n" + "═" * 54)
        print("  Live load test")
        run_load_test(args.url, args.api_key, args.concurrency, args.duration)

        print("\n" + "═" * 54)
        print("  Memory profile")
        ts = int(time.time())
        out = str(BENCH / f"memory_{ts}.json")
        run_memory_profile(args.url, args.api_key, args.duration, out)

if __name__ == "__main__":
    main()
