#!/usr/bin/env python3
"""
HTTP load test — measures QPS, latency percentiles, error rate.

Requires a running engine (cargo run --bin server or docker-compose up).

Usage:
    pip install aiohttp numpy
    python bench/load_test.py
    python bench/load_test.py --url http://localhost:8000 --concurrency 20 --duration 30
    python bench/load_test.py --api-key mykey --mode semantic

Output (example):
    ┌─────────────────────────────────────────┐
    │  Load test results (30s, 20 workers)    │
    ├─────────────────────┬───────────────────┤
    │  Total requests     │ 1842              │
    │  Errors             │ 0 (0.0%)          │
    │  Throughput         │ 61.4 req/s        │
    │  Latency p50        │ 312 ms            │
    │  Latency p95        │ 891 ms            │
    │  Latency p99        │ 1243 ms           │
    │  Latency min        │ 189 ms            │
    │  Latency max        │ 2341 ms           │
    └─────────────────────┴───────────────────┘
"""

import asyncio
import argparse
import time
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp"); sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── test queries (varied to avoid caching effects) ────────────────────────────

QUERIES = [
    {"query": "Who leads the company?",              "mode": "entity"},
    {"query": "What technologies are being used?",   "mode": "semantic"},
    {"query": "Describe the relationships between people", "mode": "relationship"},
    {"query": "What is the main product?",           "mode": "semantic"},
    {"query": "Who works in engineering?",           "mode": "entity"},
    {"query": "How are the entities connected?",     "mode": "relationship"},
    {"query": "Find important concepts",             "mode": "semantic"},
    {"query": "List all companies",                  "mode": "entity"},
]

@dataclass
class Result:
    latency_ms: float
    status: int
    error: Optional[str] = None

@dataclass
class Stats:
    results: list = field(default_factory=list)
    start: float = 0.0

    def add(self, r: Result): self.results.append(r)

    def summary(self, duration: float, concurrency: int) -> dict:
        ok      = [r for r in self.results if r.error is None and r.status == 200]
        errors  = [r for r in self.results if r.error or r.status != 200]
        lats    = sorted([r.latency_ms for r in ok])

        def pct(p):
            if not lats: return 0
            if HAS_NUMPY: return float(np.percentile(lats, p))
            idx = int(len(lats) * p / 100)
            return lats[min(idx, len(lats)-1)]

        return {
            "total":        len(self.results),
            "ok":           len(ok),
            "errors":       len(errors),
            "error_rate":   len(errors) / max(len(self.results), 1) * 100,
            "qps":          len(ok) / duration,
            "p50":          pct(50),
            "p75":          pct(75),
            "p95":          pct(95),
            "p99":          pct(99),
            "min":          min(lats) if lats else 0,
            "max":          max(lats) if lats else 0,
            "mean":         statistics.mean(lats) if lats else 0,
            "stdev":        statistics.stdev(lats) if len(lats) > 1 else 0,
            "duration":     duration,
            "concurrency":  concurrency,
        }

# ── worker ────────────────────────────────────────────────────────────────────

async def worker(
    session: aiohttp.ClientSession,
    url: str,
    stats: Stats,
    stop: asyncio.Event,
    query_idx_ref: list,
):
    while not stop.is_set():
        q = QUERIES[query_idx_ref[0] % len(QUERIES)]
        query_idx_ref[0] += 1

        t0 = time.monotonic()
        try:
            async with session.post(f"{url}/query", json=q, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.read()
                latency_ms = (time.monotonic() - t0) * 1000
                stats.add(Result(latency_ms=latency_ms, status=resp.status))
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            stats.add(Result(latency_ms=latency_ms, status=0, error=str(e)))

# ── main ──────────────────────────────────────────────────────────────────────

async def run(url: str, concurrency: int, duration: int, api_key: Optional[str]):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    stats = Stats(start=time.monotonic())
    stop  = asyncio.Event()
    idx   = [0]

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        # health check first
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    print(f"Health check failed: HTTP {r.status}"); return
        except Exception as e:
            print(f"Cannot reach {url}: {e}"); return

        print(f"Starting load test: {concurrency} workers × {duration}s → {url}")
        print("─" * 50)

        workers = [asyncio.create_task(worker(session, url, stats, stop, idx))
                   for _ in range(concurrency)]

        # progress every 5s
        for elapsed in range(5, duration + 1, 5):
            await asyncio.sleep(5)
            so_far = [r for r in stats.results if r.error is None and r.status == 200]
            print(f"  {elapsed:3d}s: {len(so_far):5d} OK  |  "
                  f"{len(stats.results) - len(so_far)} errors  |  "
                  f"{len(so_far)/elapsed:.1f} req/s")

        stop.set()
        await asyncio.gather(*workers, return_exceptions=True)

    s = stats.summary(duration, concurrency)
    print_table(s)
    return s

def print_table(s: dict):
    rows = [
        ("Total requests",  str(s["total"])),
        ("Errors",          f"{s['errors']} ({s['error_rate']:.1f}%)"),
        ("Throughput",      f"{s['qps']:.1f} req/s"),
        ("Latency p50",     f"{s['p50']:.0f} ms"),
        ("Latency p75",     f"{s['p75']:.0f} ms"),
        ("Latency p95",     f"{s['p95']:.0f} ms"),
        ("Latency p99",     f"{s['p99']:.0f} ms"),
        ("Latency min",     f"{s['min']:.0f} ms"),
        ("Latency max",     f"{s['max']:.0f} ms"),
        ("Latency mean",    f"{s['mean']:.0f} ms"),
        ("Latency stdev",   f"{s['stdev']:.0f} ms"),
    ]
    title = f"Load test results ({s['duration']}s, {s['concurrency']} workers)"
    w = max(len(r[0]) for r in rows) + 2
    v = max(len(r[1]) for r in rows) + 2

    print()
    print(f"┌{'─'*(w+v+3)}┐")
    print(f"│  {title:<{w+v+1}}│")
    print(f"├{'─'*w}┬{'─'*v}┤")
    for k, val in rows:
        print(f"│ {k:<{w-1}}│ {val:<{v-1}}│")
    print(f"└{'─'*w}┴{'─'*v}┘")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url",         default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--duration",    type=int, default=30)
    p.add_argument("--api-key",     default=None)
    args = p.parse_args()

    asyncio.run(run(args.url, args.concurrency, args.duration, args.api_key))
