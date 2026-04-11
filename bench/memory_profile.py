#!/usr/bin/env python3
"""
Memory profiler — measures RAM usage of the running engine.

Samples /health + /proc/PID/status over time, reports:
  - RSS (Resident Set Size): actual RAM in use
  - VmPeak: peak virtual memory
  - Graph nodes & edges count
  - Cache hit rate trend
  - Memory per node estimate

Usage:
    python bench/memory_profile.py
    python bench/memory_profile.py --pid 12345 --duration 60 --interval 2
    python bench/memory_profile.py --send-queries  # generate load while profiling
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp"); sys.exit(1)

# ── reading process memory ────────────────────────────────────────────────────

def read_proc_mem(pid: int) -> Optional[dict]:
    """Read /proc/PID/status on Linux. Falls back to None on macOS."""
    path = f"/proc/{pid}/status"
    if not os.path.exists(path):
        return None
    result = {}
    with open(path) as f:
        for line in f:
            if line.startswith(("VmRSS", "VmPeak", "VmSize", "VmSwap", "RssAnon", "RssFile")):
                parts = line.split()
                result[parts[0].rstrip(":")] = int(parts[1])  # kB
    return result

def find_engine_pid() -> Optional[int]:
    """Try to find the engine process PID."""
    try:
        import subprocess
        out = subprocess.check_output(["pgrep", "-f", "server"], text=True)
        pids = [int(p) for p in out.strip().split() if p.isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None

# ── sample loop ───────────────────────────────────────────────────────────────

@dataclass
class Sample:
    ts: float
    rss_mb: float
    vm_mb: float
    graph_nodes: int
    graph_edges: int
    delta_size: int
    cache_hit_rate: str
    qps: float

async def sample_once(
    session: aiohttp.ClientSession,
    url: str,
    pid: Optional[int],
    prev_queries: list,
    prev_ts: list,
) -> Optional[Sample]:
    try:
        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=3)) as resp:
            if resp.status != 200:
                return None
            body = await resp.json()
    except Exception as e:
        print(f"  health check failed: {e}")
        return None

    m = body.get("metrics", {})
    graph    = m.get("graph", {})
    cache    = m.get("cache", {})
    queries  = m.get("queries", {})
    total_q  = queries.get("total", 0)

    now = time.monotonic()
    qps = 0.0
    if prev_queries and prev_ts:
        dt = now - prev_ts[0]
        if dt > 0:
            qps = (total_q - prev_queries[0]) / dt
    prev_queries[:] = [total_q]
    prev_ts[:] = [now]

    # memory from /proc
    rss_mb = vm_mb = 0.0
    if pid:
        mem = read_proc_mem(pid)
        if mem:
            rss_mb = mem.get("VmRSS", 0) / 1024
            vm_mb  = mem.get("VmPeak", 0) / 1024

    return Sample(
        ts=now,
        rss_mb=rss_mb,
        vm_mb=vm_mb,
        graph_nodes=graph.get("nodes", 0),
        graph_edges=graph.get("edges", 0),
        delta_size=graph.get("delta_size", 0),
        cache_hit_rate=cache.get("hit_rate", "N/A"),
        qps=qps,
    )

async def send_queries(url: str, headers: dict, stop: asyncio.Event):
    """Background query sender to generate load during profiling."""
    queries = [
        {"query": "Who works here?",     "mode": "entity"},
        {"query": "What technologies?",  "mode": "semantic"},
        {"query": "Find relationships",  "mode": "relationship"},
    ]
    async with aiohttp.ClientSession(headers=headers) as session:
        i = 0
        while not stop.is_set():
            try:
                async with session.post(f"{url}/query",
                                         json=queries[i % len(queries)],
                                         timeout=aiohttp.ClientTimeout(total=10)):
                    pass
            except Exception:
                pass
            i += 1
            await asyncio.sleep(0.5)

# ── report ────────────────────────────────────────────────────────────────────

def print_report(samples: list[Sample]):
    if not samples:
        print("No samples collected."); return

    rss_vals = [s.rss_mb for s in samples if s.rss_mb > 0]
    nodes    = samples[-1].graph_nodes
    edges    = samples[-1].graph_edges

    print("\n" + "═" * 54)
    print("  Memory profile report")
    print("═" * 54)
    print(f"  Samples collected   : {len(samples)}")
    print(f"  Duration            : {samples[-1].ts - samples[0].ts:.1f}s")
    print()
    print("  Graph")
    print(f"    Nodes             : {nodes:,}")
    print(f"    Edges             : {edges:,}")
    if nodes > 0 and rss_vals:
        mb_per_node = max(rss_vals) / nodes * 1024  # KB per node
        print(f"    Approx KB/node    : {mb_per_node:.1f}")
    print()

    if rss_vals:
        print("  RSS (Resident Set Size)")
        print(f"    Min               : {min(rss_vals):.1f} MB")
        print(f"    Max               : {max(rss_vals):.1f} MB")
        print(f"    Final             : {rss_vals[-1]:.1f} MB")
        growth = rss_vals[-1] - rss_vals[0]
        print(f"    Growth            : {growth:+.1f} MB  {'⚠ possible leak' if growth > 100 else '✓ stable'}")
    else:
        print("  RSS: N/A (Linux /proc not available — run on Linux or pass --pid)")

    print()
    print("  Cache")
    print(f"    Final hit rate    : {samples[-1].cache_hit_rate}")
    print()
    print("  Throughput")
    qps_vals = [s.qps for s in samples if s.qps > 0]
    if qps_vals:
        print(f"    Mean QPS          : {sum(qps_vals)/len(qps_vals):.1f}")
        print(f"    Peak QPS          : {max(qps_vals):.1f}")
    else:
        print("    QPS: no query load during profiling (use --send-queries)")

    print("═" * 54)

    # timeline
    print("\n  Timeline (RSS MB | cache hit | delta | QPS)")
    print("  " + "─" * 50)
    for i, s in enumerate(samples):
        if i % max(1, len(samples) // 10) == 0 or i == len(samples) - 1:
            rss_str = f"{s.rss_mb:6.1f} MB" if s.rss_mb else "   N/A   "
            print(f"  t={s.ts - samples[0].ts:5.1f}s  {rss_str}  |  "
                  f"hit={s.cache_hit_rate:>6}  |  "
                  f"Δ={s.delta_size:4d}  |  "
                  f"{s.qps:5.1f} req/s")

# ── main ──────────────────────────────────────────────────────────────────────

async def main(args):
    pid = args.pid or find_engine_pid()
    if pid:
        print(f"Profiling PID {pid}")
    else:
        print("PID not found — memory stats unavailable (RSS will show 0)")
        print("Pass --pid $(pgrep server) to enable memory tracking")

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    samples = []
    prev_q  = []
    prev_ts = []
    stop    = asyncio.Event()

    async with aiohttp.ClientSession(headers=headers) as session:
        # verify server is up
        try:
            async with session.get(f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    print(f"Engine not healthy: HTTP {r.status}"); return
            print(f"Connected to {args.url}")
        except Exception as e:
            print(f"Cannot reach {args.url}: {e}"); return

        query_task = None
        if args.send_queries:
            query_task = asyncio.create_task(send_queries(args.url, headers, stop))
            print("Query load: enabled (1 req / 500ms)")

        print(f"Sampling every {args.interval}s for {args.duration}s…\n")
        end = time.monotonic() + args.duration

        while time.monotonic() < end:
            s = await sample_once(session, args.url, pid, prev_q, prev_ts)
            if s:
                samples.append(s)
                rss = f"{s.rss_mb:.1f} MB" if s.rss_mb else "N/A"
                print(f"  RSS={rss:>10}  nodes={s.graph_nodes:,}  "
                      f"cache={s.cache_hit_rate}  Δ={s.delta_size}  "
                      f"qps={s.qps:.1f}")
            await asyncio.sleep(args.interval)

        stop.set()
        if query_task:
            await query_task

    print_report(samples)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([{
                "ts": s.ts - samples[0].ts,
                "rss_mb": s.rss_mb,
                "graph_nodes": s.graph_nodes,
                "cache_hit_rate": s.cache_hit_rate,
                "qps": s.qps,
            } for s in samples], f, indent=2)
        print(f"\nRaw data saved to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url",          default="http://localhost:8000")
    p.add_argument("--pid",          type=int, default=None)
    p.add_argument("--duration",     type=int, default=60)
    p.add_argument("--interval",     type=float, default=3.0)
    p.add_argument("--api-key",      default=None)
    p.add_argument("--send-queries", action="store_true")
    p.add_argument("--output",       default=None, help="save raw JSON to file")
    asyncio.run(main(p.parse_args()))
