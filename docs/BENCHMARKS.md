# Benchmark Report — AI Graph Engine

Criterion micro-benchmarks measuring the core compute primitives.
All numbers are **median (p50)** from 100 samples unless noted.

---

## Environment

| | |
|---|---|
| Machine | Apple M1 Max |
| RAM | 16 GB |
| OS | macOS Darwin 24.6.0 |
| Rust | release profile (`opt-level = 3`) |
| SIMD | `RUSTFLAGS="-C target-cpu=native"` (ARM NEON) |
| Tool | Criterion 0.5 |

---

## CSR Graph Traversal

### Neighbor lookup — O(1) slice

| Implementation | Time (p50) | Throughput |
|---|---|---|
| `CsrGraph::neighbors()` | **7.87 ns** | 127 Melem/s |
| `AdjList` (baseline) | 4.59 ns | 218 Melem/s |

> CSR is slightly slower than AdjList on single-node lookup because the pointer
> indirection fits in L1 cache for both. CSR's advantage shows at scale where
> memory locality matters across many concurrent traversals.

### BFS traversal

| Graph size | Depth 1 | Depth 2 | Depth 3 |
|---|---|---|---|
| 1k nodes | 1.88 µs | 2.72 µs | 4.46 µs |
| 10k nodes | 1.43 µs | 4.91 µs | 10.1 µs |
| 100k nodes | 3.35 µs | 4.20 µs | 5.96 µs |

> Depth=2 at 100k is faster than at 10k because the 100k graph has the same avg
> degree (6) so BFS saturates `max_nodes=10k` earlier and exits sooner.

### CSR build from edge list

| Graph size | Time (p50) | Throughput |
|---|---|---|
| 1k nodes / 6k edges | 234 µs | 4.3 Melem/s |
| 10k nodes / 60k edges | 2.84 ms | 3.5 Melem/s |
| 100k nodes / 600k edges | 29.1 ms | 3.4 Melem/s |

---

## Vector Search

### Cosine similarity — single pair (dim=384)

| | Time (p50) | Throughput |
|---|---|---|
| Before (compiler auto-vec) | 315 ns | 3.2 Melem/s |
| **After (explicit NEON FMA)** | **42–53 ns** | **18–24 Melem/s** |

> **6–7× improvement** via explicit ARM NEON: 4-way unrolled FMA using
> `vfmaq_f32` + 4 independent accumulators to saturate the pipeline.
> 384 elements = 24 iterations × 16 f32/iter (4 × `float32x4_t`).
> Falls back to auto-vectorized iterator on non-aarch64 targets.

### Brute-force top-10 search

| Corpus size | Before | After NEON | Improvement | Throughput |
|---|---|---|---|---|
| 1k vectors | 333 µs | **61 µs** | 5.5× | 16 Melem/s |
| 10k vectors | 3.32 ms | **462 µs** | 7.2× | 22 Melem/s |
| 50k vectors | 16.7 ms | **2.27 ms** | 7.4× | 22 Melem/s |

> Throughput increased from ~3 Melem/s to ~22 Melem/s after NEON.
> Still linear with corpus size — use HNSW for corpora above ~5k vectors.

### HNSW approximate nearest-neighbor search — top-10

| Corpus size | Before | After NEON | vs brute-force (post-NEON) |
|---|---|---|---|
| 1k vectors | 177 µs | **64 µs** | — |
| 10k vectors | 192 µs | **61 µs** | **7.6× faster** |
| 100k vectors | — | **61 µs** | — |

> HNSW search is nearly flat across corpus sizes — 1k→100k adds only ~0 µs.
> NEON improved HNSW search ~3× since all distance comparisons inside graph
> traversal go through `cosine_sim`.

### HNSW build (one-time offline cost)

| Corpus size | Time (p50) |
|---|---|
| 1k vectors | 239 ms |
| 10k vectors | 2.03 s |

---

## Scoring Pipeline

### Score 500 nodes

| | Time (p50) |
|---|---|
| `score_nodes()` — 500 candidates | **168 µs** |

### LRU embed cache

| Scenario | Time (p50) | Per-item |
|---|---|---|
| Hot path — 100 cache hits | 9.54 µs | ~95 ns/hit |
| Cold path — 100 cache misses (mmap load) | 78.7 µs | ~787 ns/miss |

> Cache hit is ~8× faster than a cold mmap load. At 50k cached vectors the
> warm steady-state dominates — cold misses only occur on first access or after
> eviction.

### End-to-end pipeline slice — BFS depth=2 + score (10k graph)

| | Time (p50) |
|---|---|
| BFS expand + score all candidates | **23.4 µs** |

> This covers the pure Rust hot path: seed selection → BFS → scoring.
> LLM embed (~30–50ms) and LLM generate (~200–500ms) dominate real query latency
> and are not measured here (plugin-side, network-bound).

---

## Summary

| Operation | Latency | Bottleneck |
|---|---|---|
| Neighbor lookup | ~8 ns | L1 cache |
| BFS depth=2, 10k graph | ~5 µs | HashSet ops |
| BFS depth=2, 100k graph | ~4 µs | early `max_nodes` cutoff |
| Cosine sim (dim=384) | **~42–53 ns** | NEON FMA throughput |
| HNSW search top-10, 10k | **~61 µs** | graph traversal |
| Brute-force top-10, 50k | **~2.3 ms** | linear scan |
| Score 500 nodes | ~168 µs | dot product × 500 |
| Full Rust pipeline slice | ~23 µs | BFS + scoring |
| **Real query (with LLM)** | **~350–900 ms** | **LLM API latency** |

The Rust core contributes **< 1 ms** to total query latency.
The remaining 99%+ is plugin round-trips (embed + generate).

---

## Running the Benchmarks

```bash
# All suites
cd core && cargo bench

# Single suite
cargo bench --bench graph_traversal
cargo bench --bench vector_search
cargo bench --bench scoring

# With native SIMD (recommended on local machine — not portable)
RUSTFLAGS="-C target-cpu=native" cargo bench

# Load test (requires running engine + plugin)
python bench/load_test.py --concurrency 10 --duration 30

# All at once
b
```

HTML reports (after running Criterion):
```
open core/target/criterion/report/index.html
```
