# Architecture — LinkingMem

## 1. Overview

The system is split into separate processes, each doing what it does best:

```mermaid
graph TD
    Client["Client / API"]
    Rust["🦀 Rust Engine :8000<br/>graph · vector · scoring · HTTP"]
    Text["🐍 Text Plugin :8001<br/>embedding · extraction · generation"]
    Image["🐍 Image Plugin :8002<br/>image store · vision embed"]

    Client -->|HTTP| Rust
    Rust -->|"HTTP / Unix socket"| Text
    Rust -->|HTTP| Image
```

**Why separate?** Rust lacks a mature AI ecosystem. Python lacks the performance required for graph/vector compute. Each side focuses on its strengths.

The image plugin is optional — only required when ingesting image nodes (`image_url` set) or using `POST /query/image`.

---

## 2. Directory Structure

```
LinkingMem/
│
├── core/                        ← Rust engine
│   └── src/
│       ├── main.rs              HTTP server (axum :8000) + bootstrap
│       ├── app_state.rs         AppState — shared state, hot-swap merge
│       ├── config.rs            Reads plugins.toml + env vars
│       ├── query.rs             QueryEngine — full pipeline + cache
│       ├── entity_resolution.rs Embedding-based entity dedup on ingest
│       ├── delta.rs             DeltaStore (LSM-style) + WAL
│       ├── cache.rs             EmbedCache (moka concurrent LRU)
│       ├── plugin.rs            HTTP client Rust → Python (TTL health cache)
│       ├── metrics.rs           Prometheus-compatible counters
│       │
│       ├── graph/
│       │   ├── csr.rs           CsrGraph — bidirectional CSR + BFS
│       │   └── builder.rs       Load/save binary + JSON ingest
│       │
│       ├── storage/
│       │   ├── mod.rs           StorageBackend trait (write/read/exists/local_path)
│       │   └── local.rs         LocalStorage — wraps std::fs, Phase 1 implementation
│       │
│       ├── vector/
│       │   ├── store.rs         VectorStore — mmap, zero-copy read
│       │   └── hnsw.rs          HnswIndex + EdgeHnswIndex (instant-distance)
│       │
│       ├── api/
│       │   ├── handlers/
│       │   │   ├── query.rs     POST /query
│       │   │   ├── ingest.rs    POST /ingest/text  POST /ingest/json
│       │   │   ├── admin.rs     GET /health /metrics /graph/stats /nodes
│       │   │   └── export.rs    GET /export (graph export endpoints)
│       │   ├── dto/
│       │   │   ├── query.rs     QueryReq + PipelineControl + QueryResponse
│       │   │   ├── ingest.rs    IngestTextReq + IngestJsonReq + IngestResponse
│       │   │   └── export.rs    ExportRequest + ExportResponse
│       │   └── error.rs         ApiError — unified error type (400/422/503/502/500)
│       │
│       ├── middleware/
│       │   └── auth.rs          API key auth + token-bucket rate limiter
│       │
│       └── bin/
│           ├── server.rs        Entry point: HTTP server
│           └── ingest.rs        CLI one-time data prep
│
├── plugins/                     ← Python plugin servers
│   ├── text/                    ← Text plugin (port 8001)
│   │   ├── main.py              FastAPI: /health /info /embed/text /embed/image /extract /generate /reason
│   │   ├── embed.py             Text embedding (SentenceTransformers, lazy-loaded)
│   │   ├── embed_image.py       Image embed via Vision LLM caption → sentence-transformers
│   │   ├── extract.py           Entity extraction (LLM)
│   │   ├── generate.py          Answer generation (LLM)
│   │   ├── reason.py            Multi-hop reasoning
│   │   ├── llm.py               LLM client abstraction (OpenAI / Anthropic / Google)
│   │   ├── schemas.py           Pydantic request/response schemas
│   │   └── auth.py              Plugin-side Bearer auth helpers
│   │
│   └── image/                   ← Image plugin (port 8002)
│       ├── main.py              FastAPI: /health /info /store /images/{file} /embed
│       ├── store.py             POST /store + GET /images/{filename}: content-addressed local disk storage
│       ├── embed.py             POST /embed: vision LLM caption → sentence-transformers embed
│       ├── schemas.py           Pydantic schemas (StoreRequest/Response, EmbedRequest/Response)
│       └── auth.py              Plugin-side Bearer auth helpers
│
├── data/                        ← Binary artifacts (git-ignored)
│   ├── nodes.json               Node metadata (id, name, type, weight, props, full_context, embed_context, image_url)
│   ├── images/                  Content-addressed image files (SHA-256 named, served at /images/{file})
│   ├── edges.bin                CSR edge data
│   ├── edge_types.json          Edge type mapping
│   ├── edge_contexts.json       full_context string per edge (parallel to edges.bin)
│   ├── edge_embed_contexts.json embed_context per edge (parallel to edges.bin)
│   ├── vectors.bin              Node embedding vectors (mmap)
│   ├── edge_vectors.bin         Edge embedding vectors (mmap)
│   ├── edge_endpoints.json      (from_id, to_id) pairs parallel to edge_vectors.bin
│   └── delta.wal                Crash-recovery WAL
│
├── plugins.toml                 Plugin endpoint config + query pipeline defaults
├── .env.example                 All env vars with default values
└── docker-compose.yml           Full stack: engine + plugin
```

---

## 3. Data Flow

### 3.1 Query Pipeline

```mermaid
flowchart TD
    REQ["POST /query\n{query, vector?}"]
    AUTH["Auth middleware\nAPI key + rate limit"]
    NEED{"needs_embed\nOR needs_generate?"}
    HEALTH{"Plugin ready?\nTTL-cached 5s"}
    E503["503 plugin_unavailable"]
    ENGINE["QueryEngine.run()"]
    EMBED["[1] embed query → /embed/text\n~30–50ms\nskip if vector provided"]
    HNSW["[2] HNSW search (parallel)\nNodeHnsw ∥ EdgeHnsw\n~1–2ms each"]
    BFS["[3] BFS expand\nCsrGraph + DeltaStore\n~0.5ms"]
    SCORE["[4] Score nodes\nα·vector_sim + β·proximity + γ·node_weight\n~2ms"]
    LLM{"llm_generate?"}
    GEN["[5] generate → /generate\n~500–1500ms"]
    RESP["Response\n{answer, subgraph, stats}"]

    REQ --> AUTH --> NEED
    NEED -->|No - pure graph| ENGINE
    NEED -->|Yes| HEALTH
    HEALTH -->|Not ready| E503
    HEALTH -->|Ready| ENGINE
    ENGINE --> EMBED --> HNSW --> BFS --> SCORE --> LLM
    LLM -->|Yes| GEN --> RESP
    LLM -->|No| RESP
```

### 3.2 Ingest Pipeline (text)

```mermaid
flowchart TD
    REQ["POST /ingest/text\n{text}"]
    HEALTH{"Plugin ready?\nTTL-cached 5s"}
    E503["503 plugin_unavailable"]
    EXTRACT["extract(text) → /extract\nentities + relations"]
    EMBED["embed in parallel (tokio::join!)\nentity vecs + edge vecs → /embed/text"]
    RESOLVE["entity_resolution()\ncosine_sim > threshold → merge"]
    DELTA["DeltaStore.add_node + add_edge\nWAL write first"]
    THRESH{"delta >= threshold?"}
    MERGE["async merge\nrebuild CSR + NodeHnsw + EdgeHnsw\nhot-swap 4 components"]
    RESP["Response\n{ingested, new_nodes, resolved, delta_size}"]

    REQ --> HEALTH
    HEALTH -->|Not ready| E503
    HEALTH -->|Ready| EXTRACT --> EMBED --> RESOLVE --> DELTA --> THRESH
    THRESH -->|Yes| MERGE --> RESP
    THRESH -->|No| RESP
```

### 3.3 Delta Merge (background)

```mermaid
flowchart TD
    DELTA["DeltaStore\nthreshold reached"]
    DRAIN["drain delta"]
    MERGE["merge with CsrGraph"]
    CSR["build new CsrGraph"]
    HNSW["rebuild NodeHnswIndex\nrebuild EdgeHnswIndex\nvia spawn_blocking"]
    PERSIST["persist to data/\nnodes.json · edges.bin · vectors.bin\nedge_vectors.bin · edge_contexts.json"]
    SWAP["acquire write locks g→h→eh→st\nhot-swap 4 components atomically\nArc CsrGraph + HnswIndex + EdgeHnswIndex + VectorStore"]
    DONE["in-flight queries complete\nagainst old Arc refs\nnew queries use new graph"]

    DELTA --> DRAIN --> MERGE --> CSR --> HNSW --> PERSIST --> SWAP --> DONE
```

---

## 4. Core Components

### CsrGraph

Stores graph topology as **Compressed Sparse Row** — 2 contiguous arrays in memory:

```
offsets: [0, 2, 3, 5, 5]
edges:   [1, 3, 2, 0, 4]

neighbors(node 0) = edges[0..2] = [1, 3]
neighbors(node 1) = edges[2..3] = [2]
```

No pointers, no allocation. BFS achieves a high cache hit rate because edges are contiguous in memory.

Bidirectional: stores both forward and backward edges so BFS can traverse in both directions.

Each `NodeInfo` and `EdgeInfo` has:
- `full_context: String` — verbose description, passed to the LLM when generating an answer
- `embed_context: Option<String>` — short dense description, used for embedding into HNSW. Falls back to `full_context`, then `name`/`edge_type` if absent.
- `image_url: Option<String>` (NodeInfo only) — URL or base64 data-URI. When set, `/ingest/json` routes this node through `/embed/image` instead of `/embed/text`. The resulting vector lands in the same HNSW index as text nodes — no separate image index needed.
- `external_id: String` (NodeInfo only) — stable public identifier, never reassigned at merge.

### HnswIndex (Node HNSW)

**Hierarchical Navigable Small World** — approximate nearest neighbor search for node embeddings.

- Layer 0: all nodes, fine-grained search
- Upper layers: subset of nodes, coarse navigation
- Search: greedy descent from top layer → beam search at layer 0 → top-K results
- Complexity: O(log n) vs O(n) brute-force
- Recall: ~95% with ef_construction=200

### EdgeHnswIndex (Edge HNSW)

A dedicated HNSW index for **edge embeddings**. Each edge has its `full_context` embedded — e.g. _"Works at"_ rather than just storing `"work_at"`.

- Search returns `(from_node_id, to_node_id, distance)` — the endpoint nodes of matching edges
- These endpoints are added to the BFS seed set, enabling discovery of the correct graph region via relationship semantics
- Empty index (no edges yet) → no-op search, no error

### VectorStore

Binary file memory-mapped into virtual memory:
```
[dim: u32][num_vecs: u32][vec_0: f32×dim][vec_1: f32×dim]...
```

`get(id)` = pointer arithmetic into the mmap region, zero-copy, no syscall.

Used for both `vectors.bin` (node embeddings) and `edge_vectors.bin` (edge embeddings).

### DeltaStore + WAL

**Write path**: new data → in-memory adjacency list + WAL on disk (crash recovery).

**Read path**: queries search both the main CsrGraph and the DeltaStore in parallel.

**Compaction**: when `delta.size() >= DELTA_MERGE_THRESHOLD`, an async rebuild is triggered. Pattern mirrors LSM-tree.

WAL entries for edges include `vec: [f32]` (edge embedding) to enable full replay after a crash.

### QueryCache

Caches complete query results by query string (moka concurrent LRU).
- `QUERY_CACHE_SIZE`: max entries (default 10000)
- `QUERY_CACHE_TTL_SECS`: TTL (default 300s)
- Cache hit: < 5ms (bypasses the entire pipeline)

### EmbedCache

Caches embedding vectors of hot nodes in RAM. Avoids re-reading from mmap on every scoring pass.
- `EMBED_CACHE_SIZE`: max entries (default 50000 ≈ 75MB with dim=384)

### Auth + Rate Limiter

- Single `MASTER_KEY` env var — token == MASTER_KEY → pass, else → 401
- Constant-time comparison (timing-safe XOR, no timing leak)
- Token bucket rate limiting per IP (public routes) and per key (protected routes)
- Unset `MASTER_KEY` = dev mode (no auth required)

---

## 5. Plugin System

A plugin is any HTTP server implementing the plugin interface. The reference setup uses two plugin processes:

| Plugin | Port | Endpoints |
|--------|------|-----------|
| Text plugin | 8001 | `POST /embed/text`, `POST /embed/image`, `POST /extract`, `POST /generate`, `POST /reason` |
| Image plugin | 8002 | `POST /store`, `GET /images/{file}`, `POST /embed` |

The Rust engine calls plugins via `plugin.rs` — an HTTP/Unix-socket client with a **TTL-cached health check**:
- Before every request **that uses the plugin**, the engine calls `check_ready()`
- The health result is cached for 5 seconds — avoids per-request overhead
- `/query`: checks only when `needs_embed` (no `vector` provided) **or** `needs_generate` (`llm_generate=true`). A pure graph query (`vector` + `llm_generate=false`) never returns 503
- `/ingest/text`, `/ingest/json`: always checks because embed is always called
- If the plugin is not ready → immediately return **HTTP 503**, without attempting to call the plugin

Each endpoint (embed_text, embed_image, image_store, extract, generate) can point to a different server or socket, configured independently in `plugins.toml`. The image plugin endpoints default to `http://localhost:8002`.

**Image auto-store**: when `[image] auto_store = true`, `ingest_json` calls the image plugin's `/store` before `/embed/image`. The stored stable URL (`/images/<sha256>.<ext>`) replaces the original `image_url` on the node, so the graph never stores ephemeral base64 blobs or URLs that might change. Falls back to the original URL if the store call fails.

See details: [`docs/PLUGIN_INTERFACE.md`](PLUGIN_INTERFACE.md)

---

## 6. Concurrency Model

```mermaid
graph TD
    RT["tokio async runtime\nmulti-thread"]

    RT --> HTTP["HTTP handlers\nmany concurrent requests"]
    RT --> STATE["AppState Arc"]
    RT --> QP["Query path\n4 read locks\nNodeHnsw ∥ EdgeHnsw via join!"]
    RT --> MT["Merge task\n4 write locks g→h→eh→st\nHNSW via spawn_blocking\nhot-swap all 4"]

    STATE --> G["Arc RwLock Arc CsrGraph"]
    STATE --> H["Arc RwLock Arc HnswIndex\nnode HNSW"]
    STATE --> EH["Arc RwLock Arc EdgeHnswIndex\nedge HNSW"]
    STATE --> DS["Arc RwLock Arc VectorStore"]

    note1["read lock: many queries concurrently\nwrite lock: only during hot-swap < 1ms"]
    STATE -.-> note1
```

In-flight queries hold an `Arc` reference to the old graph — they are never interrupted during a merge hot-swap. The old reference is dropped automatically when the query completes.

---

## 7. Persistence

| File | Format | Description |
|---|---|---|
| `nodes.json` | JSON array | Node metadata (id, name, type, props, weight, full_context, embed_context, external_id, image_url) |
| `edges.bin` | Binary CSR | offsets + edges + weights + edge_type indices |
| `edge_types.json` | JSON array | String labels for edge types |
| `edge_contexts.json` | JSON array | full_context string per edge, parallel to edges.bin |
| `edge_embed_contexts.json` | JSON array | embed_context per edge, parallel to edges.bin |
| `vectors.bin` | Binary | Header (dim, count) + f32 node embedding vectors |
| `edge_vectors.bin` | Binary | Header (dim, count) + f32 edge embedding vectors |
| `edge_endpoints.json` | JSON array | `[(from_id, to_id)]` parallel to edge_vectors.bin |
| `delta.wal` | Binary WAL | Write-ahead log for crash recovery (nodes + edges + vecs) |

On startup: load `nodes.json` + `edges.bin` → build CsrGraph; mmap `vectors.bin` → build HnswIndex; load `edge_vectors.bin` + `edge_endpoints.json` → build EdgeHnswIndex; replay `delta.wal` if there are unmerged entries.

---

## 8. Limitations and Trade-offs

| Issue | Current state | Resolution path |
|---|---|---|
| Rebuild HNSW on every merge | ~several seconds with 100k nodes (both node + edge HNSW); builds offloaded to blocking thread pool | Incremental HNSW (Phase 10+) |
| Single node | Writes are not replicated | Read-only replicas sharing `data/` |
| LLM latency | p50 ~600ms, dominated by LLM | Use a smaller model, cache responses |
