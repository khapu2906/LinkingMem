# LinkingMem — Graph-native RAG Engine

A high-performance Rust + Python engine for graph-based RAG, unifying vector search, graph traversal, and LLM reasoning in a single system.

```
Query → Embedding → HNSW retrieval → Graph expansion (BFS) → Ranking → LLM answer
```

LinkingMem combines vector search and graph traversal in one tightly integrated pipeline, enabling fast multi-hop reasoning, efficient memory usage, and production-ready scalability.

---

## Architecture

```
plugins/                    ← Python (all AI work)
  text/                       text plugin — standalone uv project
    main.py                     FastAPI: /embed /extract /generate /reason
    embed.py                    SentenceTransformers + ONNX backend option
    extract.py                  LLM entity extraction
    generate.py                 LLM answer generation
    reason.py                   Multi-hop reasoning (/reason)
    llm.py                      Multi-provider LLM client (gemini/openai/anthropic)
    schemas.py                  Pydantic models incl. LlmHints
    auth.py                     Bearer token auth (skipped for unix socket)
    pyproject.toml              uv-managed dependencies + dev extras

core/                       ← Rust (all compute)
  src/
    main.rs                   axum HTTP server  :8000
    app_state.rs              AppState bootstrap + hot-swap merge
    config.rs                 unified config (plugins.toml + env vars)
    entity_resolution.rs      embedding-based entity deduplication
    query.rs                  QueryEngine — full pipeline + result cache
    delta.rs                  LSM-style delta store + WAL  (fs2 file locking)
    cache.rs                  moka concurrent EmbedCache
    plugin.rs                 HTTP/Unix-socket client Rust→Python  (LlmHints)
    metrics.rs                Prometheus-compatible metrics
    graph/
      csr.rs                  CsrGraph — bidirectional CSR, BFS
      builder.rs              load/save binary + JSON ingest
    vector/
      store.rs                mmap VectorStore (zero-copy)
      hnsw.rs                 HNSW index (instant-distance)
    api/
      handlers/
        query.rs              POST /query  /query/text  /query/vector  /query/node  /query/multihop
        ingest.rs             POST /ingest/text  POST /ingest/json
        admin.rs              GET /health  /metrics  /graph/stats  /nodes  /nodes/:id
      dto/
        query.rs              QueryReq + ResponseProfile + QueryResponse + LlmHints
        ingest.rs             IngestTextReq + IngestJsonReq + IngestResponse
      error.rs                ApiError — unified error type
    middleware/auth.rs        API key auth + token-bucket rate limiter
    bin/ingest.rs             one-time data prep CLI

data/                       ← binary artefacts (git-ignored)
  nodes.json
  edges.bin
  edge_types.json
  vectors.bin
  delta.wal                   crash-recovery log

plugins.toml                ← plugin endpoint + query pipeline config
.env.example                ← all environment variables with defaults
```

---

## Quickstart

### 1. Prerequisites

```bash
# Rust 1.80+
curl https://sh.rustup.rs | sh

# Python 3.11+ with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configure

```bash
cp .env.example .env
# edit .env — set GEMINI_API_KEY at minimum
```

Plugin endpoints and query pipeline defaults are in `plugins.toml`.

### 3. Start Python plugin server

```bash
cd plugins/text
uv sync
GEMINI_API_KEY=your-key uv run uvicorn text.main:app --host 0.0.0.0 --port 8001 --reload
```

**Optional: swap embedding model** for better multilingual support:
```bash
EMBED_MODEL=BAAI/bge-m3 GEMINI_API_KEY=your-key uv run uvicorn text.main:app --port 8001
```

### 4. Start Rust engine

```bash
cd core
DATA_DIR=../data cargo run --bin server
```

### 5. Ingest data

**From raw text** (entity extraction via LLM):
```bash
curl -X POST http://localhost:8000/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"text": "Alice is the CEO of Acme Corp and works with Bob, the CTO."}'
```

**From structured JSON:**
```bash
curl -X POST http://localhost:8000/ingest/json \
  -H 'Content-Type: application/json' \
  -d '{
    "entities": [
      {"id": "e1", "name": "Alice", "type": "Person", "props": {"role": "CEO"}},
      {"id": "e2", "name": "Acme Corp", "type": "Company", "props": {}}
    ],
    "relations": [
      {"from": "e1", "to": "e2", "type": "works_at", "weight": 1.0}
    ]
  }'
```

**From file using CLI:**
```bash
cd core
cargo run --bin ingest -- --input ../data/your_data.json --data-dir ../data
```

**Docker (full stack):**
```bash
cp .env.example .env   # set GEMINI_API_KEY
docker-compose up
# localhost:8000  ← Rust engine
# localhost:8001  ← Python plugin
```

### 6. Query

```bash
# Default (balanced mode)
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Who works at Acme Corp?"}'

# Semantic search
curl ... -d '{"query": "...", "mode": "semantic"}'

# Relationship traversal
curl ... -d '{"query": "...", "mode": "relationship"}'

# Entity lookup (hub nodes ranked higher)
curl ... -d '{"query": "...", "mode": "entity"}'

# Control response verbosity
curl ... -d '{"query": "...", "response": "minimal"}'   # answer + total_ms only
curl ... -d '{"query": "...", "response": "standard"}'  # + nodes + key stats (default)
curl ... -d '{"query": "...", "response": "full"}'      # + edges + all timing stats

# Full custom options
curl ... -d '{
  "query": "...",
  "mode": "semantic",
  "response": "full",
  "options": {
    "hnsw_k": 30,
    "bfs_depth": 3,
    "bfs_max_nodes": 1000,
    "context_top_n": 80,
    "context_min_score": 0.2,
    "bidirectional": true,
    "weights": {"alpha": 0.6, "beta": 0.3, "gamma": 0.1}
  }
}'
```

---

## API Reference

### Ingest

| Method | Path           | Description                                        |
|--------|----------------|----------------------------------------------------|
| POST   | `/ingest/text` | Extract entities from raw text, add to graph       |
| POST   | `/ingest/json` | Add pre-structured entities + relations to graph   |

**`POST /ingest/text`** request:
```json
{
  "text": "Alice is the CEO of Acme Corp.",
  "resolution": {
    "mode": "embedding",
    "threshold": 0.92,
    "match_type": false
  }
}
```

**`POST /ingest/json`** request:
```json
{
  "entities": [{"id": "e1", "name": "Alice", "type": "Person", "props": {}}],
  "relations": [{"from": "e1", "to": "e2", "type": "works_at", "weight": 1.0}],
  "resolution": {"mode": "embedding"}
}
```

**Ingest response:**
```json
{
  "ingested": 3,
  "new_nodes": 2,
  "resolved": 1,
  "delta_size": 47,
  "merge_triggered": false
}
```

`resolved` counts entities that were matched to an existing node via embedding similarity (entity resolution). `merge_triggered` indicates whether a background graph rebuild was started.

### Query

| Method | Path     | Description                              |
|--------|----------|------------------------------------------|
| POST   | `/query` | Full RAG pipeline: embed → search → LLM |

**`POST /query`** request:
```json
{
  "query": "Who leads engineering at Acme Corp?",
  "mode": "balanced",
  "response": "standard",
  "options": {
    "hnsw_k": 20,
    "bfs_depth": 2,
    "bfs_max_nodes": 500,
    "context_top_n": 50,
    "context_min_score": 0.3,
    "bidirectional": false,
    "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
  }
}
```

All fields except `query` are optional. `mode` sets scoring weights; `options` provides fine-grained control and overrides both `mode` and server defaults.

**Response profiles:**

| `response`   | Fields returned                                        |
|--------------|--------------------------------------------------------|
| `"minimal"`  | `answer`, `stats.total_ms`, `stats.cache_hit`          |
| `"standard"` | + `subgraph.nodes`, `stats.seed_nodes`, `embed_ms`, `llm_ms` |
| `"full"`     | + `subgraph.edges`, all timing stats                   |

**`standard` response example:**
```json
{
  "answer": "Bob is the CTO of Acme Corp and leads the engineering team.",
  "subgraph": {
    "nodes": [
      {
        "id": 1,
        "name": "Bob",
        "type": "Person",
        "props": {"role": "CTO"},
        "score": 0.847,
        "vector_sim": 0.91,
        "hop": 0,
        "is_seed": true
      }
    ]
  },
  "stats": {
    "total_ms": 821,
    "cache_hit": false,
    "seed_nodes": 5,
    "subgraph_nodes": 23,
    "context_nodes": 12,
    "embed_ms": 38,
    "llm_ms": 743
  }
}
```

**`full` response** also includes:
```json
{
  "subgraph": {
    "nodes": [...],
    "edges": [
      {"from": "Bob", "to": "Acme Corp", "weight": 1.0, "edge_type": "works_at"}
    ]
  },
  "stats": {
    "total_ms": 821, "cache_hit": false,
    "seed_nodes": 5, "subgraph_nodes": 23, "context_nodes": 12,
    "embed_ms": 38, "search_ms": 1, "bfs_ms": 0, "score_ms": 2, "llm_ms": 743
  }
}
```

### Management

| Method | Path            | Auth | Description                               |
|--------|-----------------|------|-------------------------------------------|
| GET    | `/health`       | —    | Liveness check + summary metrics          |
| GET    | `/metrics`      | —    | Prometheus text format                    |
| GET    | `/graph/stats`  | —    | Current graph + delta node/edge counts    |
| GET    | `/nodes`        | —    | Search nodes by name/type                 |
| GET    | `/nodes/:id`    | —    | Fetch single node by numeric ID           |
| POST   | `/delta/merge`  | key  | Force an immediate graph rebuild          |

**`GET /graph/stats`:**
```json
{
  "nodes": 1240,
  "edges": 4832,
  "delta_nodes": 7,
  "delta_edges": 12,
  "delta_pending_merge": false
}
```

**`GET /nodes?q=alice&type=Person&limit=20`:**
```json
{
  "nodes": [{"id": 5, "name": "Alice", "type": "Person", "props": {}, "weight": 0.83}],
  "count": 1
}
```
Parameters: `q` (name substring, case-insensitive), `type` (exact entity type), `limit` (max 500, default 50).

---

## Entity Resolution

When new data is ingested, the engine checks each new entity against the existing graph using embedding cosine similarity. If a match exceeds the threshold, the new entity is merged into the existing node instead of creating a duplicate.

```
New entity "Acme Corporation" (embedding: [...])
    ↓
HNSW search: nearest existing nodes
    ↓
cosine_sim("Acme Corp") = 0.97 > threshold 0.92 → MERGE
cosine_sim("Bob Smith") = 0.12 < threshold      → skip
    ↓
resolved_count += 1, no new node created
```

Resolution is also applied within a single ingest batch (intra-batch dedup).

**Configure per-request:**
```json
{
  "text": "...",
  "resolution": {
    "mode": "embedding",
    "threshold": 0.85,
    "match_type": true
  }
}
```

| Field       | Default      | Description                                          |
|-------------|--------------|------------------------------------------------------|
| `mode`      | `"embedding"` | `"embedding"` or `"none"` (disable)                |
| `threshold` | `0.92`       | Cosine similarity cutoff. Lower = more aggressive merging. |
| `match_type`| `false`      | If `true`, entities must share the same type to merge. |

---

## Scoring Formula

```
score(node) = α·vector_sim + β·graph_proximity + γ·node_weight

graph_proximity = edge_weight / (hop + 1)     (1.0 for seed nodes)
node_weight     = (in_degree + out_degree) / max_combined
```

| Mode             | α    | β    | γ    | Use case                              |
|------------------|------|------|------|---------------------------------------|
| `balanced`       | 0.50 | 0.30 | 0.20 | General RAG (default)                 |
| `semantic`       | 0.70 | 0.20 | 0.10 | Meaning-based / similarity search     |
| `relationship`   | 0.30 | 0.60 | 0.10 | Graph traversal, find connections     |
| `entity`         | 0.40 | 0.20 | 0.40 | Important hub nodes, named entity lookup |

---

## Performance (typical, single node)

| Metric              | Value                          |
|---------------------|--------------------------------|
| Vector search       | ~1–2 ms (HNSW, 1M vectors)     |
| BFS traversal       | ~0.5 ms (CSR, depth=2)         |
| Scoring 500 nodes   | ~2 ms                          |
| Embed query         | ~30–50 ms (Python)             |
| LLM generate        | ~500–1500 ms                   |
| **Total p50**       | **~600 ms**                    |
| **Cache hit**       | **< 5 ms**                     |
| RAM (1M nodes)      | ~800 MB (graph + HNSW + cache) |

---

## Environment Variables

Full list with defaults is in `.env.example`. Key variables:

**Server:**

| Variable             | Default                         | Description                      |
|----------------------|---------------------------------|----------------------------------|
| `BIND_ADDR`          | `0.0.0.0:8000`                  | Listen address                   |
| `RUST_LOG`           | `info`                          | Log level                        |
| `DATA_DIR`           | `<next to binary>/data`         | Path to binary data files        |
| `PLUGIN_CONFIG_FILE` | `<next to binary>/plugins.toml` | Plugin endpoint config file      |

**Query pipeline (server defaults, overridable per-request):**

| Variable             | Default | Description                                |
|----------------------|---------|--------------------------------------------|
| `HNSW_K`             | `20`    | Seed nodes from vector search              |
| `BFS_DEPTH`          | `2`     | BFS expansion depth                        |
| `BFS_MAX_NODES`      | `500`   | Max nodes collected during BFS             |
| `CONTEXT_TOP_N`      | `50`    | Top-N nodes sent to LLM                    |
| `CONTEXT_MIN_SCORE`  | `0.3`   | Minimum relevance score for LLM context    |

**Entity resolution:**

| Variable                  | Default      | Description                           |
|---------------------------|--------------|---------------------------------------|
| `RESOLUTION_MODE`         | `embedding`  | `embedding` or `none`                 |
| `RESOLUTION_THRESHOLD`    | `0.92`       | Cosine similarity cutoff              |
| `RESOLUTION_MATCH_TYPE`   | `false`      | Require same type to merge            |

**Cache & delta:**

| Variable               | Default | Description                               |
|------------------------|---------|-------------------------------------------|
| `EMBED_CACHE_SIZE`     | `50000` | Max cached embedding vectors              |
| `QUERY_CACHE_SIZE`     | `10000` | Max cached query results                  |
| `QUERY_CACHE_TTL_SECS` | `300`   | Query cache TTL in seconds                |
| `DELTA_MERGE_THRESHOLD`| `500`   | Trigger graph rebuild after N writes      |

**Plugin:**

| Variable               | Default                | Description                            |
|------------------------|------------------------|----------------------------------------|
| `PLUGIN_URL`           | `http://localhost:8001`| Fallback plugin URL                    |
| `PLUGIN_TIMEOUT_SECS`  | `60`                   | HTTP timeout for plugin calls          |
| `GEMINI_API_KEY`       | —                      | Required for extract + generate        |
| `EMBED_MODEL`          | `all-MiniLM-L6-v2`     | SentenceTransformers model             |
| `EXTRACT_MODEL`        | `gemini-2.5-flash-lite`| Gemini model for entity extraction     |
| `GENERATE_MODEL`       | `gemini-2.5-flash-lite`| Gemini model for answer generation     |

**Auth:**

| Variable               | Default                   | Description                          |
|------------------------|---------------------------|--------------------------------------|
| `API_KEYS`             | _(empty = auth disabled)_ | Comma-separated valid API keys       |
| `RATE_LIMIT_PER_MINUTE`| `60`                      | Requests per minute per key          |
| `RATE_LIMIT_BURST`     | `10`                      | Burst allowance                      |

---

## Plugin System

Plugins are any HTTP server that implements the interface in `docs/PLUGIN_INTERFACE.md`. Any language works — Python, Go, Node.js, etc.

Each plugin operation (embed / extract / generate) can point to a different server:

```toml
# plugins.toml
[embed]
url = "http://embed-server:8001"

[generate]
url        = "http://llm-server:8002"
auth_token = "my-secret"          # Bearer token for HTTP endpoints

# Unix socket alternative (same machine, lower latency, auth skipped):
# [embed]
# socket = "/tmp/plugin-embed.sock"
```

---

## Roadmap

| Phase | Status | Goal                                              |
|-------|--------|---------------------------------------------------|
| 1     | ✅     | Python pipeline + brute vector search             |
| 2     | ✅     | CSR graph + HNSW + mmap vector store              |
| 3     | ✅     | Delta store + WAL + hot-swap graph                |
| 4     | ✅     | Query cache + metrics + auth + rate limiter       |
| 5     | ✅     | Plugin interface spec + multi-language support    |
| 6     | ✅     | Entity resolution (embedding cosine similarity)   |
| 7     | ✅     | DTO validation + ResponseProfile + unified errors |
| 8     | ✅     | `edge_type` end-to-end + node search API          |
| 9     | ✅     | Unix socket transport for plugins (hyper direct)  |
| 10    | ✅     | Multi-hop reasoning (`/query/multihop`, `/reason`)|
| 11    | ✅     | Distributed ingest (fs2 WAL locking, instance IDs)|
