# API Reference — LinkingMem

Base URL: `http://localhost:8000`

**Auth**: Protected routes require `Authorization: Bearer <api_key>` or `X-API-Key: <api_key>`.
Set `API_KEYS` env var (comma-separated). If not set → auth disabled (dev mode).

---

## Endpoint Overview

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/query/text` | Required | Text query: embed → HNSW → BFS → score → LLM |
| POST | `/query/vector` | Required | Vector query: HNSW → BFS → score → [LLM], embed step skipped |
| POST | `/query/node` | Required | Node query: BFS from node_id → score → [LLM], no HNSW |
| POST | `/query/image` | Required | Image query: vision caption → embed → HNSW → BFS → score → LLM |
| POST | `/query/multihop` | Required | Multi-hop reasoning: iterative graph expansion until LLM has enough context |
| POST | `/query` | Required | Legacy — selects text or vector mode based on fields present |
| POST | `/ingest/text` | Required | LLM entity extraction from raw text, then ingest |
| POST | `/ingest/json` | Required | Ingest pre-structured entities + relations (supports image nodes via `image_url`) |
| POST | `/delta/merge` | Required | Force immediate graph rebuild |
| GET | `/health` | Public | Liveness check + summary |
| GET | `/metrics` | Public | Prometheus text format |
| GET | `/graph/stats` | Public | Node/edge counts (main graph + delta) |
| GET | `/nodes` | Public | Search nodes by name, type, relation, or edge type |
| GET | `/nodes/:id` | Public | Fetch node by numeric ID |
| GET | `/edges` | Public | Search edges by from, to, or edge type |

**Which endpoint to use:**

| Endpoint | Use when |
|---|---|
| `/query/text` | Client sends a query string; engine handles embedding |
| `/query/vector` | Client already has an embedding (batch pipelines, vector reuse) |
| `/query/node` | Client knows a specific node_id and wants to explore its neighbourhood |
| `/query/image` | Client has an image (URL or base64) and wants to search the graph by its visual content |
| `/query/multihop` | Query may require following multiple entity relationships to answer |
| `/query` (legacy) | Backward compat — equivalent to `/query/text` or `/query/vector` |

---

## POST /query/text

Full pipeline: embed query → HNSW search (node + edge in parallel) → BFS expand → score → [LLM].

The plugin is always checked (embed is required). If `llm_generate=false`, only the embed step needs the plugin.

### Request

```json
{
  "query": "Who leads the engineering team at Acme Corp?",
  "pipeline": { "llm_generate": true },
  "mode": "balanced",
  "response": "standard",
  "options": {
    "hnsw_k": 20,
    "bfs_depth": 2,
    "bfs_max_nodes": 500,
    "context_top_n": 50,
    "context_min_score": 0.3,
    "bidirectional": false,
    "weights": { "alpha": 0.5, "beta": 0.3, "gamma": 0.2 }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | **yes** | Query string, 1–4096 characters |
| `pipeline.llm_generate` | bool | no | `false` = skip LLM. Default: `true` |
| `pipeline.hints` | object | no | Per-request LLM customization — see [LlmHints](#llmhints) |
| `mode` | string | no | Scoring preset — see table below |
| `response` | string | no | Verbosity: `minimal` `standard` `full` |
| `options` | object | no | Fine-grained pipeline override — see options table |

---

## POST /query/vector

HNSW search → BFS → score → [LLM]. Embed step is skipped entirely.

The plugin is only checked when `llm_generate=true`. A pure graph query (`llm_generate=false`) never returns 503.

### Request

```json
{
  "vector": [0.12, -0.45, 0.33, "..."],
  "pipeline": { "llm_generate": true, "prompt": "Who leads engineering?" },
  "mode": "balanced",
  "response": "standard",
  "options": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vector` | array[float] | **yes** | Pre-computed embedding; must match the server's vector dimension |
| `pipeline.llm_generate` | bool | no | Default: `true` |
| `pipeline.prompt` | string | no | Question for the LLM. Only used when `llm_generate=true` |
| `pipeline.hints` | object | no | Per-request LLM customization — see [LlmHints](#llmhints) |
| `mode` | string | no | Scoring preset |
| `response` | string | no | Verbosity |
| `options` | object | no | Fine-grained override |

---

## POST /query/node

Direct BFS from `node_id` → score. No embed, no HNSW, no LLM by default.

Use when the client already knows a specific node (e.g. from `/nodes?q=...`) and wants to explore its neighbourhood.

The seed node is assigned `vector_sim=1.0`. Expanded nodes are scored by proximity + node_weight.
`context_min_score` defaults to `0.0` to return the full neighbourhood. Never calls the plugin.

### Request

```json
{
  "node_id": 5,
  "mode": "relationship",
  "response": "full",
  "options": { "bfs_depth": 3 }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | int | **yes** | Numeric node ID (from `/nodes` or `/nodes/:id`) |
| `mode` | string | no | Scoring preset. Default `balanced` |
| `response` | string | no | Verbosity |
| `options` | object | no | Fine-grained override |

### Error

```json
{ "error": "node 999 not found", "code": "not_found" }
```
HTTP 404 if `node_id` does not exist.

---

## POST /query/image

Query the graph from an image input. The image plugin generates a text caption via Vision LLM, embeds it in the shared vector space, then runs the standard HNSW → BFS → score → LLM pipeline.

This enables cross-modal search: "find nodes related to this photo" returns the same result format as `/query/text`.

### Request

```json
{
  "image_url": "https://example.com/product.jpg",
  "pipeline": { "llm_generate": true },
  "mode": "semantic",
  "response": "standard",
  "options": {
    "hnsw_k": 20,
    "bfs_depth": 2
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_url` | string | **yes** | `http(s)://` URL or `data:<mime>;base64,<payload>` data-URI. Max 4096 characters |
| `pipeline.llm_generate` | bool | no | Default: `true` |
| `pipeline.hints` | object | no | Per-request LLM customization — see [LlmHints](#llmhints) |
| `mode` | string | no | Scoring preset |
| `response` | string | no | Verbosity: `minimal` `standard` `full` |
| `options` | object | no | Fine-grained pipeline override |

Response format is identical to `/query/text`.

Requires the **image plugin** to be running (default: `http://localhost:8002`) and configured at `[plugins.embed_image]` in `plugins.toml`.

---

## POST /query/multihop

Multi-hop reasoning: iteratively expand the graph until the LLM has enough context to answer, or until `max_hops` is reached.

Pipeline:
1. Embed query → HNSW search → BFS → score (same as `/query/text`)
2. Call `/reason` on the plugin — LLM decides whether context is sufficient
3. If `done=false`: embed `follow_ups` entity names → expand context → repeat
4. Final call to `/generate` with the accumulated context

### Request

```json
{
  "query": "What technologies does the AI team at Acme Corp use?",
  "max_hops": 2,
  "mode": "balanced",
  "response": "standard",
  "options": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | **yes** | Query string, 1–4096 characters |
| `max_hops` | int | no | Maximum expansion iterations, 1–5. Default: `2` |
| `mode` | string | no | Scoring preset |
| `response` | string | no | Verbosity |
| `options` | object | no | Fine-grained override |

Response format is identical to `/query/text`. Query cache is **not** applied to multihop (each hop may produce a different context).

---

## Common Parameters (mode, options, response)

Applies to all query endpoints.

**`mode` — scoring weights (α · vector_sim + β · graph_proximity + γ · node_weight):**

| Mode | α | β | γ | Use when |
|------|---|---|---|----------|
| `balanced` | 0.50 | 0.30 | 0.20 | General RAG (default) |
| `semantic` | 0.70 | 0.20 | 0.10 | Meaning-based / similarity search |
| `relationship` | 0.30 | 0.60 | 0.10 | Tracing connections between entities |
| `entity` | 0.40 | 0.20 | 0.40 | Finding important hub nodes |

**`options` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hnsw_k` | int | 20 | Seed nodes retrieved from each HNSW search (node + edge) |
| `bfs_depth` | int | 2 | BFS expansion depth from seed nodes |
| `bfs_max_nodes` | int | 500 | Max nodes collected during BFS |
| `context_top_n` | int | 50 | Top-N nodes passed to the LLM as context |
| `context_min_score` | float | 0.3 (0.0 for `/query/node`) | Minimum score to include a node in LLM context |
| `bidirectional` | bool | false | Traverse edges in both directions during BFS |
| `weights.alpha` | float | — | Override α (vector_sim weight) |
| `weights.beta` | float | — | Override β (graph_proximity weight) |
| `weights.gamma` | float | — | Override γ (node_weight weight) |

## Response Format

Applies to all query endpoints.

**`response: "minimal"`** — answer + timing only:
```json
{
  "answer": "Bob is the CTO of Acme Corp and leads engineering.",
  "stats": {
    "total_ms": 821,
    "cache_hit": false
  }
}
```

**`response: "standard"` (default)** — adds nodes + key stats:
```json
{
  "answer": "Bob is the CTO of Acme Corp and leads engineering.",
  "subgraph": {
    "nodes": [
      {
        "id": 2,
        "name": "Bob",
        "type": "Person",
        "props": { "role": "CTO" },
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

**`response: "full"`** — adds edges + full timing breakdown:
```json
{
  "answer": "...",
  "subgraph": {
    "nodes": ["..."],
    "edges": [
      { "from": "Bob", "to": "Acme Corp", "weight": 1.0, "edge_type": "works_at" }
    ]
  },
  "stats": {
    "total_ms": 821,
    "cache_hit": false,
    "seed_nodes": 5,
    "subgraph_nodes": 23,
    "context_nodes": 12,
    "embed_ms": 38,
    "search_ms": 2,
    "bfs_ms": 0,
    "score_ms": 2,
    "llm_ms": 743
  }
}
```

**Node fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Numeric node ID |
| `name` | string | Entity name |
| `type` | string | Entity type (Person, Company, …) |
| `props` | object | Arbitrary metadata |
| `score` | float | Composite relevance score (0–1) |
| `vector_sim` | float | Cosine similarity with query (0–1) |
| `hop` | int | BFS distance from seed (0 = seed node) |
| `is_seed` | bool | Directly selected by HNSW (or the seed node_id for `/query/node`) |

**Cache hit**: when `cache_hit: true`, `total_ms` < 5 ms and no timing breakdown is included.

---

## LlmHints

Per-request LLM customization. Applies to `pipeline.hints` (query endpoints) and `hints` (`/ingest/text`). All fields are optional.

```json
{
  "pipeline": {
    "hints": {
      "system_prompt": "You are a concise assistant. Answer in one sentence.",
      "rules": ["Never mention entity IDs", "Respond in the same language as the question"],
      "extend_context": ["The company was founded in 2015", "HQ is in Hanoi"]
    }
  }
}
```

| Field | Type | Applies to | Effect |
|-------|------|------------|--------|
| `system_prompt` | string | extract / generate / reason | Replaces the default system prompt entirely |
| `rules` | array[string] | extract / generate / reason | Appended as extra rules after the default rules block |
| `extend_context` | array[string] | generate / reason | Extra text snippets injected into the prompt context before the question |

**Note:** when `hints` is non-null, the query cache is bypassed automatically.

---

## POST /ingest/text

Extract entities and relations from raw text using the LLM, then ingest them into the graph.

### Request

```json
{
  "text": "Alice is the CEO of Acme Corp and works with Bob, the CTO.",
  "resolution": {
    "mode": "embedding",
    "threshold": 0.92,
    "match_type": false
  },
  "hints": {
    "rules": ["Focus on people and organisations only", "Ignore dates"]
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Raw text to extract entities from |
| `resolution.mode` | string | no | `"embedding"` (default) or `"none"` |
| `resolution.threshold` | float | no | Cosine similarity cutoff (default 0.92) |
| `resolution.match_type` | bool | no | Require same entity type to merge (default false) |
| `hints` | object | no | LLM extraction customization — see [LlmHints](#llmhints) |

The plugin assigns `full_context` to each entity and relation automatically.

### Response

```json
{
  "ingested": 3,
  "new_nodes": 2,
  "resolved": 1,
  "delta_size": 47,
  "merge_triggered": false
}
```

| Field | Description |
|-------|-------------|
| `ingested` | Total entities processed |
| `new_nodes` | Nodes actually created |
| `resolved` | Entities merged into an existing node (entity resolution) |
| `delta_size` | Entries in the delta buffer after ingest |
| `merge_triggered` | `true` if a background graph rebuild was triggered |

---

## POST /ingest/json

Ingest pre-structured entities + relations directly (no LLM extraction).

### Request

```json
{
  "entities": [
    {
      "id": "e1",
      "name": "Alice",
      "type": "Person",
      "props": { "role": "CEO" },
      "full_context":  "Alice is the CEO of Acme Corp, overseeing strategy and operations.",
      "embed_context": "Alice, CEO at Acme Corp"
    },
    {
      "id": "e2",
      "name": "Acme Corp",
      "type": "Company",
      "props": {},
      "full_context":  "Acme Corp is a technology company based in Silicon Valley, founded in 2010.",
      "embed_context": "Acme Corp, Silicon Valley tech company"
    },
    {
      "id": "e3",
      "name": "Office photo",
      "type": "Image",
      "props": {},
      "full_context":  "A photo of the Acme Corp main office.",
      "image_url": "https://example.com/office.jpg"
    }
  ],
  "relations": [
    {
      "from": "e1",
      "to": "e2",
      "type": "works_at",
      "weight": 1.0,
      "full_context":  "Alice has worked at Acme Corp as CEO since 2018.",
      "embed_context": "Alice is CEO at Acme Corp"
    }
  ],
  "resolution": {
    "mode": "embedding",
    "threshold": 0.92
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entities` | array | yes | List of entities to ingest |
| `entities[].id` | string | yes | Unique ID within this batch |
| `entities[].name` | string | yes | Entity display name |
| `entities[].type` | string | yes | Entity type label |
| `entities[].props` | object | yes | Metadata; use `{}` if none |
| `entities[].full_context` | string | no | Verbose description — included in LLM context when generating answers |
| `entities[].embed_context` | string | no | Short dense text for NodeHnswIndex embedding. Falls back to `full_context` then `name` |
| `entities[].image_url` | string | no | `http(s)://` URL or `data:<mime>;base64,...` URI. When present, the engine calls `/embed/image` (Vision LLM caption → embed) instead of `/embed/text` for this node. The resulting vector lives in the same space as text nodes. |
| `relations` | array | yes | List of relations |
| `relations[].from` | string | yes | Source entity ID |
| `relations[].to` | string | yes | Target entity ID |
| `relations[].type` | string | yes | Relation label |
| `relations[].weight` | float | yes | Confidence/strength (0.0–1.0) |
| `relations[].full_context` | string | no | Verbose description of the relation — included in LLM context |
| `relations[].embed_context` | string | no | Short dense text for EdgeHnswIndex embedding. Falls back to `full_context` then `type` |
| `resolution` | object | no | Same as `/ingest/text` |

**Image node auto-store**: when `[image] auto_store = true` in `plugins.toml` (or `IMAGE_AUTO_STORE=true`), the engine sends the image to the image plugin's `/store` endpoint before embedding. The stored stable URL replaces `image_url` on the node — external URLs and raw base64 blobs are both converted to permanent `http://localhost:8002/images/<sha256>.<ext>` URLs. When `auto_store = false` (default), the original URL or data-URI is used directly for embedding without being persisted.

### Response

Same as `/ingest/text`.

---

## POST /delta/merge

Force an immediate graph rebuild instead of waiting for the auto-trigger threshold.

### Request

No body.

### Response

```json
{ "ok": true }
```

Merge runs synchronously — response is returned after the rebuild completes.

---

## GET /health

Liveness check.

### Response

```json
{
  "status": "ok",
  "nodes": 1240,
  "edges": 4832,
  "delta_size": 7
}
```

---

## GET /metrics

Prometheus text format. Use with a Prometheus scrape job.

```
# HELP queries_total Total number of queries processed
# TYPE queries_total counter
queries_total 1523

# HELP query_latency_ms Query latency histogram
# TYPE query_latency_ms histogram
query_latency_ms_bucket{le="100"} 12
query_latency_ms_bucket{le="500"} 891
query_latency_ms_bucket{le="1000"} 1421
...
```

Available metrics:
- `queries_total`, `queries_failed_total`
- `query_latency_ms` histogram (p50/p95/p99)
- `cache_hits_total`, `cache_misses_total`
- `delta_size`, `graph_nodes`, `graph_edges`

---

## GET /graph/stats

### Response

```json
{
  "nodes": 1240,
  "edges": 4832,
  "delta_nodes": 7,
  "delta_edges": 12,
  "delta_pending_merge": false
}
```

`delta_pending_merge: true` while a background merge is running.

---

## GET /nodes

Search nodes by name, type, relation, or edge type.

### Query parameters

| Param | Type | Description |
|-------|------|-------------|
| `q` | string | Substring search on name (case-insensitive) |
| `type` | string | Exact entity type match |
| `related_to` | int | Return only nodes connected to this node ID |
| `direction` | string | Edge direction for `related_to`: `out` (default) \| `in` \| `both` |
| `edge_type` | string | Filter by edge type (combine with `related_to` or use standalone) |
| `limit` | int | Max results; default 50, max 500 |

**Examples:**
```
GET /nodes?q=alice&type=Person
GET /nodes?related_to=5&direction=out
GET /nodes?related_to=5&edge_type=works_at
GET /nodes?related_to=5&direction=both&edge_type=manages
GET /nodes?edge_type=works_at
```

When `edge_type` is used without `related_to`: scans all edges O(E) — use `limit` to control result size.

### Response

```json
{
  "nodes": [
    { "id": 5, "name": "Alice", "type": "Person", "props": { "role": "CEO" }, "weight": 0.83 }
  ],
  "count": 1
}
```

---

## GET /nodes/:id

Fetch a node by its numeric ID.

Example: `GET /nodes/5`

### Response

```json
{ "id": 5, "name": "Alice", "type": "Person", "props": { "role": "CEO" }, "weight": 0.83 }
```

HTTP 404 if not found.

---

## GET /edges

Search edges by node or edge type. At least one of `from`, `to`, or `type` is required.

### Query parameters

| Param | Type | Description |
|-------|------|-------------|
| `from` | int | Outgoing edges from this node ID |
| `to` | int | Incoming edges to this node ID |
| `type` | string | Filter by edge type (case-insensitive) |
| `limit` | int | Max results; default 100, max 1000 |

**Examples:**
```
GET /edges?from=5
GET /edges?to=5&type=works_at
GET /edges?from=2&to=5
GET /edges?type=manages
```

When `type` is used without `from`/`to`: scans all edges O(E).

### Response

```json
{
  "edges": [
    {
      "from_id": 2,
      "from_name": "Alice",
      "to_id": 5,
      "to_name": "Acme Corp",
      "weight": 1.0,
      "edge_type": "works_at"
    }
  ],
  "count": 1
}
```

HTTP 400 if no filter is provided. HTTP 404 if `from`/`to` node does not exist.

---

## Error Responses

All errors return JSON:

```json
{ "error": "<human-readable message>", "code": "<error_code>" }
```

| Status | Code | When |
|--------|------|------|
| 400 | `bad_request` | Invalid request (missing field, wrong type, empty query, …) |
| 401 | — | Missing or invalid API key |
| 422 | `unprocessable_entity` | Valid request body but contents cannot be processed |
| 429 | — | Rate limit exceeded |
| 404 | — | Node not found |
| 502 | `bad_gateway` | Plugin call failed (embed / extract / generate error) |
| 503 | `plugin_unavailable` | Plugin server not ready. For `/query`: only possible when the pipeline actually needs the plugin (`vector` absent or `llm_generate=true`). A pure graph query (`vector` + `llm_generate=false`) never returns 503. |
| 500 | `internal_error` | Unexpected internal error |

---

## Entity Resolution

At ingest time, the engine checks each new entity against the existing graph using embedding cosine similarity.

```
New entity "Acme Corporation"
    ↓ HNSW search: nearest existing nodes
    ↓ cosine_sim("Acme Corp") = 0.97 > threshold 0.92 → MERGE (no new node created)
    ↓ cosine_sim("Bob Smith") = 0.12 < threshold      → skip
```

Also applied within a single ingest batch (intra-batch dedup).

Configure via `resolution` in the request body or env vars:
- `RESOLUTION_MODE` (default: `embedding`)
- `RESOLUTION_THRESHOLD` (default: `0.92`)
- `RESOLUTION_MATCH_TYPE` (default: `false`)
