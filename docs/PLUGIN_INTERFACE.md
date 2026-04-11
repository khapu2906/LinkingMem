# Plugin Interface Specification

This document defines the contract between the **AI Graph Engine core** (Rust) and any **plugin** (any language). A plugin is any HTTP server that implements the endpoints below.

The core communicates with plugins over:
- **HTTP** — for remote plugins or cross-machine deployments
- **Unix socket** — for local plugins on the same machine (lower latency, zero auth overhead)

The wire format is always **JSON over HTTP/1.1**.

---

## Transport

### HTTP
```
Base URL:  http://<host>:<port>
```

### Unix Socket
```
Socket:    /tmp/plugin-<name>.sock
Requests:  same HTTP format, host header is ignored
```

Configure in `plugins.toml`:
```toml
[embed_text]
socket = "/tmp/LinkingMem-embed.sock"   # unix socket — auth is skipped entirely

[generate]
url        = "http://localhost:8002"
auth_token = "my-secret"                      # HTTP — Bearer token required
```

The plugin must listen on at least one transport. Unix socket auth is skipped (transport-level security); HTTP auth uses `Authorization: Bearer <auth_token>` when `auth_token` is set in `plugins.toml`.

---

## Endpoints

### `GET /health`

Liveness check. The core calls this:
- At startup to confirm the plugin is ready
- Before every request that needs the plugin (embed/extract/generate), using a **5-second TTL cache**

**Request**: no body.

**Response `200 OK`**:
```json
{ "status": "ok" }
```

**Response `503 Service Unavailable`** (plugin not ready):
```json
{ "status": "loading", "reason": "model not yet loaded" }
```

If the health check fails (non-200 or `"status" != "ok"`), the core returns `503 plugin_unavailable` to the caller immediately — it does not attempt embed/extract/generate.

The TTL cache means at most one health probe per 5 seconds, so there is no per-request overhead in the steady state.

---

### `POST /embed/text`

Convert texts into dense vector embeddings.

**Request**:
```json
{
  "texts": ["Alice works at Acme Corp.", "Bob leads the engineering team."]
}
```

| Field   | Type             | Required | Description                  |
|---------|------------------|----------|------------------------------|
| `texts` | `array[string]`  | yes      | Batch of texts to embed. Min 1, max 512 items. |

**Response `200 OK`**:
```json
{
  "vectors": [
    [0.021, -0.143, 0.887, "..."],
    [0.034,  0.201, 0.756, "..."]
  ],
  "dim":   384,
  "model": "all-MiniLM-L6-v2"
}
```

| Field     | Type                    | Description                                           |
|-----------|-------------------------|-------------------------------------------------------|
| `vectors` | `array[array[float32]]` | One vector per input text, same order as `texts`.     |
| `dim`     | `integer`               | Dimension of each vector. Must be consistent across all calls. |
| `model`   | `string`                | Model name used for this request (informational, ignored by core). |

**Constraints**:
- `vectors[i]` must correspond to `texts[i]`.
- All vectors must have the same dimension.
- Vectors should be **unit-normalised** (L2 norm ≈ 1.0). The core uses cosine similarity.
- `dim` must not change between calls for the same plugin instance.

The core calls `/embed/text` for **both** node texts and edge texts concurrently (two parallel requests per ingest). The batch size for each call is bounded by the number of entities or relations in the request.

**Error `400 Bad Request`**:
```json
{ "error": "texts array is empty" }
```

---

### `POST /extract`

Extract entities and relations from a text passage.

**Request**:
```json
{
  "text": "Alice is the CEO of Acme Corp and works with Bob, the CTO."
}
```

| Field  | Type     | Required | Description           |
|--------|----------|----------|-----------------------|
| `text` | `string` | yes      | Raw text to analyse.  |
| `hints` | `object` | no      | Per-request LLM customization — see [LlmHints](#llmhints). |

**Response `200 OK`**:
```json
{
  "entities": [
    {
      "id": "e1",
      "name": "Alice",
      "type": "Person",
      "props": { "role": "CEO" },
      "full_context":  "Alice là CEO của Acme Corp, phụ trách chiến lược và vận hành toàn công ty",
      "embed_context": "Alice, CEO tại Acme Corp"
    },
    {
      "id": "e2",
      "name": "Bob",
      "type": "Person",
      "props": { "role": "CTO" },
      "full_context":  "Bob là CTO của Acme Corp, phụ trách kỹ thuật và sản phẩm",
      "embed_context": "Bob, CTO tại Acme Corp"
    },
    {
      "id": "e3",
      "name": "Acme Corp",
      "type": "Company",
      "props": {},
      "full_context":  "Acme Corp là công ty công nghệ có trụ sở tại Silicon Valley",
      "embed_context": "Acme Corp, công ty công nghệ Silicon Valley"
    }
  ],
  "relations": [
    {
      "from": "e1", "to": "e3",
      "type": "works_at", "weight": 1.0,
      "full_context":  "Alice làm việc tại Acme Corp với vai trò CEO",
      "embed_context": "Alice là CEO tại Acme Corp"
    },
    {
      "from": "e2", "to": "e3",
      "type": "works_at", "weight": 1.0,
      "full_context":  "Bob làm việc tại Acme Corp với vai trò CTO",
      "embed_context": "Bob là CTO tại Acme Corp"
    },
    {
      "from": "e1", "to": "e2",
      "type": "collaborates_with", "weight": 0.7,
      "full_context":  "Alice và Bob cộng tác chặt chẽ trong việc điều hành Acme Corp",
      "embed_context": "Alice cộng tác với Bob tại Acme Corp"
    }
  ]
}
```

**Entity object**:

| Field          | Type     | Required | Description                                                                        |
|----------------|----------|----------|------------------------------------------------------------------------------------|
| `id`           | `string` | yes      | Unique within this response. Used as key in `relations`.                           |
| `name`         | `string` | yes      | Display name.                                                                      |
| `type`         | `string` | yes      | Entity type label e.g. `Person`, `Company`, `Product`.                            |
| `props`        | `object` | yes      | Arbitrary key-value metadata. Use `{}` if none.                                    |
| `full_context`  | `string` | yes      | Verbose description for LLM answer generation. Can be long.                        |
| `embed_context` | `string` | no       | Short dense text (1–2 sentences) for NodeHnswIndex embedding. Falls back to `full_context` then `name` if absent. Recommended to include. |

**Relation object**:

| Field           | Type     | Required | Description                                                                              |
|-----------------|----------|----------|------------------------------------------------------------------------------------------|
| `from`          | `string` | yes      | `id` of the source entity.                                                               |
| `to`            | `string` | yes      | `id` of the target entity.                                                               |
| `type`          | `string` | yes      | Relation label e.g. `works_at`, `leads`, `uses`.                                        |
| `weight`        | `float`  | yes      | Confidence score in `[0.0, 1.0]`.                                                        |
| `full_context`  | `string` | yes      | Verbose description of the relation for LLM context.                                    |
| `embed_context` | `string` | no       | Short dense text for EdgeHnswIndex embedding. Falls back to `full_context` then `type` if absent. Recommended to include. |

**Constraints**:
- `from` and `to` must reference `id` values present in `entities`.
- `weight` must be in `[0.0, 1.0]`.
- Empty `entities` and `relations` arrays are valid (no entities found).
- `full_context` must be non-empty. If the LLM cannot produce one, fall back to `name` (entity) or `type` (relation).
- `embed_context` should be 1–2 sentences capturing the key identity/role. Keep it under ~200 characters for best embedding quality.

---

### `POST /generate`

Generate a natural-language answer given retrieved context nodes and a query.

**Request**:
```json
{
  "query": "Who leads the engineering team at Acme Corp?",
  "context": [
    {
      "id":           2,
      "name":         "Bob",
      "node_type":    "Person",
      "props":        { "role": "CTO" },
      "full_context": "Bob là CTO của Acme Corp, phụ trách kỹ thuật và sản phẩm",
      "score":        0.92
    },
    {
      "id":           3,
      "name":         "Acme Corp",
      "node_type":    "Company",
      "props":        {},
      "full_context": "Acme Corp là công ty công nghệ tại Silicon Valley",
      "score":        0.81
    }
  ],
  "relations": [
    { "from_node": "Bob", "to_node": "Acme Corp", "weight": 1.0, "edge_type": "works_at" }
  ]
}
```

| Field       | Type            | Required | Description                                                     |
|-------------|-----------------|----------|-----------------------------------------------------------------|
| `query`     | `string`        | yes      | The original user question.                                     |
| `context`   | `array[object]` | yes      | Top-ranked nodes from the graph, sorted by `score` desc.        |
| `relations` | `array[object]` | no       | Edges between context nodes. Empty array if no edges exist.     |
| `hints`     | `object`        | no       | Per-request LLM customization — see [LlmHints](#llmhints).     |

**Context node object**:

| Field          | Type      | Required | Description                                       |
|----------------|-----------|----------|---------------------------------------------------|
| `id`           | `integer` | yes      | Internal node ID assigned by the core.            |
| `name`         | `string`  | yes      | Entity name.                                      |
| `node_type`    | `string`  | yes      | Entity type label e.g. `Person`, `Company`.       |
| `props`        | `object`  | yes      | Entity properties. Use `{}` if none.              |
| `full_context` | `string`  | yes      | Rich semantic description. Use in LLM prompt.     |
| `score`        | `float`   | yes      | Relevance score computed by the core.             |

**Relation object**:

| Field       | Type     | Required | Description                                                              |
|-------------|----------|----------|--------------------------------------------------------------------------|
| `from_node` | `string` | yes      | Name of the source node.                                                 |
| `to_node`   | `string` | yes      | Name of the target node.                                                 |
| `weight`    | `float`  | yes      | Edge weight in `[0.0, 1.0]`.                                             |
| `edge_type` | `string` | no       | Relation label (e.g. `works_at`, `leads`). Omitted if empty or unknown.  |

**Response `200 OK`**:
```json
{
  "answer": "Bob is the CTO of Acme Corp and leads the engineering team."
}
```

| Field    | Type     | Description             |
|----------|----------|-------------------------|
| `answer` | `string` | Non-empty answer string. |

---

---

### `POST /reason`

Multi-hop reasoning step. The core calls this iteratively when using `/query/multihop`.

**Request**:
```json
{
  "query":          "What technologies does the AI team at Acme Corp use?",
  "context":        [ /* same shape as /generate context */ ],
  "relations":      [ /* same shape as /generate relations */ ],
  "iteration":      0,
  "max_iterations": 2,
  "hints":          { "rules": ["Only request follow_ups if absolutely necessary"] }
}
```

| Field            | Type            | Required | Description                                            |
|------------------|-----------------|----------|--------------------------------------------------------|
| `query`          | `string`        | yes      | Original user question.                                |
| `context`        | `array[object]` | yes      | Current accumulated context nodes.                     |
| `relations`      | `array[object]` | no       | Edges between context nodes.                           |
| `iteration`      | `integer`       | yes      | Current hop index (0-based).                           |
| `max_iterations` | `integer`       | yes      | Total hops allowed. Use to decide when to force answer. |
| `hints`          | `object`        | no       | Per-request LLM customization — see [LlmHints](#llmhints). |

**Response `200 OK`** — Form 1 (final answer):
```json
{ "answer": "The AI team uses PyTorch and CUDA.", "follow_ups": [], "done": true }
```

**Response `200 OK`** — Form 2 (needs more context):
```json
{ "answer": "Missing info about tech stack.", "follow_ups": ["PyTorch", "TensorFlow"], "done": false }
```

| Field        | Type            | Description                                                                    |
|--------------|-----------------|--------------------------------------------------------------------------------|
| `answer`     | `string`        | Final answer (when `done=true`) or brief explanation of what's missing.        |
| `follow_ups` | `array[string]` | Entity names the core should look up next. Max 5. Empty when `done=true`.      |
| `done`       | `boolean`       | `true` = stop iterating and use `answer`. `false` = expand and call again.     |

**Rules the plugin must follow**:
- When `iteration >= max_iterations - 1`, always set `done=true`.
- `follow_ups` must be exact entity names (as they appear or would appear in the graph).
- Only request follow_ups when the missing information would materially change the answer.

---

## LlmHints

Optional object sent in `/extract`, `/generate`, and `/reason` requests. Plugins should apply these to customise the LLM call for that request.

```json
{
  "system_prompt":   "You are a concise assistant. Answer in one sentence.",
  "rules":           ["Answer in Vietnamese", "Never mention entity IDs"],
  "extend_context":  ["The company was founded in 2015", "HQ is in Hanoi"]
}
```

| Field            | Type            | Applied by       | Effect                                                                     |
|------------------|-----------------|------------------|----------------------------------------------------------------------------|
| `system_prompt`  | `string \| null` | extract / generate / reason | Replaces the operation's default system prompt entirely.  |
| `rules`          | `array[string]` | extract / generate / reason | Appended as extra rules after the default rules block.    |
| `extend_context` | `array[string]` | generate / reason           | Injected as additional context snippets before the question. |

All fields are optional with safe defaults (null / [] / []). Plugins must handle a missing `hints` field gracefully.

---

## Error Handling

All error responses use standard HTTP status codes with a JSON body:

```json
{ "error": "<human-readable message>" }
```

| Status | Meaning                                              |
|--------|------------------------------------------------------|
| `400`  | Invalid request (missing fields, wrong types, etc.)  |
| `500`  | Internal plugin error (model crash, OOM, etc.)       |
| `503`  | Plugin not ready (still loading model)               |

The core treats any non-`2xx` response as a failed call and surfaces it as `502 bad_gateway` to the API caller. A `503` from the health endpoint specifically triggers `503 plugin_unavailable` instead.

---

## Plugin Descriptor (optional but recommended)

### `GET /info`

Returns metadata about the plugin. Not required for operation but useful for debugging and the plugin registry.

**Response `200 OK`**:
```json
{
  "name":         "minilm-embed",
  "version":      "1.2.0",
  "capabilities": ["embed/text"],
  "dim":          384,
  "language":     "python",
  "model":        "all-MiniLM-L6-v2"
}
```

| Field          | Type            | Description                                         |
|----------------|-----------------|-----------------------------------------------------|
| `name`         | `string`        | Unique plugin identifier.                           |
| `version`      | `string`        | SemVer string.                                      |
| `capabilities` | `array[string]` | Subset of `["embed", "extract", "generate", "reason"]`. |
| `dim`          | `integer`       | Embedding dimension. Present only if `embed` in capabilities. |
| `language`     | `string`        | Implementation language (informational).            |
| `model`        | `string`        | Underlying model name (informational).              |

---

## Versioning

The interface is versioned via the `X-Plugin-API-Version` request header:

```
X-Plugin-API-Version: 1
```

The current version is **1**. The core always sends this header. Plugins should:
- Accept requests without this header (treat as version 1).
- Return `400` with `{ "error": "unsupported API version: 2" }` if the version is not supported.

Breaking changes increment the major version. Non-breaking additions do not.

---

## Reference Implementation (Python)

The reference plugin at `plugins/server.py` implements all endpoints and serves over HTTP:

```python
import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "name": "minilm-embed",
        "version": "1.0.0",
        "capabilities": ["embed/text", "extract", "generate"],
        "dim": 384,
        "language": "python",
        "model": "all-MiniLM-L6-v2",
    }

@app.post("/embed/text")
def embed_text(body: dict):
    texts = body["texts"]
    vecs = model.encode(texts, normalize_embeddings=True)
    return {"vectors": vecs.tolist(), "dim": vecs.shape[1]}

@app.post("/extract")
def extract(body: dict): ...  # returns entities + relations with full_context

@app.post("/generate")
def generate(body: dict): ...  # uses context[].full_context in LLM prompt

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

> **Unix socket transport** (lower latency, same machine) is planned for Phase 9. Currently only HTTP transport is supported.

---

## Plugin Deployment Modes

There are three ways to use plugins with the engine. Configure which server handles each operation in `plugins.toml`.

### Mode 1 — Built-in reference plugin (simplest)

Use the Python reference server at `plugins/server.py` which handles all three operations (embed, extract, generate) in a single process.

```bash
cd plugins
uv sync
GEMINI_API_KEY=your-key uv run uvicorn server:app --host 0.0.0.0 --port 8001
```

`plugins.toml` — all three operations point to the same server:
```toml
[plugins.embed_text]
transport = "http"
url       = "http://localhost:8001"

[plugins.extract]
transport = "http"
url       = "http://localhost:8001"

[plugins.generate]
transport = "http"
url       = "http://localhost:8001"
```

### Mode 2 — Custom plugin in the same `plugins/` folder

Develop your plugin alongside the reference server, then configure `plugins.toml` to route specific operations to it.

Example: replace the embedding model with a custom multilingual one, keep Gemini for extraction and generation.

```
plugins/
  server.py            ← reference (handles extract + generate)
  my_embed_server.py   ← your custom embedding server
  pyproject.toml
```

Run both servers:
```bash
# Terminal 1: your custom embed server
cd plugins && uv run uvicorn my_embed_server:app --port 8002

# Terminal 2: reference server (extract + generate only)
cd plugins && GEMINI_API_KEY=your-key uv run uvicorn server:app --port 8001
```

`plugins.toml` — route each operation independently:
```toml
[plugins.embed_text]
transport = "http"
url       = "http://localhost:8002"   # your custom server

[plugins.extract]
transport = "http"
url       = "http://localhost:8001"   # reference server

[plugins.generate]
transport = "http"
url       = "http://localhost:8001"   # reference server
```

Your `my_embed_server.py` only needs to implement `/health` and `POST /embed/text`:
```python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("BAAI/bge-m3")   # multilingual

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(body: dict):
    vecs = model.encode(body["texts"], normalize_embeddings=True)
    return {"vectors": vecs.tolist(), "dim": vecs.shape[1]}
```

### Mode 3 — External plugin over HTTP

Run your plugin as a completely separate service — different machine, different language, different deployment lifecycle. The engine just calls it over HTTP.

Example: a Go embedding service running on another host.

`plugins.toml`:
```toml
[plugins.embed_text]
transport = "http"
url       = "http://embed-service.internal:9000"

[plugins.extract]
transport = "http"
url       = "http://llm-service.internal:9001"

[plugins.generate]
transport = "http"
url       = "http://llm-service.internal:9001"
```

Any language works as long as the server implements the endpoints in this spec. The engine does not care about the plugin's internal implementation.

**Plugin timeout**: configure `PLUGIN_TIMEOUT_SECS` (default: 60s) if your external plugin is slower.

---

## Minimal Plugin Checklist

A plugin is valid if it satisfies all of the following:

- [ ] `GET /health` returns `{ "status": "ok" }` when ready (within 60 seconds of startup)
- [ ] `POST /embed/text` returns vectors with consistent `dim` across all calls
- [ ] All vectors are unit-normalised (L2 norm ≈ 1.0)
- [ ] `POST /extract` returns `entities` and `relations` arrays (can be empty)
- [ ] Each entity has a non-empty `full_context` string
- [ ] Each entity has an `embed_context` string (recommended: 1–2 sentences, identity + role)
- [ ] Each relation has a non-empty `full_context` string
- [ ] Each relation has an `embed_context` string (recommended: 1–2 sentences describing the relationship)
- [ ] `POST /generate` receives `full_context` per context node and uses it in the LLM prompt
- [ ] `POST /generate` returns non-empty `answer` string
- [ ] Non-`2xx` responses include `{ "error": "..." }` body
