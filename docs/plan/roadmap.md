# Roadmap — AI Graph Engine

## v0.1.0 — Current ✅

Core engine, fully functional for single-node deployments.

**Engine**
- CSR graph, HNSW vector index, mmap vector store
- Delta buffer + WAL (crash recovery, hot-swap merge)
- Multi-mode scoring: `balanced / semantic / relationship / entity`
- BFS graph expansion, entity resolution at ingest time
- Query result cache (moka, TTL-based), embed cache (LRU)
- Response profiles: `minimal / standard / full`
- API key auth + per-key token-bucket rate limiter
- Prometheus metrics (`/metrics`)
- Multi-hop reasoning (`/query/multihop`, iterative LLM)
- Distributed ingest (WAL file locking, instance-partitioned node IDs)
- Per-request LLM customization (`LlmHints`: system_prompt, rules, extend_context)
- Export (`GET|POST /export/graph`, JSON + NDJSON)
- Import (`POST /import/graph`, `POST /import/graph/upload`)

**Plugin (Python text)**
- `/embed/text` — SentenceTransformers (torch or ONNX backend)
- `/extract` — LLM entity extraction
- `/generate` — LLM answer generation
- `/reason` — multi-hop reasoning step
- Unix socket transport (`transport = "unix"` in plugins.toml)
- Model format: `provider/model-name` (e.g. `openai/gpt-4o-mini`)
- Providers: `google`, `openai`, `anthropic`

**Deployment**
- Two Docker targets: `engine` (Rust only) and `full` (Rust + Python, unix socket)
- docker-compose: HTTP mode (2 containers) and full mode (1 container)

---

## v0.2.0 — Observability & Developer Experience

Targets: easier debugging, better introspection, smoother onboarding.

- [ ] **Structured logging** — JSON log output (`RUST_LOG_FORMAT=json`) for log aggregators
- [ ] **Request tracing** — trace ID propagated from API through plugin calls, visible in response headers
- [ ] **Query explain** — `POST /query/explain` returns scoring breakdown per node without calling the LLM
- [ ] **Graph diff** — `GET /graph/diff?since=<timestamp>` shows what changed since a given time
- [ ] **Plugin `/info` standardized** — engine surfaces plugin metadata at `GET /plugin/info`
- [ ] **CLI tool** — `LinkingMem ingest`, `query`, `export` as a standalone binary
- [ ] **OpenAPI spec** — auto-generated from axum handlers

---

## v0.3.0 — Plugin Ecosystem

Targets: make it easy to write and distribute custom plugins.

- [ ] **Plugin SDK** — thin Python base class + auto-wiring of auth, health, LlmHints
- [ ] **Plugin registry** — `plugins.toml` supports multiple named plugins, engine routes by capability
- [ ] **Embed plugin separation** — embed, extract, generate can run as completely separate services (already partly supported, needs docs + examples)
- [ ] **Data hook plugin** — `POST /on_merge` notification after each merge (push to S3, notify downstream, etc.)
- [ ] **Ollama provider** — `ollama/llama3.2` model format, maps to OpenAI-compatible base URL

---

## v0.4.0 — Performance & Scale

Targets: handle larger graphs and higher query throughput.

- [ ] **Incremental HNSW** — insert new vectors into existing index without full rebuild
- [ ] **Parallel BFS** — multi-threaded graph expansion using rayon
- [ ] **Vector quantization** — int8 / binary quantization for HNSW to cut RAM usage
- [ ] **Streaming LLM responses** — `POST /query/stream` returns SSE, LLM tokens streamed as they arrive
- [ ] **Batch query** — `POST /query/batch` accepts multiple questions, runs embed in parallel

---

## v0.5.0 — Cluster (primary / replica)

Targets: horizontal read scaling, HA deployments.

See [cluster-development.md](cluster-development.md) for full design.

- [ ] **Cluster plugin interface** — engine delegates all coordination to `CLUSTER_PLUGIN_URL`
- [ ] **Role-based write gating** — replica returns 409 with primary address
- [ ] **Snapshot endpoint** — `GET /snapshot` streams `tar.gz` of `data/`
- [ ] **Admin sync endpoint** — `POST /admin/sync` triggers snapshot pull and hot-swap
- [ ] **Reference cluster plugin** — simple single-primary implementation using a shared file lock

---

## v1.0.0 — Production Ready

Targets: stable API, enterprise-grade reliability.

- [ ] **Stable API contract** — no breaking changes after v1.0.0 without major version bump
- [ ] **TLS termination** — native TLS support without requiring a reverse proxy
- [ ] **WAL streaming** — push WAL entries to replicas in real time (sub-second replica lag)
- [ ] **Horizontal sharding** — partition graph across instances by entity type or ID range
- [ ] **Backup API** — `POST /backup` triggers an atomic snapshot to a configured destination
- [ ] **Migration tool** — `LinkingMem migrate` for schema/format upgrades between versions

---

## Not Planned

- **Built-in UI** — use Grafana for metrics, standard REST clients for queries
- **Graph query language** (Cypher/Gremlin) — the scoring pipeline is the query interface
- **Multi-tenancy** — run separate instances per tenant instead
