# Changelog

## v0.3.0 (2026-06-21)

### New feature: Image nodes (first-class graph citizens)

Image nodes live in the same HNSW vector space as text nodes — no separate index needed.

**Architecture: Vision LLM caption → text embed**
- Image URL (or base64 data-URI) sent to plugin `/embed/image`
- Plugin calls configurable Vision LLM (`IMAGE_CAPTION_MODEL` env var, default: `google/gemini-2.5-flash-lite`) to generate a dense text caption
- Caption embedded via the same sentence-transformers model → same 384-dim vector space as text nodes
- Cross-modal similarity search works without any HNSW / VectorStore changes

### New feature: Image plugin (`plugins/image/`)

Separate Python process (default port 8002) with two responsibilities:

**Local image storage** (`POST /store` + `GET /images/{filename}`)
- Accepts a base64 data-URI (`data` field) or external URL (`url` field)
- Saves images content-addressed by `SHA-256(raw_bytes)` → `data/images/<hash>.<ext>`
- Returns stable URL: `http://localhost:8002/images/<hash>.<ext>`
- 50 MB hard limit; MIME-to-extension mapping for JPEG, PNG, GIF, WebP, BMP, TIFF, SVG
- Path-traversal protection on `GET /images/{filename}`

**Image embedding** (`POST /embed`)
- Same Vision LLM caption pipeline as the text plugin's `/embed/image`
- Own lazy-loaded sentence-transformers instance (independent from text plugin process)
- Must use the same `EMBED_MODEL` as the text plugin for cross-modal search to work correctly

### New feature: auto-store (`[image] auto_store`)

When `auto_store = true`, `/ingest/json` transparently persists every image node before embedding:
1. Call `POST /store` on the image plugin → get stable content-addressed URL
2. Replace `image_url` on the node with the stable URL (stored in `nodes.json`)
3. Call `POST /embed/image` with the stable URL

Falls back to the original URL/data-URI if the store call fails (warning logged, ingest continues).
Default: `auto_store = false`.

### New fields
- `NodeInfo.image_url: Option<String>` — URL or base64 data-URI. When set, `/ingest/json` routes the node through `/embed/image` instead of `/embed/text`. Persisted in `nodes.json` (backward-compatible: old snapshots without this field load cleanly, value defaults to `None`).
- `ExportNode.image_url: Option<String>` — persisted in export/import JSON and NDJSON. Omitted from output when `null` (`skip_serializing_if`).

### New APIs (engine)
- `POST /query/image` — query the graph from an image (URL or base64 data-URI). Plugin captions the image and runs the standard HNSW → BFS → score → LLM pipeline. Response format identical to `/query/text`.
- `PluginClient::embed_image(url: &str) -> Result<Vec<f32>>` — calls `/embed/image` on the configured embed_image endpoint.
- `PluginClient::store_image(data: &str) -> Result<String>` — calls `POST /store` on the image_store endpoint; returns the stable URL. Routes `data:` prefixes to the `data` field, `http(s)://` to the `url` field.

### New config
- `[plugins.embed_image]` in `plugins.toml` — image embed endpoint. Default: `http://localhost:8002`.
- `[plugins.image_store]` in `plugins.toml` — image store endpoint. Default: `http://localhost:8002`.
- `[image] auto_store = false` — set to `true` to persist images at ingest time.
- `PLUGIN_EMBED_IMAGE_URL` env var — override for embed_image endpoint.
- `PLUGIN_IMAGE_STORE_URL` env var — override for image_store endpoint.
- `IMAGE_AUTO_STORE` env var — runtime override for auto_store (`true`/`false`).
- `IMAGE_CAPTION_MODEL` env var in Python plugins — vision model (format: `<provider>/<name>`; supports `google/…`, `openai/…`, `anthropic/…`).
- `IMAGE_LOCAL_DIR` env var in image plugin — storage directory (default: `./data/images`).
- `IMAGE_SERVE_BASE_URL` env var in image plugin — base URL for returned image URLs (default: `http://localhost:8002`).

### New plugin files (Python)
- `plugins/text/embed_image.py` — `POST /embed/image` on the text plugin. Uses the text plugin's existing embedding model; sufficient for deployments that don't need local storage.
- `plugins/image/` — new `LinkingMem-image-plugin` v0.1.0 package:
  - `store.py` — content-addressed local storage + static file serving
  - `embed.py` — vision LLM caption + sentence-transformers embed (own model instance)
  - `main.py` — FastAPI app, lifespan warmup, `/store`, `/images/{file}`, `/embed`, `/health`, `/info`
  - `schemas.py`, `auth.py`, `pyproject.toml`

### Tests
- `core/tests/test_image_nodes.rs` — 16 tests: NodeInfo serde roundtrip (with/without `image_url`), backward compat (old JSON without field → `None`), builder JSON parsing, base64 data-URI preservation, `ingest_json` text/image split logic, CSR build + BFS with image nodes, delta WAL roundtrip, merge preservation, save/load roundtrip, ExportNode serde (with + omit-null).

---

## v0.2.0 (2026-06-21)

### Breaking changes (internal format)
- `nodes.json` now includes `external_id` field on each node. Old snapshots load cleanly (`external_id` defaults to `""`) but nodes without `external_id` are not indexed in `external_id_index`.
- `edge_ids.json` added as a parallel file alongside `edge_types.json`. Ignored by older engine versions.

### Bug fixes
- **Distributed ingest data corruption (critical)** — In `merge_into()`, delta edges kept their original alloc'd IDs (e.g. `16_777_216` for INSTANCE_ID=1) as `from`/`to` after node IDs were reassigned to sequential CSR indices. This caused `CsrGraph::build()` to index out-of-bounds, silently corrupting the graph after every merge on non-zero instances. Fixed by building a `delta_id → new_csr_index` remap and applying it to all delta edges before CSR construction.

### New fields
- `NodeInfo.external_id: String` — stable public identifier set at ingest time. User-provided string ID from the payload, or auto-generated as `"{node_type}:{name}"`. Never reassigned at merge. Persisted in `nodes.json`.
- `EdgeInfo.edge_id: u64` — monotonic edge ID allocated by `DeltaStore.alloc_edge_id()` at ingest time. Unique within a process lifetime. Persisted in `edge_ids.json` (0 for edges from pre-v0.2.0 snapshots or base graph edges after merge).

### New APIs (engine-internal)
- `CsrGraph::get_by_external_id(id: &str) -> Option<u32>` — O(1) lookup from stable external ID to current CSR index.
- `DeltaStore::alloc_edge_id() -> u64` — monotonic edge ID allocator.

---

## v0.1.0

Initial release — see README for full feature list.
