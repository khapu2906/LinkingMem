# Algorithms — LinkingMem

## 1. CSR Graph

### 1.1 Building CSR from an Edge List

**Input**: list of nodes, list of edges (from, to, weight)

**Step 1** — count the out-degree of each node:
```
degree = [0] × num_nodes
for each edge (u, v, w):
    degree[u] += 1
```

**Step 2** — prefix sum → offsets:
```
offsets[0] = 0
offsets[i+1] = offsets[i] + degree[i]
```

Example with 4 nodes, 5 edges:
```
degree  = [2, 1, 2, 0]
offsets = [0, 2, 3, 5, 5]
```

**Step 3** — fill edges in order:
```
cursor = copy(offsets)   ← tracks current fill position
for each edge (u, v, w):
    pos = cursor[u]
    edges[pos] = v
    weights[pos] = w
    cursor[u] += 1
```

**Complexity**: O(V + E) time, O(V + E) space.

**Result**:
```
offsets = [0, 2, 3, 5, 5]
edges   = [1, 3, 2, 0, 4]
                ↑
         neighbors(node=1) = edges[offsets[1]..offsets[2]]
                           = edges[2..3] = [2]
```

### 1.2 BFS with Hop Tracking

**Input**: seed nodes, max_depth, max_nodes

```
visited = HashSet{}
queue   = VecDeque{ (seed, depth=0) for seed in seeds }
result  = []

while queue is not empty AND result.len() < max_nodes:
    (node, depth) = queue.pop_front()

    if depth >= max_depth: continue

    for neighbor in neighbors(node):
        if neighbor not in visited:
            visited.insert(neighbor)
            queue.push_back((neighbor, depth+1))
            result.push((neighbor, depth+1))
```

**Why BFS instead of DFS**: BFS guarantees accurate hop distances, which is critical for scoring (proximity = 1/hop). DFS may encounter the same node at a larger hop distance.

**Cache behavior**: CSR edges are contiguous in memory — when reading neighbors(node_i), the CPU prefetcher automatically loads neighbors(node_i+1) into cache. BFS over CSR has a significantly higher cache hit rate than an adjacency list.

---

## 2. HNSW

### 2.1 Structure

HNSW is a multi-layer graph. Every node appears at layer 0. The probability that a node appears at layer L is:

```
P(node appears at layer L) = exp(-L / mL)
where mL = 1 / ln(M)
```

Parameters:
- `M` = maximum connections per node (default 16)
- `ef_construction` = beam width during build (default 200)
- `ef_search` = beam width during query (default = k×2)

### 2.2 Insert Node

```
entry_point = node at the highest current layer
current_layer = random_layer()  ← drawn from exponential distribution

// Phase 1: descend from top layer down to current_layer+1
// keep only 1 best candidate (greedy)
for L = max_layer downto current_layer+1:
    candidates = greedy_search(entry_point, query, ef=1, layer=L)
    entry_point = best(candidates)

// Phase 2: descend from current_layer down to 0
// keep ef_construction candidates (beam search)
for L = current_layer downto 0:
    candidates = beam_search(entry_point, query, ef=ef_construction, layer=L)
    neighbors  = select_neighbors(candidates, M)
    connect new node to neighbors at layer L

    // reverse update: if a neighbor has too many connections, prune
    for each neighbor in neighbors:
        if degree(neighbor, L) > M_max:
            prune(neighbor, L, M_max)

    entry_point = best(candidates)
```

### 2.3 Search

```
entry_point = global entry point (node at highest layer)

// Phase 1: greedy descent from top layer down to layer 1
for L = max_layer downto 1:
    entry_point = greedy_search(entry_point, query, ef=1, layer=L)

// Phase 2: beam search at layer 0 with ef_search candidates
candidates = beam_search(entry_point, query, ef=ef_search, layer=0)
return top-k from candidates
```

### 2.4 Beam Search (greedy_search with ef > 1)

```
candidates = min-heap (by distance, capacity=ef)
visited    = HashSet{}

candidates.push((dist(entry, query), entry))
visited.insert(entry)

while candidates is not empty:
    current = candidates.pop_min()   ← closest unprocessed node

    if dist(current, query) > dist(worst_in_result, query):
        break   ← cannot improve further

    for neighbor in graph_neighbors(current, layer):
        if neighbor not in visited:
            visited.insert(neighbor)
            d = dist(neighbor, query)
            if d < dist(worst_in_result, query) OR len(result) < ef:
                candidates.push((d, neighbor))

return result
```

**Complexity**: O(log n) expected with fixed M.

### 2.5 Cosine Similarity

Vectors are normalized to unit length before storage:

```
normalize(v) = v / ||v||
```

After normalization, cosine similarity equals dot product:

```
cosine_sim(a, b) = Σ(a_i × b_i)
```

No sqrt required — faster. Distance = 1 - cosine_sim.

### 2.6 Dual HNSW — Node + Edge

The system maintains **two** separate HNSW indexes:

**NodeHnswIndex**: indexed on the text of each node.
Embedding priority: `embed_context` → `full_context` → `name`.

**EdgeHnswIndex**: indexed on the text of each edge.
Embedding priority: `embed_context` → `full_context` → `edge_type`.

`embed_context` is a short dense description (1–2 sentences), optimized for cosine similarity.
`full_context` is a verbose description used for the LLM — may be long, not optimal for HNSW.
If `embed_context` is absent: fall back to `full_context`, then `name`/`edge_type`.
- Stores `endpoints: Vec<(from_id, to_id)>` to map from edge index → node IDs
- Search returns `(from_node_id, to_node_id, distance)` — the endpoint nodes of matching edges
- Empty index (no edges yet) → no-op search, no error

At query time, both searches run **in parallel** via `tokio::join!`:

```
(node_hits, edge_hits) = join!(
    node_hnsw.search(query_vec, k),
    edge_hnsw.search(query_vec, k)
)

seed_ids = node_hits.map(|(id, _)| id)
         ∪ edge_hits.flatmap(|(from, to, _)| [from, to])
```

Result: BFS can reach parts of the graph that are relevant only through the semantics of relationships, not just through node semantics.

---

## 3. Scoring

### 3.1 Formula

```
score(node) = α × vector_sim
            + β × graph_proximity
            + γ × node_weight
```

**vector_sim**: cosine similarity between the node embedding and the query embedding.

```
vector_sim(node, query) = cosine_sim(embed(node), embed(query))
∈ [-1, 1], in practice ∈ [0, 1] with normalized positive embeddings
```

**graph_proximity**: rewards nodes that are close to the seed in graph terms.

```
graph_proximity(hop) = 1 / (hop + 1)
hop=0 → 1.0
hop=1 → 0.5
hop=2 → 0.33
hop=3 → 0.25
```

**node_weight**: importance of the node in the graph (degree-normalized).

```
node_weight = out_degree(node) / max_out_degree
∈ [0, 1]
```

### 3.2 Presets

| Mode | α | β | γ | Use when |
|---|---|---|---|---|
| `balanced` | 0.5 | 0.3 | 0.2 | General-purpose RAG |
| `semantic` | 0.7 | 0.2 | 0.1 | Semantic search |
| `relationship` | 0.3 | 0.6 | 0.1 | Relationship traversal |
| `entity` | 0.4 | 0.2 | 0.4 | Finding important nodes |

α + β + γ = 1.0 in all presets.

### 3.3 Why Not PageRank?

PageRank requires iterative computation over the entire graph — O(V×E) per rebuild. With a frequently changing graph (delta updates), the cost is prohibitive. `node_weight = degree / max_degree` is a sufficiently good approximation, computable in O(1) during CSR construction.

---

## 4. Delta Merge

### 4.1 LSM-Tree Pattern

```
Write path:  new data → DeltaStore (adjacency list, mutable)
Read path:   query searches BOTH CsrGraph AND DeltaStore
Compaction:  async merge when delta.size() >= threshold
```

### 4.2 Merge Algorithm

```
// 1. drain delta (hold write lock as briefly as possible)
delta = swap(DeltaStore, empty)

// 2. load existing node vectors from mmap
all_node_vecs = load_all(vectors.bin)

// 3. load existing edge vectors + endpoints from disk
(all_edge_vecs, all_endpoints) = load(edge_vectors.bin, edge_endpoints.json)

// 4. merge nodes — assign new numeric ids
offset = base_graph.num_nodes()
for (i, node) in delta.new_nodes.enumerate():
    node.id = offset + i
    merged_nodes.push(node)
all_node_vecs.extend(delta.new_node_vecs)

// 5. merge edges + edge vecs
all_edges = reconstruct_from_csr(base_graph)
all_edges.extend(delta.new_edges)
for (edge, vec) in delta.new_edges.zip(delta.new_edge_vecs):
    all_endpoints.push((edge.from, edge.to))
    all_edge_vecs.push(vec)

// 6. build new CSR
new_graph = CsrGraph::build(merged_nodes, all_edges)

// 7. persist
save(new_graph, data_dir)          // nodes.json, edges.bin, edge_contexts.json
write(vectors.bin, all_node_vecs)
write(edge_vectors.bin, all_edge_vecs)
write(edge_endpoints.json, all_endpoints)

// 8. rebuild HNSW indexes (offloaded to blocking thread pool via spawn_blocking)
new_node_hnsw = HnswIndex::build(all_node_vecs, ...)
new_edge_hnsw = EdgeHnswIndex::build(all_edge_vecs, all_endpoints, ...)

// 9. hot-swap 4 components (acquire write locks in sequence: g → h → eh → st)
*graph_rw.write()     = Arc::new(new_graph)
*hnsw_rw.write()      = Arc::new(new_node_hnsw)
*edge_hnsw_rw.write() = Arc::new(new_edge_hnsw)
*store_rw.write()     = Arc::new(empty_delta)
```

**Hot-swap**: uses `tokio::sync::RwLock<Arc<T>>`. Queries in flight hold a reference to the old data until they complete — they are never interrupted.

### 4.3 Query with Delta

During a query, the engine searches in parallel:

```
// Node HNSW search on main index + brute-force delta
seeds_node = node_hnsw.search(query_vec, k=20)
seeds_delta = brute_search(delta.new_node_vecs, query_vec, k=5)

// Edge HNSW search — concurrent with node search (tokio::join!)
edge_hits = edge_hnsw.search(query_vec, k=20)
edge_endpoints = edge_hits.map(|(from, to, _)| [from, to]).flatten()

// BFS expansion over BOTH graphs
seeds = merge(seeds_node, seeds_delta) ∪ edge_endpoints
subgraph = bfs_expand_both(csr_graph, delta_graph, seeds, depth=2)
```

---

## 5. Token Bucket Rate Limiting

### 5.1 Algorithm

Each API key has its own bucket:

```
State: { tokens: f64, last_refill: Instant }

consume(bucket):
    now = Instant::now()
    elapsed = now - bucket.last_refill

    // refill proportional to elapsed time
    bucket.tokens = min(
        bucket.capacity,
        bucket.tokens + elapsed × refill_rate
    )
    bucket.last_refill = now

    if bucket.tokens >= 1.0:
        bucket.tokens -= 1.0
        return ALLOWED
    else:
        return DENIED
```

**Why token bucket instead of fixed window?**

A fixed window allows a burst of 2× capacity at the boundary (end of the old window + start of the new one). Token bucket provides tighter control — bursts are capped by `capacity`.

### 5.2 Timing-Safe Key Comparison

```rust
// DO NOT do this — early exit on first differing byte → timing leak
if stored_key == provided_key { ... }

// DO this — always compares all bytes, no early exit
let equal = stored_key.len() == provided_key.len()
    && stored_key.bytes()
        .zip(provided_key.bytes())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0;
```

XOR each byte pair, OR all results. If all bytes are equal the final result is 0. No branch on key content — no timing leak.

---

## 6. Plugin Health Check (TTL Cache)

Before every request **that uses the plugin**, the engine calls `check_ready()`. For `/query`, the engine first determines whether the pipeline actually needs the plugin — if not, it is skipped entirely:

```
// For /query — check before calling check_ready()
needs_embed    = request.vector IS NULL
needs_generate = request.pipeline.llm_generate   // default true
if NOT (needs_embed OR needs_generate):
    skip   // pure graph query, plugin not needed → no 503

// check_ready() — called when the handler determines the plugin is needed
check_ready():
    // check cache
    if cache.is_some() AND cache.age < TTL(5s):
        return cache.value

    // stale or no cache → real probe
    ready = GET /health → status 200 AND body.status == "ok"
    // For unix socket: check sock.exists() instead of HTTP

    // update cache
    cache = (ready, now)
    return ready
```

The cache uses `std::sync::Mutex<Option<(bool, Instant)>>` — the critical section is synchronous, so a tokio Mutex is not required.

If `check_ready() == false` → immediately return `503 plugin_unavailable`, without calling embed/extract/generate.

---

## 7. Memory Layout

### RAM Estimate (100k nodes, 200k edges, dim=384)

```
CsrGraph (topology):
  offsets:  100k × 4 bytes =  0.4 MB
  edges:    600k × 4 bytes =  2.4 MB  (avg degree 6)
  weights:  600k × 4 bytes =  2.4 MB
  metadata: 100k × ~100 bytes = 10 MB
  Total: ~15 MB

NodeHnswIndex (M=16):
  ~600 MB

EdgeHnswIndex:
  ~1.2 GB worst case

LRU Cache (50k × 384-dim):
  50k × 384 × 4 bytes = ~75 MB

VectorStore (mmap — node + edge):
  Node: 100k × 384 × 4 bytes = ~153 MB
  Edge: 200k × 384 × 4 bytes = ~307 MB
  (OS-managed, does not occupy full RAM)

Actual RSS: ~1.5–2 GB with full node + edge HNSW
```

### Binary Format of vectors.bin / edge_vectors.bin

```
Offset 0:   dim      (u32, little-endian) = 384
Offset 4:   num_vecs (u32, little-endian) = 100000
Offset 8:   vec[0]   (f32 × 384 = 1536 bytes)
Offset 1544: vec[1]  (f32 × 384)
...
Offset 8 + i×1536: vec[i]
```

`VectorStore::get(id)` = pointer into `mmap[8 + id×dim×4]`, zero-copy, no allocation.
