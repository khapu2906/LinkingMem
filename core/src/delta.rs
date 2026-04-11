/// Delta store — accepts new nodes/edges without rebuilding CSR/HNSW.
///
/// Design:
///   - New nodes/edges go into a small adjacency list (DeltaGraph)
///   - Queries search BOTH the main CSR index AND the delta
///   - When delta grows past threshold → trigger async merge into CSR
///
/// This mirrors the LSM-tree pattern: fast writes to a mutable buffer,
/// periodic compaction into an immutable read-optimised structure.
///
/// WAL (Write-Ahead Log):
///   Every add_node / add_edge is first appended to data_dir/delta.wal
///   before touching in-memory state. On crash before merge, the WAL is
///   replayed at startup to restore the delta buffer. After a successful
///   merge the WAL is truncated.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicU32, Ordering};
use anyhow::Result;
use fs2::FileExt;

use crate::graph::csr::{CsrGraph, EdgeInfo, NodeInfo};
use crate::vector::hnsw::normalise;

// ── WAL checksum (FNV-1a 32-bit, no external dependency) ─────────────────────

/// FNV-1a 32-bit hash — fast, good distribution for short strings.
/// Used to detect corrupt or partially-written WAL lines.
fn fnv1a32(data: &[u8]) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

// ── delta graph (mutable, adjacency list) ────────────────────────────────────

#[derive(Default)]
pub struct DeltaGraph {
    /// node_id → adjacency list (to, weight)
    pub adj: HashMap<u32, Vec<(u32, f32)>>,
    /// new nodes not yet in the main CSR
    pub new_nodes: Vec<NodeInfo>,
    /// embeddings for new nodes (parallel to new_nodes, already normalised)
    pub new_vecs: Vec<Vec<f32>>,
    /// new edges between existing CSR nodes
    pub new_edges: Vec<EdgeInfo>,
    /// embeddings for new edges (parallel to new_edges, already normalised).
    /// An empty vec means the edge has no full_context embedding.
    pub new_edge_vecs: Vec<Vec<f32>>,
}

impl DeltaGraph {
    pub fn is_empty(&self) -> bool {
        self.new_nodes.is_empty() && self.new_edges.is_empty()
    }

    pub fn size(&self) -> usize {
        self.new_nodes.len() + self.new_edges.len()
    }

    pub fn nodes_len(&self) -> usize { self.new_nodes.len() }
    pub fn edges_len(&self) -> usize { self.new_edges.len() }

    /// Look up a node by ID in the delta buffer.
    pub fn get_node(&self, id: u32) -> Option<&NodeInfo> {
        self.new_nodes.iter().find(|n| n.id == id)
    }

    /// Neighbors of a node in the delta (returns empty slice if unknown)
    pub fn neighbors(&self, node_id: u32) -> &[(u32, f32)] {
        self.adj.get(&node_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn add_node(&mut self, node: NodeInfo, vec: Vec<f32>) {
        let id = node.id;
        self.new_nodes.push(node);
        self.new_vecs.push(vec);
        self.adj.entry(id).or_default();
    }

    pub fn add_edge(&mut self, edge: EdgeInfo, vec: Vec<f32>) {
        self.adj
            .entry(edge.from)
            .or_default()
            .push((edge.to, edge.weight));
        self.new_edges.push(edge);
        self.new_edge_vecs.push(vec);
    }
}

// ── delta store (thread-safe wrapper + WAL) ───────────────────────────────────

pub struct DeltaStore {
    inner: RwLock<DeltaGraph>,
    /// auto-merge when delta exceeds this many entries
    merge_threshold: usize,
    data_dir: std::path::PathBuf,
    /// WAL file handle — opened lazily, protected by Mutex.
    /// Multiple processes on the same filesystem share the same WAL file;
    /// each write acquires an exclusive OS-level file lock so entries from
    /// different instances never interleave.
    wal: Mutex<Option<File>>,
    /// Instance identifier (0–255).
    /// Loaded from INSTANCE_ID env var at construction time; defaults to 0.
    ///
    /// Node IDs allocated by this instance use the upper 8 bits for the
    /// instance ID and the lower 24 bits for a local monotonic counter:
    ///   node_id = (instance_id << 24) | local_counter
    /// This allows 256 instances × 16 M nodes each without ID collisions.
    instance_id: u32,
    /// Monotonically increasing 24-bit local counter.
    /// Initialised at boot from: max(graph_nodes, instance_id<<24) + replayed.
    next_node_id: AtomicU32,
}

impl DeltaStore {
    pub fn new(data_dir: std::path::PathBuf, merge_threshold: usize) -> Self {
        let instance_id: u32 = std::env::var("INSTANCE_ID")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0) as u32;

        if instance_id > 0 {
            tracing::info!("distributed mode: instance_id={instance_id}");
        }

        Self {
            inner: RwLock::new(DeltaGraph::default()),
            merge_threshold,
            data_dir,
            wal: Mutex::new(None),
            instance_id,
            next_node_id: AtomicU32::new(instance_id << 24),
        }
    }

    /// Set the starting point for node ID allocation.
    /// Must be called once at boot after loading the base graph and replaying WAL.
    /// `base` = graph.num_nodes() + replayed delta nodes.
    pub fn init_node_ids(&self, base: u32) {
        // For instance 0: start right after the base graph + replayed delta nodes.
        // For other instances: start from the instance's partition but never
        // below the base (handles small graphs where base > instance_offset).
        let instance_floor = self.instance_id << 24;
        let start = std::cmp::max(base, instance_floor);
        self.next_node_id.store(start, Ordering::SeqCst);
    }

    /// Allocate a node ID within this instance's partition.
    ///
    /// IDs are formatted as `(instance_id << 24) | local_seq`, giving each
    /// of up to 256 instances a non-overlapping range of 16 M nodes.
    pub fn alloc_node_id(&self) -> Result<u32> {
        let id    = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let limit = (self.instance_id << 24) | 0x00FF_FFFF;
        if id >= limit {
            self.next_node_id.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!(
                "node ID space exhausted for instance {} (limit: {limit})",
                self.instance_id
            );
        }
        Ok(id)
    }

    fn wal_path(&self) -> std::path::PathBuf {
        self.data_dir.join("delta.wal")
    }

    /// Append a JSON line to the WAL. Opens the file lazily on first call.
    /// Never panics — logs error and continues if WAL is unavailable.
    fn append_wal(&self, entry: &serde_json::Value) {
        let mut guard = match self.wal.lock() {
            Ok(g)  => g,
            Err(e) => e.into_inner(), // recover from poison — non-fatal
        };

        // Open lazily; on failure log and skip (degrade gracefully, don't crash)
        if guard.is_none() {
            match OpenOptions::new().create(true).append(true).open(self.wal_path()) {
                Ok(f)  => *guard = Some(f),
                Err(e) => {
                    tracing::error!("failed to open delta.wal: {e} — entry will NOT be recoverable on crash");
                    return;
                }
            }
        }

        let file = guard.as_mut().expect("Some — just set above");

        let json = match serde_json::to_string(entry) {
            Ok(j)  => j,
            Err(e) => { tracing::error!("WAL serialise failed: {e}"); return; }
        };

        // Format: "<8-hex-checksum>|<json>\n"
        let csum = fnv1a32(json.as_bytes());
        let line = format!("{:08x}|{}\n", csum, json);

        // Acquire an exclusive OS-level file lock before writing.
        // Multiple instances sharing the same WAL file (distributed ingest)
        // use this to ensure writes are never interleaved.
        if let Err(e) = file.lock_exclusive() {
            tracing::error!("WAL lock failed: {e} — entry may be lost on crash");
            return;
        }

        let write_ok =
            file.write_all(line.as_bytes()).is_ok() &&
            file.flush().is_ok() &&
            file.sync_data().is_ok();

        let _ = file.unlock(); // always unlock, even on write failure

        if !write_ok {
            tracing::error!("WAL write failed — entry may be lost on crash");
        }
    }

    /// Replay delta.wal on startup to restore in-memory state after a crash.
    ///
    /// Returns the number of entries replayed.
    /// Silently skips malformed or truncated lines (partial writes at crash time).
    pub fn replay_wal(&self) -> usize {
        let path = self.wal_path();
        if !path.exists() {
            return 0;
        }

        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("could not open delta.wal for replay: {e}");
                return 0;
            }
        };

        let mut count = 0;
        let mut delta = match self.inner.write() {
            Ok(g)  => g,
            Err(e) => e.into_inner(),
        };

        for line in std::io::BufReader::new(file).lines() {
            let Ok(line) = line else { continue };
            let line = line.trim();
            if line.is_empty() { continue; }

            // Support both formats:
            //   new: "<8-hex-csum>|<json>"  — verify checksum before parsing
            //   old: "<json>"               — no checksum (pre-Phase 3 WAL files)
            let json_str = if let Some((csum_str, json_part)) = line.split_once('|') {
                if csum_str.len() == 8 && csum_str.chars().all(|c| c.is_ascii_hexdigit()) {
                    match u32::from_str_radix(csum_str, 16) {
                        Ok(expected) => {
                            let actual = fnv1a32(json_part.as_bytes());
                            if actual != expected {
                                tracing::warn!(
                                    "delta.wal: checksum mismatch (expected {:08x}, got {:08x}) — skipping corrupt entry",
                                    expected, actual
                                );
                                continue;
                            }
                            json_part
                        }
                        Err(_) => line, // malformed hex — fall through to JSON parse
                    }
                } else {
                    line // not a checksum prefix — treat whole line as JSON
                }
            } else {
                line // no '|' — legacy format
            };

            let Ok(entry) = serde_json::from_str::<serde_json::Value>(json_str) else {
                tracing::warn!("delta.wal: skipping malformed line");
                continue;
            };

            match entry["op"].as_str() {
                Some("add_node") => {
                    let node = match serde_json::from_value::<NodeInfo>(entry["node"].clone()) {
                        Ok(n) => n,
                        Err(_) => continue,
                    };
                    let vec: Vec<f32> = match entry["vec"].as_array() {
                        Some(arr) => arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect(),
                        None => continue,
                    };
                    // vec is already normalised — write directly to DeltaGraph
                    delta.add_node(node, vec);
                    count += 1;
                }
                Some("add_edge") => {
                    let edge = match serde_json::from_value::<EdgeInfo>(entry["edge"].clone()) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                    let vec: Vec<f32> = entry["vec"].as_array()
                        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                        .unwrap_or_default();
                    delta.add_edge(edge, vec);
                    count += 1;
                }
                _ => {}
            }
        }

        if count > 0 {
            tracing::info!("replayed {count} entries from delta.wal");
        }
        count
    }

    /// Truncate the WAL after a successful merge — called at end of merge_into().
    fn truncate_wal(&self) {
        // Close existing handle first so we can truncate safely
        let mut guard = match self.wal.lock() { Ok(g) => g, Err(e) => e.into_inner() };
        *guard = None;
        drop(guard);
        if let Err(e) = std::fs::write(self.wal_path(), b"") {
            tracing::warn!("failed to truncate delta.wal: {e}");
        }
    }

    pub fn add_node(&self, node: NodeInfo, mut vec: Vec<f32>) {
        let _ = normalise(&mut vec);
        self.append_wal(&serde_json::json!({
            "op": "add_node", "node": &node, "vec": &vec, "instance_id": self.instance_id,
        }));
        match self.inner.write() { Ok(mut g) => g.add_node(node, vec), Err(e) => e.into_inner().add_node(node, vec) };
    }

    pub fn add_edge(&self, edge: EdgeInfo, mut vec: Vec<f32>) {
        let _ = normalise(&mut vec);
        self.append_wal(&serde_json::json!({
            "op": "add_edge", "edge": &edge, "vec": &vec, "instance_id": self.instance_id,
        }));
        match self.inner.write() { Ok(mut g) => g.add_edge(edge, vec), Err(e) => e.into_inner().add_edge(edge, vec) };
    }

    /// Commit a batch of nodes in one WAL fsync + one RwLock acquisition.
    /// Dramatically faster than calling add_node() in a loop for large imports.
    pub fn add_nodes_batch(&self, mut nodes: Vec<(NodeInfo, Vec<f32>)>) {
        for (_, vec) in &mut nodes { let _ = normalise(vec); }

        let entries: Vec<serde_json::Value> = nodes.iter()
            .map(|(node, vec)| serde_json::json!({
                "op": "add_node", "node": node, "vec": vec, "instance_id": self.instance_id,
            }))
            .collect();
        self.append_wal_batch(&entries);

        let mut g = match self.inner.write() { Ok(g) => g, Err(e) => e.into_inner() };
        for (node, vec) in nodes { g.add_node(node, vec); }
    }

    /// Commit a batch of edges in one WAL fsync + one RwLock acquisition.
    pub fn add_edges_batch(&self, mut edges: Vec<(EdgeInfo, Vec<f32>)>) {
        for (_, vec) in &mut edges { let _ = normalise(vec); }

        let entries: Vec<serde_json::Value> = edges.iter()
            .map(|(edge, vec)| serde_json::json!({
                "op": "add_edge", "edge": edge, "vec": vec, "instance_id": self.instance_id,
            }))
            .collect();
        self.append_wal_batch(&entries);

        let mut g = match self.inner.write() { Ok(g) => g, Err(e) => e.into_inner() };
        for (edge, vec) in edges { g.add_edge(edge, vec); }
    }

    /// Write multiple WAL entries with a single fsync.
    fn append_wal_batch(&self, entries: &[serde_json::Value]) {
        if entries.is_empty() { return; }

        let mut guard = match self.wal.lock() { Ok(g) => g, Err(e) => e.into_inner() };
        if guard.is_none() {
            match OpenOptions::new().create(true).append(true).open(self.wal_path()) {
                Ok(f)  => *guard = Some(f),
                Err(e) => { tracing::error!("failed to open delta.wal: {e}"); return; }
            }
        }
        let file = guard.as_mut().expect("Some");

        // Serialise all entries into a single buffer — one write call.
        let mut buf = String::new();
        for entry in entries {
            match serde_json::to_string(entry) {
                Ok(json) => {
                    let csum = fnv1a32(json.as_bytes());
                    buf.push_str(&format!("{:08x}|{}\n", csum, json));
                }
                Err(e) => { tracing::error!("WAL serialise failed: {e}"); }
            }
        }

        if let Err(e) = file.lock_exclusive() {
            tracing::error!("WAL lock failed: {e}"); return;
        }
        let ok = file.write_all(buf.as_bytes()).is_ok()
            && file.flush().is_ok()
            && file.sync_data().is_ok();
        let _ = file.unlock();

        if !ok { tracing::error!("WAL batch write failed — entries may be lost on crash"); }
    }

    pub fn size(&self) -> usize {
        match self.inner.read() { Ok(g) => g.size(), Err(e) => e.into_inner().size() }
    }

    pub fn needs_merge(&self) -> bool {
        self.size() >= self.merge_threshold
    }

    /// Read-only access for query time — minimal lock duration
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, DeltaGraph> {
        match self.inner.read() { Ok(g) => g, Err(e) => e.into_inner() }
    }

    /// Merge delta into the existing CSR graph + vector stores.
    /// Returns the freshly built CsrGraph, node vectors (for node HNSW rebuild),
    /// edge vectors and endpoints (for edge HNSW rebuild).
    /// Truncates the WAL on success.
    pub async fn merge_into(
        &self,
        base_graph: Arc<CsrGraph>,
        storage: &dyn crate::storage::StorageBackend,
    ) -> Result<(CsrGraph, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<(u32, u32)>)> {
        use crate::vector::store::VectorStore;
        use crate::vector::hnsw::save_edge_hnsw_data;

        let delta = {
            let mut guard = match self.inner.write() { Ok(g) => g, Err(e) => e.into_inner() };
            std::mem::take(&mut *guard) // drain delta, release lock fast
        };

        tracing::info!(
            "merging delta: {} new nodes, {} new edges",
            delta.new_nodes.len(),
            delta.new_edges.len()
        );

        // ── node vectors ──────────────────────────────────────────────────────
        let store = VectorStore::open(&storage.local_path().join("vectors.bin"))?;
        let mut all_node_vecs: Vec<Vec<f32>> = (0..store.num_vecs as u32)
            .map(|id| store.get(id).to_vec())
            .collect();

        // merge nodes — assign new numeric ids starting after existing ones
        let base_count = base_graph.num_nodes() as u32;
        let mut merged_nodes: Vec<NodeInfo> = base_graph.nodes.clone();
        for (i, mut node) in delta.new_nodes.into_iter().enumerate() {
            node.id = base_count + i as u32;
            merged_nodes.push(node);
        }
        all_node_vecs.extend(delta.new_vecs);

        // ── edge vectors ──────────────────────────────────────────────────────
        // Load existing edge vectors (may not exist on first run)
        let (mut all_edge_vecs, mut all_edge_endpoints): (Vec<Vec<f32>>, Vec<(u32, u32)>) =
            if storage.exists("edge_vectors.bin") && storage.exists("edge_endpoints.json") {
                let ev_store = VectorStore::open(&storage.local_path().join("edge_vectors.bin"))?;
                let vecs = (0..ev_store.num_vecs as u32)
                    .map(|id| ev_store.get(id).to_vec())
                    .collect();
                let eps: Vec<(u32, u32)> = serde_json::from_str(&storage.read_string("edge_endpoints.json")?)?;
                (vecs, eps)
            } else {
                (vec![], vec![])
            };

        // merge edges — reconstruct from CSR (with full_context) + append delta
        let mut all_edges: Vec<EdgeInfo> = base_graph.all_edges();

        for (edge, vec) in delta.new_edges.iter().zip(delta.new_edge_vecs.iter()) {
            all_edge_endpoints.push((edge.from, edge.to));
            all_edge_vecs.push(vec.clone());
        }
        all_edges.extend(delta.new_edges);

        let new_graph = CsrGraph::build(merged_nodes, &all_edges);

        // ── persist ───────────────────────────────────────────────────────────
        let node_dim = all_node_vecs.first().map(|v| v.len()).unwrap_or(0);
        crate::graph::builder::save(&new_graph, storage)?;
        VectorStore::write(&storage.local_path().join("vectors.bin"), node_dim, &all_node_vecs)?;
        save_edge_hnsw_data(storage, &all_edge_vecs, &all_edge_endpoints)?;

        tracing::info!(
            "merge complete: {} nodes, {} edges, {} edge embeddings",
            new_graph.num_nodes(),
            new_graph.num_edges(),
            all_edge_vecs.len(),
        );

        self.truncate_wal();

        Ok((new_graph, all_node_vecs, all_edge_vecs, all_edge_endpoints))
    }
}
