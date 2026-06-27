use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Node metadata stored alongside the graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: u32,
    pub name: String,
    pub node_type: String,
    pub weight: f32, // combined in+out degree importance (0..1)
    pub props: serde_json::Value,
    /// Verbose description passed to the LLM during answer generation.
    /// e.g. "Alice is CEO of Acme Corp since 2018. She leads a team of 50 engineers..."
    #[serde(default)]
    pub full_context: String,
    /// Short dense text used for HNSW embedding (vector search).
    /// e.g. "Alice, CEO at Acme Corp, engineering leader"
    /// Falls back to `full_context` (if non-empty) then `name` when absent.
    #[serde(default)]
    pub embed_context: Option<String>,
    /// Stable public identifier set at ingest time, never reassigned at merge.
    /// User-provided string ID, or auto-generated as "{node_type}:{name}".
    /// Empty string for nodes loaded from pre-v0.2.0 snapshots.
    #[serde(default)]
    pub external_id: String,
    /// URL or base64 data-URI of the image associated with this node.
    /// When set, the engine calls /embed/image instead of /embed/text at ingest time.
    #[serde(default)]
    pub image_url: Option<String>,
}

/// Edge with optional weight/type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInfo {
    pub from: u32,
    pub to: u32,
    pub edge_type: String,
    pub weight: f32,
    /// Verbose description passed to the LLM during answer generation.
    #[serde(default)]
    pub full_context: String,
    /// Short dense text used for edge HNSW embedding.
    /// Falls back to `full_context` (if non-empty) then `edge_type` when absent.
    #[serde(default)]
    pub embed_context: Option<String>,
    /// Monotonic edge ID allocated at ingest time. 0 for edges from pre-v0.2.0 snapshots.
    #[serde(default)]
    pub edge_id: u64,
}

/// Compressed Sparse Row graph.
///
/// Memory layout (forward edges):
///   offsets: [0, 2, 5, 6, 8]   <- node i's neighbors start at offsets[i]
///   edges:   [1, 3, 0, 2, 4, 1, 0, 4]  <- all neighbor ids packed together
///
/// Reverse edges stored separately for bidirectional BFS (relationship mode).
/// node_weight uses both in-degree and out-degree for a better importance signal.
pub struct CsrGraph {
    /// offsets[i] = start index in `edges` for node i's out-neighbors
    offsets: Vec<u32>,
    /// packed out-neighbor ids
    edges: Vec<u32>,
    /// node metadata (separate from topology for cache efficiency)
    pub nodes: Vec<NodeInfo>,
    /// edge weights parallel to `edges`
    edge_weights: Vec<f32>,
    /// edge type labels parallel to `edges`
    edge_types: Vec<String>,
    /// verbose LLM context parallel to `edges`
    edge_full_contexts: Vec<String>,
    /// short dense embed text parallel to `edges` — used for edge HNSW
    edge_embed_contexts: Vec<Option<String>>,
    /// offsets for reverse (in-coming) edges — used by bidirectional BFS
    rev_offsets: Vec<u32>,
    /// packed in-neighbor ids (reverse direction)
    rev_edges: Vec<u32>,
    /// Maps external_id → CSR array index for stable node lookups.
    /// Built from NodeInfo.external_id during build(). Only non-empty external_ids are indexed.
    pub external_id_index: HashMap<String, u32>,
}

impl CsrGraph {
    /// Build CSR from a flat list of edges.
    /// Computes both forward and reverse edge indices.
    /// node_weight = (out_degree + in_degree) / max_combined — captures importance from both sides.
    pub fn build(nodes: Vec<NodeInfo>, edge_list: &[EdgeInfo]) -> Self {
        let n = nodes.len();

        // count out-degree and in-degree per node
        let mut out_degree = vec![0u32; n];
        let mut in_degree  = vec![0u32; n];
        for e in edge_list {
            out_degree[e.from as usize] += 1;
            in_degree[e.to as usize]   += 1;
        }

        // ── forward index ────────────────────────────────────────────────────
        let mut offsets = vec![0u32; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + out_degree[i];
        }

        let total_fwd = offsets[n] as usize;
        let mut edges                = vec![0u32;          total_fwd];
        let mut edge_weights         = vec![0.0f32;        total_fwd];
        let mut edge_types           = vec![String::new(); total_fwd];
        let mut edge_full_contexts   = vec![String::new(); total_fwd];
        let mut edge_embed_contexts: Vec<Option<String>> = vec![None; total_fwd];
        let mut cursor = offsets.clone();

        for e in edge_list {
            let pos = cursor[e.from as usize] as usize;
            edges[pos]               = e.to;
            edge_weights[pos]        = e.weight;
            edge_types[pos]          = e.edge_type.clone();
            edge_full_contexts[pos]  = e.full_context.clone();
            edge_embed_contexts[pos] = e.embed_context.clone();
            cursor[e.from as usize] += 1;
        }

        // ── reverse index (for bidirectional BFS) ────────────────────────────
        let mut rev_offsets = vec![0u32; n + 1];
        for i in 0..n {
            rev_offsets[i + 1] = rev_offsets[i] + in_degree[i];
        }

        let total_rev = rev_offsets[n] as usize;
        let mut rev_edges  = vec![0u32; total_rev];
        let mut rev_cursor = rev_offsets.clone();

        for e in edge_list {
            let pos = rev_cursor[e.to as usize] as usize;
            rev_edges[pos] = e.from;
            rev_cursor[e.to as usize] += 1;
        }

        // ── node_weight = (out + in) / max_combined ──────────────────────────
        // Using both directions gives a better importance signal than out-degree alone.
        // Hub nodes (many in-edges) and authority nodes (many out-edges) are both rewarded.
        let max_combined = (0..n)
            .map(|i| out_degree[i] + in_degree[i])
            .max()
            .unwrap_or(1) as f32;

        let mut nodes = nodes;
        for (i, node) in nodes.iter_mut().enumerate() {
            node.weight = (out_degree[i] + in_degree[i]) as f32 / max_combined;
        }

        let external_id_index: HashMap<String, u32> = nodes.iter()
            .filter(|n| !n.external_id.is_empty())
            .map(|n| (n.external_id.clone(), n.id))
            .collect();

        Self { offsets, edges, nodes, edge_weights, edge_types, edge_full_contexts, edge_embed_contexts, rev_offsets, rev_edges, external_id_index }
    }

    /// O(1) slice of out-neighbors — no allocation, cache-friendly.
    /// Returns empty slice for out-of-range node IDs instead of panicking.
    #[inline]
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let idx = node as usize;
        if idx + 1 >= self.offsets.len() { return &[]; }
        let start = self.offsets[idx] as usize;
        let end   = self.offsets[idx + 1] as usize;
        &self.edges[start..end]
    }

    #[inline]
    pub fn neighbor_weights(&self, node: u32) -> &[f32] {
        let idx = node as usize;
        if idx + 1 >= self.offsets.len() { return &[]; }
        let start = self.offsets[idx] as usize;
        let end   = self.offsets[idx + 1] as usize;
        &self.edge_weights[start..end]
    }

    #[inline]
    pub fn neighbor_types(&self, node: u32) -> &[String] {
        let idx = node as usize;
        if idx + 1 >= self.offsets.len() { return &[]; }
        let start = self.offsets[idx] as usize;
        let end   = self.offsets[idx + 1] as usize;
        &self.edge_types[start..end]
    }

    #[inline]
    pub fn neighbor_full_contexts(&self, node: u32) -> &[String] {
        let idx = node as usize;
        if idx + 1 >= self.offsets.len() { return &[]; }
        let start = self.offsets[idx] as usize;
        let end   = self.offsets[idx + 1] as usize;
        &self.edge_full_contexts[start..end]
    }

    #[inline]
    pub fn neighbor_embed_contexts(&self, node: u32) -> &[Option<String>] {
        let idx = node as usize;
        if idx + 1 >= self.offsets.len() { return &[]; }
        let start = self.offsets[idx] as usize;
        let end   = self.offsets[idx + 1] as usize;
        &self.edge_embed_contexts[start..end]
    }

    /// Iterate all edges as `EdgeInfo` — used by merge and persistence.
    pub fn all_edges(&self) -> Vec<EdgeInfo> {
        let mut result = Vec::with_capacity(self.edges.len());
        for nid in 0..self.num_nodes() as u32 {
            let neighbors      = self.neighbors(nid);
            let weights        = self.neighbor_weights(nid);
            let types          = self.neighbor_types(nid);
            let full_ctxs      = self.neighbor_full_contexts(nid);
            let embed_ctxs     = self.neighbor_embed_contexts(nid);
            for i in 0..neighbors.len() {
                result.push(EdgeInfo {
                    from:          nid,
                    to:            neighbors[i],
                    weight:        weights[i],
                    edge_type:     types[i].clone(),
                    full_context:  full_ctxs[i].clone(),
                    embed_context: embed_ctxs.get(i).cloned().unwrap_or(None),
                    edge_id:       0,
                });
            }
        }
        result
    }

    /// O(1) slice of in-neighbors (reverse direction) — for bidirectional BFS.
    /// Returns empty slice for out-of-range node IDs instead of panicking.
    #[inline]
    pub fn rev_neighbors(&self, node: u32) -> &[u32] {
        let idx = node as usize;
        if idx + 1 >= self.rev_offsets.len() { return &[]; }
        let start = self.rev_offsets[idx] as usize;
        let end   = self.rev_offsets[idx + 1] as usize;
        &self.rev_edges[start..end]
    }

    pub fn num_nodes(&self) -> usize { self.nodes.len() }
    pub fn num_edges(&self) -> usize { self.edges.len() }

    /// Look up a node's CSR index by its stable external_id.
    pub fn get_by_external_id(&self, external_id: &str) -> Option<u32> {
        self.external_id_index.get(external_id).copied()
    }

    /// Returns all edges (from, to, weight, edge_type) where both endpoints are in `node_ids`.
    /// Deduplicates by (from, to) pair — keeps the highest-weight edge when duplicates exist.
    /// Used to build the relation context sent to the LLM.
    pub fn edges_between(&self, node_ids: &HashSet<u32>) -> Vec<(u32, u32, f32, String)> {
        let mut best: std::collections::HashMap<(u32, u32), (f32, String)> =
            std::collections::HashMap::new();
        for &nid in node_ids {
            let nbrs  = self.neighbors(nid);
            let wts   = self.neighbor_weights(nid);
            let types = self.neighbor_types(nid);
            for ((&nb, &w), et) in nbrs.iter().zip(wts.iter()).zip(types.iter()) {
                if node_ids.contains(&nb) {
                    let entry = best.entry((nid, nb)).or_insert_with(|| (w, et.clone()));
                    if w > entry.0 { *entry = (w, et.clone()); }
                }
            }
        }
        best.into_iter().map(|((from, to), (w, et))| (from, to, w, et)).collect()
    }

    /// BFS expansion from seed nodes up to `max_depth` hops.
    ///
    /// Returns `(node_id, hop_distance, path_weight)` where `path_weight` is the
    /// weight of the edge that first reached this node (1.0 for seeds).
    /// This lets scoring reward strong-edge connections over weak ones.
    ///
    /// When `bidirectional=true`, also expands through reverse edges —
    /// useful for relationship mode where knowledge graphs have directed edges
    /// but logical connections flow both ways.
    pub fn bfs_expand(
        &self,
        seeds: &[u32],
        max_depth: u8,
        max_nodes: usize,
        bidirectional: bool,
    ) -> Vec<(u32, u8, f32)> {
        let mut visited: HashSet<u32>         = HashSet::with_capacity(max_nodes * 2);
        let mut result:  Vec<(u32, u8, f32)>  = Vec::with_capacity(max_nodes);
        let mut queue:   VecDeque<(u32, u8)>  = VecDeque::new();

        for &seed in seeds {
            if visited.insert(seed) {
                queue.push_back((seed, 0));
                result.push((seed, 0, 1.0)); // seeds: path_weight=1.0
            }
        }

        'bfs: while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // forward edges — include edge weights
            let fwd = self.neighbors(node);
            let wts = self.neighbor_weights(node);
            for (&nb, &w) in fwd.iter().zip(wts.iter()) {
                if visited.insert(nb) {
                    queue.push_back((nb, depth + 1));
                    result.push((nb, depth + 1, w));
                    if result.len() >= max_nodes { break 'bfs; }
                }
            }

            // reverse edges (relationship mode) — weight=1.0, no reverse weights stored
            if bidirectional {
                for &nb in self.rev_neighbors(node) {
                    if visited.insert(nb) {
                        queue.push_back((nb, depth + 1));
                        result.push((nb, depth + 1, 1.0));
                        if result.len() >= max_nodes { break 'bfs; }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> CsrGraph {
        let nodes: Vec<NodeInfo> = (0..5)
            .map(|i| NodeInfo {
                id:           i,
                name:         format!("node_{}", i),
                node_type:    "Entity".into(),
                weight:       0.0,
                props:        serde_json::Value::Null,
                full_context:  String::new(),
                embed_context: None,
                external_id:   format!("node_{}", i),
                image_url:     None,
            })
            .collect();

        let edges = vec![
            EdgeInfo { from: 0, to: 1, edge_type: "rel".into(), weight: 1.0, full_context: String::new(), embed_context: None, edge_id: 0 },
            EdgeInfo { from: 0, to: 2, edge_type: "rel".into(), weight: 0.5, full_context: String::new(), embed_context: None, edge_id: 1 },
            EdgeInfo { from: 1, to: 3, edge_type: "rel".into(), weight: 0.8, full_context: String::new(), embed_context: None, edge_id: 2 },
            EdgeInfo { from: 2, to: 4, edge_type: "rel".into(), weight: 0.6, full_context: String::new(), embed_context: None, edge_id: 3 },
            EdgeInfo { from: 3, to: 4, edge_type: "rel".into(), weight: 1.0, full_context: String::new(), embed_context: None, edge_id: 4 },
        ];

        CsrGraph::build(nodes, &edges)
    }

    #[test]
    fn test_neighbors() {
        let g = make_test_graph();
        assert_eq!(g.neighbors(0), &[1, 2]);
        assert_eq!(g.neighbors(1), &[3]);
    }

    #[test]
    fn test_rev_neighbors() {
        let g = make_test_graph();
        // node 4 is reached from node 2 and node 3
        let mut rev: Vec<u32> = g.rev_neighbors(4).to_vec();
        rev.sort();
        assert_eq!(rev, &[2, 3]);
        // node 1 is reached only from node 0
        assert_eq!(g.rev_neighbors(1), &[0]);
    }

    #[test]
    fn test_bfs_unidirectional() {
        let g = make_test_graph();
        let result = g.bfs_expand(&[0], 2, 100, false);
        let ids: Vec<u32> = result.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
    }

    #[test]
    fn test_bfs_path_weights() {
        let g = make_test_graph();
        let result = g.bfs_expand(&[0], 1, 100, false);
        // node 1 should have path_weight = 1.0 (edge 0→1)
        let n1 = result.iter().find(|(id, _, _)| *id == 1).unwrap();
        assert!((n1.2 - 1.0).abs() < 1e-6);
        // node 2 should have path_weight = 0.5 (edge 0→2)
        let n2 = result.iter().find(|(id, _, _)| *id == 2).unwrap();
        assert!((n2.2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bfs_bidirectional() {
        let g = make_test_graph();
        // from node 4, bidirectional should reach 2 and 3 (reverse edges)
        let result = g.bfs_expand(&[4], 1, 100, true);
        let ids: Vec<u32> = result.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_node_weight_uses_both_degrees() {
        let g = make_test_graph();
        // node 0: out=2, in=0 → combined=2
        // node 4: out=0, in=2 → combined=2
        // both should have the same weight (max_combined=3 for node 1: out=1,in=1... wait
        // node 1: out=1 (→3), in=1 (←0) → combined=2
        // node 3: out=1 (→4), in=1 (←1) → combined=2
        // node 2: out=1 (→4), in=1 (←0) → combined=2
        // max_combined = 2 → all weights = 1.0 except node 4 (in=2, out=0 → 2/2=1.0)
        // Actually let me re-check: node 0 has out=2 (→1,→2), in=0 → combined=2
        // max = 2 → node 0 weight = 2/2 = 1.0
        assert!((g.nodes[0].weight - 1.0).abs() < 1e-6);
        assert!((g.nodes[4].weight - 1.0).abs() < 1e-6);
    }
}
