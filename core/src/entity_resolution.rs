/// Entity Resolution — link new entities to existing graph nodes.
///
/// When new text is ingested, the extraction plugin creates entity names that may
/// already exist in the graph (e.g. "Alice" extracted again after first ingest).
/// Without resolution these become duplicate, disconnected nodes.
///
/// This module provides embedding-similarity-based resolution:
///   1. Embed the new entity name.
///   2. Search the HNSW index for the top-k nearest existing nodes.
///   3. If the best match has cosine similarity ≥ threshold → resolve to that node.
///   4. Also check the in-memory delta buffer (not yet in HNSW) via linear scan.
///   5. Also check other entities in the same ingest batch (intra-batch dedup).
///
/// Resolution modes (extensible):
///   - `none`      — always create new nodes (original behaviour)
///   - `embedding` — similarity search (default, no extra LLM calls)
///
/// The `llm` mode (hybrid: HNSW candidates + LLM disambiguation) is reserved for
/// a future release and can be selected per-request via the `resolution` payload field.

use serde::{Deserialize, Serialize};

use crate::delta::DeltaGraph;
use crate::graph::csr::CsrGraph;
use crate::vector::{hnsw::HnswIndex, store::cosine_sim};

// ── mode ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionMode {
    /// Always create a new node — no matching attempted.
    None,
    /// Match by embedding cosine similarity (default).
    Embedding,
    // Llm,  ← reserved for hybrid LLM-assisted resolution
}

impl Default for ResolutionMode {
    fn default() -> Self {
        Self::Embedding
    }
}

// ── config ───────────────────────────────────────────────────────────────────

/// Controls how entity resolution behaves during ingest.
///
/// Can be set globally in `plugins.toml` / env vars, or overridden
/// per-request in the ingest payload body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolutionConfig {
    pub mode: ResolutionMode,
    /// Minimum cosine similarity [0..1] to consider two entities as the same.
    /// Higher = stricter (fewer false merges). Recommended: 0.90–0.95.
    pub threshold: f32,
    /// When true, entities must have the same `node_type` to be merged.
    /// E.g. "Apple" (Company) will NOT merge with "Apple" (Fruit).
    pub match_type: bool,
}

impl Default for ResolutionConfig {
    fn default() -> Self {
        Self {
            mode:       ResolutionMode::Embedding,
            threshold:  0.92,
            match_type: false,
        }
    }
}

// ── result ───────────────────────────────────────────────────────────────────

pub enum ResolveResult {
    /// A sufficiently similar node already exists — reuse this ID.
    /// No new node needs to be created; edges can reference this ID directly.
    Existing(u32),
    /// No match found — caller should create a new node.
    New,
}

// ── resolver ─────────────────────────────────────────────────────────────────

/// Try to resolve a single candidate entity against:
///   1. The main CSR graph via HNSW (fast ANN search).
///   2. The in-memory delta buffer via linear scan (small, not yet in HNSW).
///
/// Returns `Existing(id)` on the best match above threshold, or `New`.
pub fn resolve(
    entity_type: &str,
    embedding:   &[f32],
    graph:       &CsrGraph,
    delta:       &DeltaGraph,
    hnsw:        &HnswIndex,
    cfg:         &ResolutionConfig,
) -> ResolveResult {
    match cfg.mode {
        ResolutionMode::None      => ResolveResult::New,
        ResolutionMode::Embedding => resolve_by_embedding(entity_type, embedding, graph, delta, hnsw, cfg),
    }
}

fn resolve_by_embedding(
    entity_type: &str,
    embedding:   &[f32],
    graph:       &CsrGraph,
    delta:       &DeltaGraph,
    hnsw:        &HnswIndex,
    cfg:         &ResolutionConfig,
) -> ResolveResult {
    // ── 1. Main graph via HNSW ────────────────────────────────────────────
    // All vectors are unit-normalised; HNSW distance ≈ 1 − cosine_similarity.
    let candidates = hnsw.search(embedding, 5);

    let best_graph = candidates.iter()
        .map(|&(id, dist)| (id, 1.0_f32 - dist))
        .filter(|&(id, sim)| {
            sim >= cfg.threshold && {
                let node = &graph.nodes[id as usize];
                !cfg.match_type || node.node_type == entity_type
            }
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((id, sim)) = best_graph {
        tracing::debug!(
            "entity_resolution: graph  id={id} name={:?} sim={sim:.3}",
            graph.nodes[id as usize].name
        );
        return ResolveResult::Existing(id);
    }

    // ── 2. Delta buffer (linear scan — delta is small by design) ─────────
    let best_delta = delta.new_nodes.iter()
        .zip(delta.new_vecs.iter())
        .map(|(node, vec)| (node.id, node.node_type.as_str(), cosine_sim(embedding, vec)))
        .filter(|&(_, ntype, sim)| {
            sim >= cfg.threshold && (!cfg.match_type || ntype == entity_type)
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((id, _, sim)) = best_delta {
        tracing::debug!("entity_resolution: delta  id={id} sim={sim:.3}");
        return ResolveResult::Existing(id);
    }

    ResolveResult::New
}

/// Intra-batch dedup: check new entities resolved within the CURRENT ingest
/// batch (not yet committed to delta).
///
/// `batch` is a list of `(NodeInfo, embedding)` pairs already decided as "New"
/// in this batch. Returns the id of the first match, or None.
pub fn resolve_in_batch(
    entity_type: &str,
    embedding:   &[f32],
    batch:       &[(crate::graph::csr::NodeInfo, Vec<f32>)],
    cfg:         &ResolutionConfig,
) -> Option<u32> {
    if matches!(cfg.mode, ResolutionMode::None) {
        return None;
    }

    batch.iter()
        .find(|(node, vec)| {
            let sim = cosine_sim(embedding, vec);
            sim >= cfg.threshold && (!cfg.match_type || node.node_type == entity_type)
        })
        .map(|(node, _)| node.id)
}
