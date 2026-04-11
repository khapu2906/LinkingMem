//! Unit tests for entity_resolution — resolve(), resolve_in_batch().
//!
//! Each test uses a small 4-node graph with orthogonal unit basis vectors
//! so similarity scores are exact and predictable.

use ai_graph_engine::{
    delta::DeltaGraph,
    entity_resolution::{resolve, resolve_in_batch, ResolutionConfig, ResolutionMode, ResolveResult},
    graph::csr::{CsrGraph, NodeInfo},
    vector::hnsw::HnswIndex,
};
use serde_json::Value;

// ── test fixtures ─────────────────────────────────────────────────────────────

/// Unit basis vector at dimension `hot` (already normalised — |v| = 1).
fn unit_vec(hot: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 8];
    v[hot] = 1.0;
    v
}

fn make_node(id: u32, name: &str, node_type: &str) -> NodeInfo {
    NodeInfo {
        id,
        name:          name.into(),
        node_type:     node_type.into(),
        weight:        0.0,
        props:         Value::Null,
        full_context:  String::new(),
        embed_context: None,
    }
}

/// 4-node graph — each node has a distinct unit basis vector.
///
/// id=0  "Alice"   Person    vec[0]
/// id=1  "Bob"     Person    vec[1]
/// id=2  "Acme"    Company   vec[2]
/// id=3  "Widget"  Product   vec[3]
fn make_fixtures() -> (CsrGraph, HnswIndex) {
    let nodes = vec![
        make_node(0, "Alice",  "Person"),
        make_node(1, "Bob",    "Person"),
        make_node(2, "Acme",   "Company"),
        make_node(3, "Widget", "Product"),
    ];
    let vecs: Vec<Vec<f32>> = (0..4).map(unit_vec).collect();
    let ids: Vec<u32>        = (0..4).collect();
    let graph = CsrGraph::build(nodes, &[]);
    let hnsw  = HnswIndex::build(vecs, ids, 4, 50).unwrap();
    (graph, hnsw)
}

fn cfg_embedding(threshold: f32, match_type: bool) -> ResolutionConfig {
    ResolutionConfig { mode: ResolutionMode::Embedding, threshold, match_type }
}

fn cfg_none() -> ResolutionConfig {
    ResolutionConfig { mode: ResolutionMode::None, threshold: 0.9, match_type: false }
}

// ── resolve() — mode: None ────────────────────────────────────────────────────

#[test]
fn resolve_none_mode_always_returns_new() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    // Even an exact match vector returns New when mode=None.
    let result = resolve("Person", &unit_vec(0), &graph, &delta, &hnsw, &cfg_none());
    assert!(matches!(result, ResolveResult::New));
}

// ── resolve() — exact match in graph ─────────────────────────────────────────

#[test]
fn resolve_exact_vec_matches_existing_node() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    // vec[0] is Alice's exact vector — cosine similarity = 1.0 ≥ threshold 0.9
    let result = resolve("Person", &unit_vec(0), &graph, &delta, &hnsw, &cfg_embedding(0.9, false));
    assert!(matches!(result, ResolveResult::Existing(0)));
}

#[test]
fn resolve_bob_vec_matches_bob() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    let result = resolve("Person", &unit_vec(1), &graph, &delta, &hnsw, &cfg_embedding(0.9, false));
    assert!(matches!(result, ResolveResult::Existing(1)));
}

// ── resolve() — below threshold → New ────────────────────────────────────────

#[test]
fn resolve_orthogonal_vec_returns_new() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    // vec[7] is orthogonal to all 4 graph vectors → cosine similarity = 0.0 < threshold
    let result = resolve("Person", &unit_vec(7), &graph, &delta, &hnsw, &cfg_embedding(0.9, false));
    assert!(matches!(result, ResolveResult::New));
}

// ── resolve() — match_type filtering ─────────────────────────────────────────

#[test]
fn resolve_match_type_wrong_type_returns_new() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    // vec[0] is Alice (Person). We're querying as "Company" → type mismatch → New.
    let result = resolve("Company", &unit_vec(0), &graph, &delta, &hnsw, &cfg_embedding(0.9, true));
    assert!(matches!(result, ResolveResult::New));
}

#[test]
fn resolve_match_type_same_type_matches() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    // vec[0] is Alice (Person). Same type → match.
    let result = resolve("Person", &unit_vec(0), &graph, &delta, &hnsw, &cfg_embedding(0.9, true));
    assert!(matches!(result, ResolveResult::Existing(0)));
}

#[test]
fn resolve_match_type_false_ignores_type_mismatch() {
    let (graph, hnsw) = make_fixtures();
    let delta = DeltaGraph::default();
    // match_type=false → type is irrelevant, similarity wins.
    let result = resolve("Company", &unit_vec(0), &graph, &delta, &hnsw, &cfg_embedding(0.9, false));
    assert!(matches!(result, ResolveResult::Existing(0)));
}

// ── resolve() — delta buffer path ────────────────────────────────────────────

#[test]
fn resolve_finds_match_in_delta_buffer() {
    let (graph, hnsw) = make_fixtures();
    // vec[5] is outside the HNSW (which only has 0..3) → HNSW returns 0-sim candidates.
    // But we add the same vector to the delta → delta linear scan finds it.
    let mut delta = DeltaGraph::default();
    delta.add_node(make_node(99, "DeltaNode", "Person"), unit_vec(5));

    let result = resolve("Person", &unit_vec(5), &graph, &delta, &hnsw, &cfg_embedding(0.9, false));
    assert!(matches!(result, ResolveResult::Existing(99)));
}

#[test]
fn resolve_delta_match_type_enforced() {
    let (graph, hnsw) = make_fixtures();
    let mut delta = DeltaGraph::default();
    // DeltaNode is a Company, but we query as Person with match_type=true → miss.
    delta.add_node(make_node(99, "DeltaNode", "Company"), unit_vec(5));

    let result = resolve("Person", &unit_vec(5), &graph, &delta, &hnsw, &cfg_embedding(0.9, true));
    assert!(matches!(result, ResolveResult::New));
}

#[test]
fn resolve_delta_below_threshold_returns_new() {
    let (graph, hnsw) = make_fixtures();
    let mut delta = DeltaGraph::default();
    // Put an orthogonal vector in delta → similarity 0.0 < threshold
    delta.add_node(make_node(99, "Unrelated", "Person"), unit_vec(6));

    let result = resolve("Person", &unit_vec(5), &graph, &delta, &hnsw, &cfg_embedding(0.9, false));
    assert!(matches!(result, ResolveResult::New));
}

// ── resolve_in_batch() ────────────────────────────────────────────────────────

#[test]
fn resolve_in_batch_finds_duplicate() {
    let cfg = cfg_embedding(0.9, false);
    // First entity already accepted into batch.
    let batch = vec![(make_node(10, "Alice", "Person"), unit_vec(0))];

    // Second entity with the same vector should be deduped to id=10.
    let result = resolve_in_batch("Person", &unit_vec(0), &batch, &cfg);
    assert_eq!(result, Some(10));
}

#[test]
fn resolve_in_batch_orthogonal_vec_returns_none() {
    let cfg = cfg_embedding(0.9, false);
    let batch = vec![(make_node(10, "Alice", "Person"), unit_vec(0))];

    // vec[7] is orthogonal → no match
    let result = resolve_in_batch("Person", &unit_vec(7), &batch, &cfg);
    assert_eq!(result, None);
}

#[test]
fn resolve_in_batch_mode_none_always_returns_none() {
    let cfg = cfg_none();
    // Even an exact duplicate is ignored when mode=None.
    let batch = vec![(make_node(10, "Alice", "Person"), unit_vec(0))];
    let result = resolve_in_batch("Person", &unit_vec(0), &batch, &cfg);
    assert_eq!(result, None);
}

#[test]
fn resolve_in_batch_match_type_enforced() {
    let cfg = cfg_embedding(0.9, true);
    let batch = vec![(make_node(10, "Alice", "Person"), unit_vec(0))];
    // Same vector but different entity_type → no match
    let result = resolve_in_batch("Company", &unit_vec(0), &batch, &cfg);
    assert_eq!(result, None);
}

#[test]
fn resolve_in_batch_empty_batch_returns_none() {
    let cfg = cfg_embedding(0.9, false);
    let result = resolve_in_batch("Person", &unit_vec(0), &[], &cfg);
    assert_eq!(result, None);
}

#[test]
fn resolve_in_batch_returns_first_match() {
    let cfg = cfg_embedding(0.9, false);
    // Two candidates with the same vector — should return the first one.
    let batch = vec![
        (make_node(10, "Alice", "Person"), unit_vec(0)),
        (make_node(20, "Alicia", "Person"), unit_vec(0)),
    ];
    let result = resolve_in_batch("Person", &unit_vec(0), &batch, &cfg);
    assert_eq!(result, Some(10));
}
