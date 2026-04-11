//! Unit tests — scoring formula, QueryOptions
//! cargo test scoring

use ai_graph_engine::query::{QueryOptions, ScoringWeights};

// ── ScoringWeights presets ────────────────────────────────────────────────────

#[test]
fn all_weights_sum_to_1() {
    let presets = [
        ScoringWeights::balanced(),
        ScoringWeights::semantic_search(),
        ScoringWeights::relationship(),
        ScoringWeights::entity_lookup(),
    ];
    for w in &presets {
        let sum = w.alpha + w.beta + w.gamma;
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights don't sum to 1: α={} β={} γ={} sum={}",
            w.alpha, w.beta, w.gamma, sum
        );
    }
}

#[test]
fn semantic_mode_has_highest_alpha() {
    let s = ScoringWeights::semantic_search();
    assert!(s.alpha > s.beta);
    assert!(s.alpha > s.gamma);
}

#[test]
fn relationship_mode_has_highest_beta() {
    let r = ScoringWeights::relationship();
    assert!(r.beta > r.alpha);
    assert!(r.beta > r.gamma);
}

#[test]
fn entity_mode_has_highest_gamma() {
    let e = ScoringWeights::entity_lookup();
    assert!(e.gamma > e.beta);
}

// ── scoring formula ───────────────────────────────────────────────────────────

fn score(vsim: f32, hop: u8, nw: f32, w: &ScoringWeights) -> f32 {
    let proximity = 1.0 / (hop as f32 + 1.0);
    w.alpha * vsim + w.beta * proximity + w.gamma * nw
}

#[test]
fn seed_node_scores_higher_than_distant_node() {
    let w = ScoringWeights::balanced();
    // seed: hop=0, high vsim
    let seed_score  = score(0.9, 0, 0.8, &w);
    // distant: hop=2, low vsim
    let dist_score  = score(0.3, 2, 0.2, &w);
    assert!(seed_score > dist_score, "seed={seed_score}, distant={dist_score}");
}

#[test]
fn high_vsim_dominates_in_semantic_mode() {
    let w = ScoringWeights::semantic_search();
    let high_vsim = score(0.95, 2, 0.1, &w);
    let low_vsim  = score(0.20, 0, 1.0, &w);
    assert!(high_vsim > low_vsim);
}

#[test]
fn hop0_proximity_is_1() {
    let proximity = 1.0f32 / (0u8 as f32 + 1.0);
    assert!((proximity - 1.0).abs() < 1e-6);
}

#[test]
fn proximity_decreases_with_hops() {
    let p: Vec<f32> = (0..4).map(|h| 1.0 / (h as f32 + 1.0)).collect();
    for w in p.windows(2) {
        assert!(w[0] > w[1], "proximity should decrease: {} > {}", w[0], w[1]);
    }
}

#[test]
fn score_always_in_valid_range() {
    // with unit-normalised vectors, vsim ∈ [-1, 1].
    // proximity ∈ (0, 1], node_weight ∈ [0, 1].
    // worst-case min score (all weights balanced):
    let w = ScoringWeights::balanced();
    let min_score = score(-1.0, 10, 0.0, &w);
    let max_score = score(1.0,   0, 1.0, &w);
    assert!(min_score <= max_score);
    // max score should be ≤ 1 when all inputs are at their maximum
    assert!(max_score <= 1.0 + 1e-5);
}

// ── QueryOptions defaults ─────────────────────────────────────────────────────

#[test]
fn default_options_sensible() {
    let opts = QueryOptions::default();
    assert!(opts.hnsw_k > 0);
    assert!(opts.bfs_depth > 0);
    assert!(opts.bfs_max_nodes >= opts.context_top_n);
    assert!(opts.context_top_n > 0);
}
