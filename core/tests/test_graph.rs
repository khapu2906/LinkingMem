//! Unit tests — CsrGraph
//! cargo test graph

use ai_graph_engine::graph::csr::{CsrGraph, EdgeInfo, NodeInfo};

fn node(id: u32) -> NodeInfo {
    NodeInfo { id, name: format!("node_{id}"), node_type: "Entity".into(), weight: 0.0, props: serde_json::Value::Null, full_context: String::new(), embed_context: None }
}
fn edge(from: u32, to: u32) -> EdgeInfo {
    EdgeInfo { from, to, edge_type: "rel".into(), weight: 1.0, full_context: String::new(), embed_context: None }
}
fn wedge(from: u32, to: u32, w: f32) -> EdgeInfo {
    EdgeInfo { from, to, edge_type: "rel".into(), weight: w, full_context: String::new(), embed_context: None }
}

/// Graph:  0 → 1 (w=1.0), 0 → 2 (w=0.5)
///         1 → 3 (w=0.8)
///         2 → 3, 2 → 4
///         3 → 4
fn make_graph() -> CsrGraph {
    CsrGraph::build(
        (0..5).map(node).collect(),
        &[wedge(0,1,1.0), wedge(0,2,0.5), wedge(1,3,0.8), edge(2,3), edge(2,4), edge(3,4)],
    )
}

// ── structure ────────────────────────────────────────────────────────────────

#[test]
fn node_and_edge_counts() {
    let g = make_graph();
    assert_eq!(g.num_nodes(), 5);
    assert_eq!(g.num_edges(), 6);
}

#[test]
fn neighbors_correct() {
    let g = make_graph();
    assert_eq!(g.neighbors(0), &[1, 2]);
    assert_eq!(g.neighbors(1), &[3]);
    assert_eq!(g.neighbors(2), &[3, 4]);
    assert_eq!(g.neighbors(3), &[4]);
    assert_eq!(g.neighbors(4), &[] as &[u32]);
}

#[test]
fn neighbor_weights_parallel() {
    let g = make_graph();
    for id in 0..5u32 {
        assert_eq!(g.neighbors(id).len(), g.neighbor_weights(id).len());
    }
}

#[test]
fn edge_weights_stored_correctly() {
    let g = make_graph();
    let wts = g.neighbor_weights(0);
    assert!((wts[0] - 1.0).abs() < 1e-6, "0→1 weight");
    assert!((wts[1] - 0.5).abs() < 1e-6, "0→2 weight");
}

// ── reverse edges ─────────────────────────────────────────────────────────────

#[test]
fn rev_neighbors_correct() {
    let g = make_graph();
    let mut rev3: Vec<u32> = g.rev_neighbors(3).to_vec(); rev3.sort();
    assert_eq!(rev3, &[1, 2]);
    let mut rev4: Vec<u32> = g.rev_neighbors(4).to_vec(); rev4.sort();
    assert_eq!(rev4, &[2, 3]);
    assert!(g.rev_neighbors(0).is_empty());
}

// ── node_weight uses in+out degree ────────────────────────────────────────────

#[test]
fn node_weights_in_0_1() {
    let g = make_graph();
    for n in &g.nodes {
        assert!(n.weight >= 0.0 && n.weight <= 1.0, "node {} weight={}", n.id, n.weight);
    }
}

#[test]
fn node_weight_max_is_1() {
    let g = make_graph();
    let max = g.nodes.iter().map(|n| n.weight).fold(0.0f32, f32::max);
    assert!((max - 1.0).abs() < 1e-6);
}

#[test]
fn node_weight_considers_in_degree() {
    let g = make_graph();
    // node 4: out=0 in=2 — old out-only formula → 0, new combined → same as node 0 (out=2 in=0)
    assert!(g.nodes[4].weight > 0.0, "leaf node with in-edges must have weight > 0");
    assert!(
        (g.nodes[0].weight - g.nodes[4].weight).abs() < 1e-6,
        "node 0 (2 out) and node 4 (2 in) should have equal combined weight"
    );
}

// ── BFS ───────────────────────────────────────────────────────────────────────

#[test]
fn bfs_depth0_returns_only_seeds() {
    let g = make_graph();
    let r = g.bfs_expand(&[0], 0, 100, false);
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].0, r[0].1), (0, 0));
}

#[test]
fn bfs_depth1_correct_hops() {
    let g = make_graph();
    let hops: std::collections::HashMap<u32, u8> =
        g.bfs_expand(&[0], 1, 100, false).into_iter().map(|(id,h,_)| (id,h)).collect();
    assert_eq!(hops[&0], 0);
    assert_eq!(hops[&1], 1);
    assert_eq!(hops[&2], 1);
    assert!(!hops.contains_key(&3));
}

#[test]
fn bfs_depth2_reaches_all_nodes() {
    let g = make_graph();
    let ids: Vec<u32> = g.bfs_expand(&[0], 2, 100, false).iter().map(|(id,_,_)| *id).collect();
    for i in 0..5u32 { assert!(ids.contains(&i), "node {i} missing"); }
}

#[test]
fn bfs_respects_max_nodes() {
    let g = make_graph();
    assert!(g.bfs_expand(&[0], 10, 2, false).len() <= 2);
}

#[test]
fn bfs_no_duplicates_with_multiple_seeds() {
    let g = make_graph();
    let result = g.bfs_expand(&[0, 1], 1, 100, false);
    let ids: Vec<u32> = result.iter().map(|(id,_,_)| *id).collect();
    let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(ids.len(), unique.len());
}

// ── BFS path weights ──────────────────────────────────────────────────────────

#[test]
fn bfs_seed_has_path_weight_1() {
    let g = make_graph();
    let r = g.bfs_expand(&[0], 1, 100, false);
    let seed = r.iter().find(|(id,_,_)| *id == 0).unwrap();
    assert!((seed.2 - 1.0).abs() < 1e-6);
}

#[test]
fn bfs_path_weight_matches_edge_weight() {
    let g = make_graph();
    let r = g.bfs_expand(&[0], 1, 100, false);
    let n1 = r.iter().find(|(id,_,_)| *id == 1).unwrap();
    let n2 = r.iter().find(|(id,_,_)| *id == 2).unwrap();
    assert!((n1.2 - 1.0).abs() < 1e-6, "0→1 w=1.0, got {}", n1.2);
    assert!((n2.2 - 0.5).abs() < 1e-6, "0→2 w=0.5, got {}", n2.2);
}

// ── BFS bidirectional ─────────────────────────────────────────────────────────

#[test]
fn bidirectional_reaches_upstream_nodes() {
    let g = make_graph();
    let uni_ids: std::collections::HashSet<u32> =
        g.bfs_expand(&[4], 2, 100, false).iter().map(|(id,_,_)| *id).collect();
    let bi_ids: std::collections::HashSet<u32> =
        g.bfs_expand(&[4], 2, 100, true).iter().map(|(id,_,_)| *id).collect();

    assert!(!uni_ids.contains(&2), "node 2 unreachable uni from leaf 4");
    assert!( bi_ids.contains(&2), "node 2 reachable via reverse edge");
    assert!( bi_ids.contains(&3), "node 3 reachable via reverse edge");
}

#[test]
fn bidirectional_no_duplicates() {
    let g = make_graph();
    let result = g.bfs_expand(&[2], 2, 100, true);
    let ids: Vec<u32> = result.iter().map(|(id,_,_)| *id).collect();
    let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(ids.len(), unique.len());
}

// ── builder round-trip ────────────────────────────────────────────────────────

#[test]
fn save_and_load_roundtrip() {
    use ai_graph_engine::graph::builder::{load, save};
    use ai_graph_engine::storage::LocalStorage;
    use tempfile::tempdir;

    let g = make_graph();
    let dir = tempdir().unwrap();
    let storage = LocalStorage::new(dir.path().to_path_buf());
    save(&g, &storage).unwrap();
    let g2 = load(&storage).unwrap();

    assert_eq!(g2.num_nodes(), g.num_nodes());
    assert_eq!(g2.num_edges(), g.num_edges());
    for id in 0..5u32 {
        assert_eq!(g2.neighbors(id), g.neighbors(id));
    }
}

#[test]
fn from_json_payload_parses_correctly() {
    use ai_graph_engine::graph::builder::from_json_payload;

    let g = from_json_payload(&serde_json::json!({
        "entities": [
            {"id":"a","name":"Alice","type":"Person","props":{}},
            {"id":"b","name":"Bob",  "type":"Person","props":{}},
            {"id":"c","name":"Acme", "type":"Company","props":{}}
        ],
        "relations": [
            {"from":"a","to":"c","type":"works_at","weight":1.0},
            {"from":"b","to":"c","type":"works_at","weight":1.0}
        ]
    })).unwrap();

    assert_eq!(g.num_nodes(), 3);
    assert_eq!(g.num_edges(), 2);
    assert_eq!(g.neighbors(0), &[2]);
    assert_eq!(g.neighbors(1), &[2]);
    assert!(g.neighbors(2).is_empty());
}

#[test]
fn from_json_payload_ignores_unknown_refs() {
    use ai_graph_engine::graph::builder::from_json_payload;

    let g = from_json_payload(&serde_json::json!({
        "entities": [{"id":"a","name":"A","type":"X","props":{}}],
        "relations": [{"from":"a","to":"GHOST","type":"r","weight":1.0}]
    })).unwrap();

    assert_eq!(g.num_nodes(), 1);
    assert_eq!(g.num_edges(), 0);
}
