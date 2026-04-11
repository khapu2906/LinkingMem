//! Tests for DeltaStore batch methods: add_nodes_batch, add_edges_batch.
//!
//! Each test creates a DeltaStore backed by a temp directory so WAL I/O
//! is exercised for real (not mocked), then verifies:
//!   - in-memory state matches what sequential add_node/add_edge would produce
//!   - WAL round-trips correctly (replay restores identical state)

use std::sync::Arc;
use serde_json::Value;
use tempfile::TempDir;

use ai_graph_engine::{
    delta::DeltaStore,
    graph::csr::{EdgeInfo, NodeInfo},
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_node(id: u32, name: &str) -> NodeInfo {
    NodeInfo {
        id,
        name:          name.into(),
        node_type:     "Person".into(),
        weight:        0.0,
        props:         Value::Null,
        full_context:  String::new(),
        embed_context: None,
    }
}

fn make_edge(from: u32, to: u32) -> EdgeInfo {
    EdgeInfo {
        from,
        to,
        edge_type:     "knows".into(),
        weight:        1.0,
        full_context:  String::new(),
        embed_context: None,
    }
}

/// Unit basis vector at dimension `hot` (already normalised — |v| = 1).
fn unit_vec(hot: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 8];
    v[hot] = 1.0;
    v
}

fn new_store(tmp: &TempDir) -> Arc<DeltaStore> {
    Arc::new(DeltaStore::new(tmp.path().to_path_buf(), 10_000))
}

// ── add_nodes_batch ───────────────────────────────────────────────────────────

#[test]
fn add_nodes_batch_commits_correct_count() {
    let tmp = TempDir::new().unwrap();
    let store = new_store(&tmp);

    let batch: Vec<(NodeInfo, Vec<f32>)> = (0..5)
        .map(|i| (make_node(i, &format!("node{i}")), unit_vec(i as usize)))
        .collect();

    store.add_nodes_batch(batch);

    assert_eq!(store.size(), 5);
}

#[test]
fn add_nodes_batch_same_state_as_sequential() {
    let tmp_batch = TempDir::new().unwrap();
    let tmp_seq   = TempDir::new().unwrap();

    let store_batch = new_store(&tmp_batch);
    let store_seq   = new_store(&tmp_seq);

    let items: Vec<(NodeInfo, Vec<f32>)> = (0..4)
        .map(|i| (make_node(i, &format!("n{i}")), unit_vec(i as usize)))
        .collect();

    // batch path
    store_batch.add_nodes_batch(items.clone());

    // sequential path
    for (node, vec) in items {
        store_seq.add_node(node, vec);
    }

    assert_eq!(store_batch.size(), store_seq.size());

    let snap_batch = store_batch.read();
    let snap_seq   = store_seq.read();

    assert_eq!(snap_batch.new_nodes.len(), snap_seq.new_nodes.len());
    for i in 0..snap_batch.new_nodes.len() {
        assert_eq!(snap_batch.new_nodes[i].id,   snap_seq.new_nodes[i].id);
        assert_eq!(snap_batch.new_nodes[i].name, snap_seq.new_nodes[i].name);
    }
}

#[test]
fn add_nodes_batch_empty_is_noop() {
    let tmp = TempDir::new().unwrap();
    let store = new_store(&tmp);

    store.add_nodes_batch(vec![]);

    assert_eq!(store.size(), 0);
    // WAL should not have been created (or if it was, it's empty)
    let wal_path = tmp.path().join("delta.wal");
    if wal_path.exists() {
        assert_eq!(std::fs::read_to_string(&wal_path).unwrap().trim(), "");
    }
}

#[test]
fn add_nodes_batch_wal_replays_all_entries() {
    let tmp = TempDir::new().unwrap();
    let store = new_store(&tmp);

    let batch: Vec<(NodeInfo, Vec<f32>)> = (0..6)
        .map(|i| (make_node(i, &format!("name{i}")), unit_vec(i as usize)))
        .collect();

    store.add_nodes_batch(batch);
    assert_eq!(store.size(), 6);

    // Drop the store and create a fresh one backed by the same directory.
    // replay_wal() should restore all 6 nodes.
    drop(store);

    let store2 = new_store(&tmp);
    let replayed = store2.replay_wal();

    assert_eq!(replayed, 6);
    assert_eq!(store2.size(), 6);
}

// ── add_edges_batch ───────────────────────────────────────────────────────────

#[test]
fn add_edges_batch_commits_correct_count() {
    let tmp = TempDir::new().unwrap();
    let store = new_store(&tmp);

    // add nodes first so the adjacency map is populated
    let nodes: Vec<(NodeInfo, Vec<f32>)> = (0..4)
        .map(|i| (make_node(i, &format!("n{i}")), unit_vec(i as usize)))
        .collect();
    store.add_nodes_batch(nodes);

    let edges: Vec<(EdgeInfo, Vec<f32>)> = vec![
        (make_edge(0, 1), unit_vec(4)),
        (make_edge(1, 2), unit_vec(5)),
        (make_edge(2, 3), unit_vec(6)),
    ];
    store.add_edges_batch(edges);

    // size = nodes + edges
    assert_eq!(store.size(), 4 + 3);
}

#[test]
fn add_edges_batch_same_state_as_sequential() {
    let tmp_batch = TempDir::new().unwrap();
    let tmp_seq   = TempDir::new().unwrap();

    let store_batch = new_store(&tmp_batch);
    let store_seq   = new_store(&tmp_seq);

    let edges: Vec<(EdgeInfo, Vec<f32>)> = vec![
        (make_edge(0, 1), unit_vec(0)),
        (make_edge(1, 2), unit_vec(1)),
        (make_edge(3, 0), unit_vec(2)),
    ];

    store_batch.add_edges_batch(edges.clone());
    for (edge, vec) in edges {
        store_seq.add_edge(edge, vec);
    }

    let snap_b = store_batch.read();
    let snap_s = store_seq.read();

    assert_eq!(snap_b.new_edges.len(), snap_s.new_edges.len());
    for i in 0..snap_b.new_edges.len() {
        assert_eq!(snap_b.new_edges[i].from, snap_s.new_edges[i].from);
        assert_eq!(snap_b.new_edges[i].to,   snap_s.new_edges[i].to);
    }
}

#[test]
fn add_edges_batch_wal_replays_all_entries() {
    let tmp = TempDir::new().unwrap();
    let store = new_store(&tmp);

    let edges: Vec<(EdgeInfo, Vec<f32>)> = (0..4_u32)
        .map(|i| (make_edge(i, i + 1), unit_vec(i as usize)))
        .collect();

    store.add_edges_batch(edges);
    assert_eq!(store.size(), 4);

    drop(store);

    let store2 = new_store(&tmp);
    let replayed = store2.replay_wal();

    assert_eq!(replayed, 4);
    assert_eq!(store2.size(), 4);
}

// ── mixed batch (nodes + edges) ───────────────────────────────────────────────

#[test]
fn mixed_batch_nodes_and_edges_wal_replays_correctly() {
    let tmp = TempDir::new().unwrap();
    let store = new_store(&tmp);

    let nodes: Vec<(NodeInfo, Vec<f32>)> = (0..3)
        .map(|i| (make_node(i, &format!("x{i}")), unit_vec(i as usize)))
        .collect();
    let edges: Vec<(EdgeInfo, Vec<f32>)> = vec![
        (make_edge(0, 1), unit_vec(3)),
        (make_edge(1, 2), unit_vec(4)),
    ];

    store.add_nodes_batch(nodes);
    store.add_edges_batch(edges);

    assert_eq!(store.size(), 5);

    drop(store);

    let store2 = new_store(&tmp);
    let replayed = store2.replay_wal();

    assert_eq!(replayed, 5);
    assert_eq!(store2.size(), 5);

    let snap = store2.read();
    assert_eq!(snap.new_nodes.len(), 3);
    assert_eq!(snap.new_edges.len(), 2);
}
