//! Tests for image node support (v0.3.0).
//!
//! Covers:
//!   - NodeInfo serde roundtrip with image_url (present and absent)
//!   - from_json_payload() picks up image_url from entity JSON
//!   - ExportNode serde roundtrip with image_url
//!   - Image node persisted through DeltaStore WAL and survives replay
//!   - Image node merge: image_url preserved after merge_into()
//!   - CsrGraph::build() preserves image_url in node metadata
//!   - graph save/load roundtrip via builder::save / builder::load

use std::sync::Arc;

use ai_graph_engine::{
    delta::DeltaStore,
    graph::{
        builder::{from_json_payload, load as load_graph, save as save_graph},
        csr::{CsrGraph, EdgeInfo, NodeInfo},
    },
    storage::LocalStorage,
    vector::{hnsw::normalise, store::VectorStore},
};
use serde_json::{json, Value};
use tempfile::tempdir;

// ── helpers ───────────────────────────────────────────────────────────────────

fn text_node(id: u32, name: &str) -> NodeInfo {
    NodeInfo {
        id,
        external_id:   format!("Entity:{name}"),
        name:          name.into(),
        node_type:     "Entity".into(),
        weight:        0.0,
        props:         Value::Null,
        full_context:  String::new(),
        embed_context: None,
        image_url:     None,
    }
}

fn image_node(id: u32, name: &str, url: &str) -> NodeInfo {
    NodeInfo {
        id,
        external_id:   format!("Image:{name}"),
        name:          name.into(),
        node_type:     "Image".into(),
        weight:        0.0,
        props:         Value::Null,
        full_context:  String::new(),
        embed_context: None,
        image_url:     Some(url.into()),
    }
}

// ── serde ─────────────────────────────────────────────────────────────────────

#[test]
fn node_info_serde_roundtrip_no_image() {
    let n = text_node(0, "Alice");
    let json = serde_json::to_string(&n).unwrap();
    let back: NodeInfo = serde_json::from_str(&json).unwrap();
    assert!(back.image_url.is_none());
    assert_eq!(back.name, "Alice");
}

#[test]
fn node_info_serde_roundtrip_with_image() {
    let n = image_node(1, "photo_01", "https://example.com/img.jpg");
    let json = serde_json::to_string(&n).unwrap();
    let back: NodeInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.image_url.as_deref(), Some("https://example.com/img.jpg"));
    assert_eq!(back.node_type, "Image");
}

#[test]
fn node_info_serde_default_image_url_missing() {
    // Old snapshots without image_url field should default to None.
    let raw = r#"{"id":0,"name":"Alice","node_type":"Entity","weight":0.5,"props":null,"full_context":"","external_id":"Entity:Alice"}"#;
    let n: NodeInfo = serde_json::from_str(raw).unwrap();
    assert!(n.image_url.is_none());
}

// ── from_json_payload ─────────────────────────────────────────────────────────

#[test]
fn builder_parses_image_url_from_entity_json() {
    let payload = json!({
        "entities": [
            {"id": "img1", "name": "Landscape", "type": "Image",  "image_url": "https://cdn.example.com/landscape.jpg"},
            {"id": "txt1", "name": "Alice",     "type": "Person"}
        ],
        "relations": []
    });
    let partial = from_json_payload(&payload).unwrap();
    assert_eq!(partial.nodes.len(), 2);

    let img = partial.nodes.iter().find(|n| n.name == "Landscape").unwrap();
    assert_eq!(img.image_url.as_deref(), Some("https://cdn.example.com/landscape.jpg"));
    assert_eq!(img.node_type, "Image");

    let txt = partial.nodes.iter().find(|n| n.name == "Alice").unwrap();
    assert!(txt.image_url.is_none());
}

#[test]
fn builder_empty_image_url_becomes_none() {
    let payload = json!({
        "entities": [{"id": "e1", "name": "Alice", "type": "Person", "image_url": ""}],
        "relations": []
    });
    let partial = from_json_payload(&payload).unwrap();
    assert!(partial.nodes[0].image_url.is_none());
}

#[test]
fn builder_base64_data_uri_preserved() {
    let payload = json!({
        "entities": [{"id": "img2", "name": "Diagram", "type": "Image", "image_url": "data:image/png;base64,abc123=="}],
        "relations": []
    });
    let partial = from_json_payload(&payload).unwrap();
    assert_eq!(partial.nodes[0].image_url.as_deref(), Some("data:image/png;base64,abc123=="));
}

// ── image_url split logic ─────────────────────────────────────────────────────

#[test]
fn image_node_indices_split_correctly() {
    let nodes = vec![
        text_node(0, "Alice"),
        image_node(1, "photo_01", "https://example.com/a.jpg"),
        text_node(2, "Bob"),
        image_node(3, "photo_02", "data:image/png;base64,abc"),
    ];

    let mut text_indices:  Vec<usize> = Vec::new();
    let mut image_indices: Vec<usize> = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        if n.image_url.is_some() { image_indices.push(i); }
        else                     { text_indices.push(i); }
    }

    assert_eq!(text_indices,  vec![0, 2]);
    assert_eq!(image_indices, vec![1, 3]);
}

#[test]
fn all_text_nodes_means_no_image_indices() {
    let nodes = vec![text_node(0, "Alice"), text_node(1, "Bob")];
    let image_indices: Vec<usize> = nodes.iter().enumerate()
        .filter(|(_, n)| n.image_url.is_some())
        .map(|(i, _)| i)
        .collect();
    assert!(image_indices.is_empty());
}

#[test]
fn all_image_nodes_means_no_text_embed_needed() {
    let nodes = vec![
        image_node(0, "photo_a", "https://a.jpg"),
        image_node(1, "photo_b", "https://b.jpg"),
    ];
    let text_indices: Vec<usize> = nodes.iter().enumerate()
        .filter(|(_, n)| n.image_url.is_none())
        .map(|(i, _)| i)
        .collect();
    assert!(text_indices.is_empty());
}

// ── CsrGraph preserves image_url ─────────────────────────────────────────────

#[test]
fn csr_build_preserves_image_url() {
    let nodes = vec![
        text_node(0, "Alice"),
        image_node(1, "photo_01", "https://example.com/img.jpg"),
    ];
    let edges = vec![
        EdgeInfo { from: 0, to: 1, edge_type: "depicted_in".into(), weight: 1.0,
                   full_context: String::new(), embed_context: None, edge_id: 0 },
    ];
    let g = CsrGraph::build(nodes, &edges);

    assert!(g.nodes[0].image_url.is_none());
    assert_eq!(g.nodes[1].image_url.as_deref(), Some("https://example.com/img.jpg"));
    // edges still exist
    assert_eq!(g.num_edges(), 1);
}

#[test]
fn csr_image_node_included_in_bfs() {
    let nodes = vec![
        text_node(0, "Alice"),
        image_node(1, "photo_01", "https://example.com/img.jpg"),
        text_node(2, "Bob"),
    ];
    let edges = vec![
        EdgeInfo { from: 0, to: 1, edge_type: "has_photo".into(), weight: 1.0,
                   full_context: String::new(), embed_context: None, edge_id: 0 },
        EdgeInfo { from: 1, to: 2, edge_type: "depicts".into(), weight: 0.8,
                   full_context: String::new(), embed_context: None, edge_id: 1 },
    ];
    let g = CsrGraph::build(nodes, &edges);
    let reachable: Vec<u32> = g.bfs_expand(&[0], 2, 100, false)
        .into_iter().map(|(id, _, _)| id).collect();

    // Alice → photo_01 → Bob should all be reachable
    assert!(reachable.contains(&0));
    assert!(reachable.contains(&1));
    assert!(reachable.contains(&2));
}

// ── DeltaStore WAL roundtrip with image_url ───────────────────────────────────

#[tokio::test]
async fn delta_wal_roundtrip_preserves_image_url() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let delta = DeltaStore::new(data_dir.clone(), 100);

    let id = delta.alloc_node_id().unwrap();
    delta.add_node(image_node(id, "photo_01", "https://example.com/img.jpg"), vec![0.1f32; 8]);

    assert_eq!(delta.size(), 1);
    assert_eq!(
        delta.read().new_nodes[0].image_url.as_deref(),
        Some("https://example.com/img.jpg")
    );

    // Replay WAL from disk — image_url must survive.
    // DeltaStore::new() does not auto-replay; replay_wal() must be called explicitly
    // (mirrors the boot sequence in app_state.rs).
    let delta2 = DeltaStore::new(data_dir, 100);
    delta2.replay_wal();
    assert_eq!(delta2.size(), 1);
    assert_eq!(
        delta2.read().new_nodes[0].image_url.as_deref(),
        Some("https://example.com/img.jpg")
    );
}

// ── merge_into() preserves image_url ─────────────────────────────────────────

#[tokio::test]
async fn merge_preserves_image_url() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let storage = LocalStorage::new(data_dir.clone());

    let dim = 8usize;

    // Base: one text node — must persist the graph AND vectors.bin before merge_into
    let base_nodes = vec![text_node(0, "Alice")];
    let base = Arc::new(CsrGraph::build(base_nodes, &[]));
    save_graph(&base, &storage).unwrap();
    let base_vecs: Vec<Vec<f32>> = vec![{ let mut v = vec![0.1f32; dim]; normalise(&mut v); v }];
    VectorStore::write(&dir.path().join("vectors.bin"), dim, &base_vecs).unwrap();

    let delta = DeltaStore::new(data_dir, 100);
    let id = delta.alloc_node_id().unwrap();
    delta.add_node(image_node(id, "photo_01", "https://example.com/img.jpg"), vec![0.1f32; dim]);

    let (merged, vecs, _, _) = delta.merge_into(base, &storage).await.unwrap();
    assert_eq!(merged.num_nodes(), 2);
    assert_eq!(vecs.len(), 2);

    let img = merged.nodes.iter().find(|n| n.node_type == "Image").unwrap();
    assert_eq!(img.image_url.as_deref(), Some("https://example.com/img.jpg"));
    // text node untouched
    let txt = merged.nodes.iter().find(|n| n.name == "Alice").unwrap();
    assert!(txt.image_url.is_none());
}

// ── graph save/load roundtrip preserves image_url ────────────────────────────

#[test]
fn save_load_roundtrip_preserves_image_url() {
    let dir = tempdir().unwrap();
    let storage = LocalStorage::new(dir.path().to_path_buf());

    let nodes = vec![
        text_node(0, "Alice"),
        image_node(1, "photo_01", "https://example.com/img.jpg"),
    ];
    let edges = vec![
        EdgeInfo { from: 0, to: 1, edge_type: "has_photo".into(), weight: 0.9,
                   full_context: String::new(), embed_context: None, edge_id: 0 },
    ];
    let graph = CsrGraph::build(nodes, &edges);

    save_graph(&graph, &storage).unwrap();
    let loaded = load_graph(&storage).unwrap();

    let img = loaded.nodes.iter().find(|n| n.name == "photo_01").unwrap();
    assert_eq!(img.image_url.as_deref(), Some("https://example.com/img.jpg"));
    assert_eq!(img.node_type, "Image");

    let txt = loaded.nodes.iter().find(|n| n.name == "Alice").unwrap();
    assert!(txt.image_url.is_none());
}

// ── ExportNode serde with image_url ──────────────────────────────────────────

#[test]
fn export_node_serde_roundtrip_with_image() {
    use ai_graph_engine::api::dto::export::ExportNode;

    let en = ExportNode {
        name:          "photo_01".into(),
        node_type:     "Image".into(),
        props:         Value::Null,
        full_context:  String::new(),
        embed_context: None,
        image_url:     Some("https://example.com/img.jpg".into()),
    };
    let json = serde_json::to_string(&en).unwrap();
    let back: ExportNode = serde_json::from_str(&json).unwrap();
    assert_eq!(back.image_url.as_deref(), Some("https://example.com/img.jpg"));
}

#[test]
fn export_node_serde_omits_null_image_url() {
    use ai_graph_engine::api::dto::export::ExportNode;

    let en = ExportNode {
        name:          "Alice".into(),
        node_type:     "Entity".into(),
        props:         Value::Null,
        full_context:  String::new(),
        embed_context: None,
        image_url:     None,
    };
    let json = serde_json::to_string(&en).unwrap();
    // skip_serializing_if = "Option::is_none" → key should not appear
    assert!(!json.contains("image_url"));
}
