//! Integration tests — full query pipeline with a mock Python plugin.
//!
//! Spins up a tiny axum mock server that mimics /embed /extract /generate,
//! then runs the complete QueryEngine pipeline end-to-end without needing
//! a real Python process or Anthropic API key.
//!
//! cargo test integration -- --nocapture

use std::sync::Arc;
use tempfile::tempdir;

use ai_graph_engine::{
    cache::EmbedCache,
    config::PluginsConfig,
    graph::builder::{from_json_payload, save as save_graph},
    plugin::PluginClient,
    query::{QueryEngine, QueryOptions, ScoringWeights},
    storage::LocalStorage,
    vector::{
        hnsw::{EdgeHnswIndex, HnswIndex, normalise},
        store::VectorStore,
    },
};

// ── mock plugin server ────────────────────────────────────────────────────────

use axum::{routing::post, Json, Router};
use serde_json::Value;

async fn mock_embed(Json(body): Json<Value>) -> Json<Value> {
    let n = body["texts"].as_array().map(|a| a.len()).unwrap_or(1);
    let vec: Vec<f32> = (0..8).map(|i| if i == 0 { 1.0f32 } else { 0.0 }).collect();
    Json(serde_json::json!({ "vectors": vec![vec; n], "dim": 8 }))
}

async fn mock_extract(Json(_): Json<Value>) -> Json<Value> {
    Json(serde_json::json!({
        "entities": [{"id":"x1","name":"MockEntity","type":"Concept","props":{}}],
        "relations": []
    }))
}

async fn mock_generate(Json(_): Json<Value>) -> Json<Value> {
    Json(serde_json::json!({ "answer": "Mock answer from integration test." }))
}

async fn mock_health() -> Json<Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn start_mock_server() -> String {
    use axum::routing::get;
    let app = Router::new()
        .route("/health",   get(mock_health))
        .route("/embed/text", post(mock_embed))
        .route("/extract",  post(mock_extract))
        .route("/generate", post(mock_generate));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    format!("http://127.0.0.1:{}", addr.port())
}

// ── test data ─────────────────────────────────────────────────────────────────

fn sample_payload() -> Value {
    serde_json::json!({
        "entities": [
            {"id":"e1","name":"Alice",       "type":"Person", "props":{"role":"CEO"}},
            {"id":"e2","name":"Bob",         "type":"Person", "props":{"role":"CTO"}},
            {"id":"e3","name":"Acme Corp",   "type":"Company","props":{}},
            {"id":"e4","name":"GraphEngine", "type":"Product","props":{}},
            {"id":"e5","name":"Rust",        "type":"Tech",   "props":{}},
        ],
        "relations": [
            {"from":"e1","to":"e3","type":"leads",    "weight":1.0},
            {"from":"e2","to":"e3","type":"works_at", "weight":0.8},
            {"from":"e3","to":"e4","type":"builds",   "weight":1.0},
            {"from":"e4","to":"e5","type":"uses",     "weight":0.9},
        ]
    })
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn build_test_engine(plugin_url: &str) -> Arc<QueryEngine> {
    let dir = tempdir().unwrap();

    let graph = Arc::new(from_json_payload(&sample_payload()).unwrap());
    let dim = 8usize;
    let vecs: Vec<Vec<f32>> = (0..graph.num_nodes())
        .map(|i| { let mut v = vec![0.0f32; dim]; v[i % dim] = 1.0; normalise(&mut v); v })
        .collect();

    let storage = LocalStorage::new(dir.path().to_path_buf());
    save_graph(&graph, &storage).unwrap();
    VectorStore::write(&dir.path().join("vectors.bin"), dim, &vecs).unwrap();

    let data_dir = dir.keep(); // keep alive

    let store_inner = Arc::new(VectorStore::open(&data_dir.join("vectors.bin")).unwrap());
    let ids: Vec<u32> = (0..store_inner.num_vecs as u32).collect();
    let hnsw_vecs: Vec<Vec<f32>> = (0..store_inner.num_vecs as u32).map(|id| store_inner.get(id).to_vec()).collect();
    let store = Arc::new(tokio::sync::RwLock::new(store_inner));

    // QueryEngine now takes Arc<RwLock<Arc<_>>> for hot-swappable graph/hnsw/edge_hnsw
    let graph_rw     = Arc::new(tokio::sync::RwLock::new(graph));
    let hnsw_rw      = Arc::new(tokio::sync::RwLock::new(
        Arc::new(HnswIndex::build(hnsw_vecs, ids, 8, 50).unwrap())
    ));
    let edge_hnsw_rw = Arc::new(tokio::sync::RwLock::new(
        Arc::new(EdgeHnswIndex::build(vec![], vec![], 8, 50).unwrap())
    ));

    Arc::new(QueryEngine::new(
        graph_rw,
        hnsw_rw,
        store,
        edge_hnsw_rw,
        Arc::new(EmbedCache::new(1000)),
        Arc::new(PluginClient::new(&PluginsConfig::from_single_url(plugin_url)).unwrap()),
    ))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn full_pipeline_returns_answer() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    let result = engine.query("Who works at Acme Corp?", QueryOptions::default()).await.unwrap();
    assert!(!result.answer.is_empty());
    assert!(!result.subgraph.nodes.is_empty());
}

#[tokio::test]
async fn pipeline_stats_are_populated() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    let s = engine.query("test", QueryOptions::default()).await.unwrap().stats;
    assert!(s.seed_nodes > 0);
    assert!(s.subgraph_nodes > 0);
    assert!(s.context_nodes > 0);
    assert!(!s.cache_hit, "first query must not be a cache hit");
}

#[tokio::test]
async fn second_identical_query_is_cache_hit() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    let opts = QueryOptions::default();

    let r1 = engine.query("same question", opts.clone()).await.unwrap();
    let r2 = engine.query("same question", opts).await.unwrap();

    assert!(!r1.stats.cache_hit, "first call should not be a cache hit");
    assert!( r2.stats.cache_hit, "second identical call should be a cache hit");
    assert_eq!(r1.answer, r2.answer, "cached answer must match original");
}

#[tokio::test]
async fn different_query_not_cache_hit() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    engine.query("question A", QueryOptions::default()).await.unwrap();
    let r = engine.query("question B", QueryOptions::default()).await.unwrap();
    assert!(!r.stats.cache_hit);
}

#[tokio::test]
async fn context_nodes_sorted_by_score() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    let result = engine.query("test", QueryOptions::default()).await.unwrap();
    for w in result.subgraph.nodes.windows(2) {
        assert!(w[0].score >= w[1].score, "not sorted: {} < {}", w[0].score, w[1].score);
    }
}

#[tokio::test]
async fn context_capped_at_top_n() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    let opts = QueryOptions { context_top_n: 2, ..QueryOptions::default() };
    let result = engine.query("test", opts).await.unwrap();
    assert!(result.subgraph.nodes.len() <= 2);
}

#[tokio::test]
async fn deeper_bfs_finds_more_nodes() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;

    let shallow = engine.query("test", QueryOptions { bfs_depth: 1, ..QueryOptions::default() }).await.unwrap();
    let deep    = engine.query("deep test", QueryOptions { bfs_depth: 3, ..QueryOptions::default() }).await.unwrap();

    assert!(
        deep.stats.subgraph_nodes >= shallow.stats.subgraph_nodes,
        "deep={} < shallow={}", deep.stats.subgraph_nodes, shallow.stats.subgraph_nodes
    );
}

#[tokio::test]
async fn relationship_mode_uses_bidirectional() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;
    let opts = QueryOptions {
        weights: ScoringWeights::relationship(),
        bidirectional: true,
        ..QueryOptions::default()
    };
    let result = engine.query("test relationship", opts).await.unwrap();
    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn cache_invalidation_clears_results() {
    let url = start_mock_server().await;
    let engine = build_test_engine(&url).await;

    engine.query("cached query", QueryOptions::default()).await.unwrap();
    engine.invalidate_query_cache();

    // after invalidation, same query must NOT be a cache hit
    let r = engine.query("cached query", QueryOptions::default()).await.unwrap();
    assert!(!r.stats.cache_hit, "query should miss after cache invalidation");
}

#[tokio::test]
async fn plugin_health_check_ok() {
    let url = start_mock_server().await;
    assert!(PluginClient::new(&PluginsConfig::from_single_url(&url)).unwrap().health().await);
}

#[tokio::test]
async fn plugin_embed_returns_correct_dim() {
    let url = start_mock_server().await;
    let vecs = PluginClient::new(&PluginsConfig::from_single_url(&url)).unwrap().embed(vec!["hello".into(), "world".into()]).await.unwrap();
    assert_eq!(vecs.len(), 2);
    assert_eq!(vecs[0].len(), 8);
}

#[tokio::test]
async fn plugin_extract_returns_entities() {
    let url = start_mock_server().await;
    let resp = PluginClient::new(&PluginsConfig::from_single_url(&url)).unwrap().extract("Alice works at Acme.", None).await.unwrap();
    assert!(!resp.entities.is_empty());
}

#[tokio::test]
async fn plugin_generate_returns_string() {
    let url = start_mock_server().await;
    let answer = PluginClient::new(&PluginsConfig::from_single_url(&url)).unwrap().generate(&[], &[], "test question", None).await.unwrap();
    assert!(!answer.is_empty());
}
