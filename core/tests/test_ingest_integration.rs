//! Integration tests for /ingest/text, /ingest/json, auth, metrics endpoints.
//! Uses the same mock plugin pattern as integration.rs.
//! cargo test ingest_integration -- --nocapture

use std::sync::Arc;
use axum::{routing::{get, post}, Json, Router};
use serde_json::Value;
use tempfile::tempdir;
use reqwest::Client;

use ai_graph_engine::{
    graph::builder::{from_json_payload, save as save_graph},
    storage::LocalStorage,
    vector::{hnsw::normalise, store::VectorStore},
};

// ── mock plugin ───────────────────────────────────────────────────────────────

async fn mock_health() -> Json<Value> { Json(serde_json::json!({"status":"ok"})) }

async fn mock_embed(Json(body): Json<Value>) -> Json<Value> {
    let n = body["texts"].as_array().map(|a| a.len()).unwrap_or(1);
    let v: Vec<f32> = (0..8).map(|i| if i == 0 { 1.0f32 } else { 0.0 }).collect();
    Json(serde_json::json!({ "vectors": vec![v; n] }))
}

async fn mock_extract(Json(_): Json<Value>) -> Json<Value> {
    Json(serde_json::json!({
        "entities": [
            {"id":"n1","name":"NewNode","type":"Concept","props":{}},
            {"id":"n2","name":"OtherNode","type":"Concept","props":{}}
        ],
        "relations": [
            {"from":"n1","to":"n2","type":"related_to","weight":1.0}
        ]
    }))
}

async fn mock_generate(Json(_): Json<Value>) -> Json<Value> {
    Json(serde_json::json!({"answer":"mock answer"}))
}

async fn start_mock(port: u16) {
    let app = Router::new()
        .route("/health",   get(mock_health))
        .route("/embed/text", post(mock_embed))
        .route("/extract",  post(mock_extract))
        .route("/generate", post(mock_generate));
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await.unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
}

// ── seed data ─────────────────────────────────────────────────────────────────

fn seed_data() -> (tempfile::TempDir, String) {
    let dir = tempdir().unwrap();
    let payload = serde_json::json!({
        "entities": [
            {"id":"e1","name":"Alice","type":"Person","props":{}},
            {"id":"e2","name":"Bob",  "type":"Person","props":{}},
        ],
        "relations": [{"from":"e1","to":"e2","type":"knows","weight":1.0}]
    });
    let g = from_json_payload(&payload).unwrap();
    let storage = LocalStorage::new(dir.path().to_path_buf());
    save_graph(&g, &storage).unwrap();

    let dim = 8usize;
    let vecs: Vec<Vec<f32>> = (0..2).map(|i| {
        let mut v = vec![0.0f32; dim]; v[i] = 1.0; normalise(&mut v); v
    }).collect();
    VectorStore::write(&dir.path().join("vectors.bin"), dim, &vecs).unwrap();

    let data_dir = dir.path().to_string_lossy().to_string();
    (dir, data_dir)
}

// We can't easily start the full Rust axum server in-process for integration tests
// without circular dependencies, so these tests hit the mock plugin directly
// and test the delta/metrics/auth logic through unit-level helpers.

#[tokio::test]
async fn mock_plugin_extract_returns_entities() {
    start_mock(19001).await;
    let client = Client::new();
    let resp = client
        .post("http://127.0.0.1:19001/extract")
        .json(&serde_json::json!({"text": "Alice works at Acme."}))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let body: Value = resp.json().await.unwrap();
    assert!(body["entities"].as_array().unwrap().len() > 0);
}

#[tokio::test]
async fn mock_plugin_embed_batch() {
    start_mock(19002).await;
    let client = Client::new();
    let resp = client
        .post("http://127.0.0.1:19002/embed/text")
        .json(&serde_json::json!({"texts": ["hello","world","test"]}))
        .send().await.unwrap();
    assert!(resp.status().is_success());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["vectors"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn delta_store_ingest_and_merge_flow() {
    use ai_graph_engine::delta::DeltaStore;
    use ai_graph_engine::graph::csr::NodeInfo;

    let (_dir, data_path) = seed_data();
    let data_dir = std::path::PathBuf::from(&data_path);

    let delta = DeltaStore::new(data_dir.clone(), 10);
    assert_eq!(delta.size(), 0);

    // simulate what /ingest/text handler does after extract+embed
    let base = from_json_payload(&serde_json::json!({
        "entities": [
            {"id":"e1","name":"Alice","type":"Person","props":{}},
            {"id":"e2","name":"Bob",  "type":"Person","props":{}},
        ],
        "relations": [{"from":"e1","to":"e2","type":"knows","weight":1.0}]
    })).unwrap();
    let base = Arc::new(base);

    delta.add_node(NodeInfo { id: 2, name: "NewNode".into(), node_type: "Concept".into(), weight: 0.0, props: Value::Null, full_context: String::new(), embed_context: None, external_id: "Concept:NewNode".into(), image_url: None },
        { let mut v = vec![0.0f32; 8]; v[2] = 1.0; normalise(&mut v); v });

    assert_eq!(delta.size(), 1);
    assert!(!delta.needs_merge()); // threshold = 10

    let storage = LocalStorage::new(data_dir.clone());
    let (merged, vecs, _, _) = delta.merge_into(base, &storage).await.unwrap();
    assert_eq!(merged.num_nodes(), 3);
    assert_eq!(vecs.len(), 3);
    assert_eq!(delta.size(), 0); // drained after merge
}

#[test]
fn metrics_track_query_lifecycle() {
    use ai_graph_engine::metrics::Metrics;

    let m = Metrics::new();
    m.queries_total.inc();
    m.embed_latency.observe(35);
    m.llm_latency.observe(800);
    m.query_latency.observe(840);
    m.cache_hits.add(3);
    m.cache_misses.add(1);

    let json = m.summary_json();
    assert_eq!(json["queries"]["total"].as_u64().unwrap(), 1);
    assert_eq!(json["cache"]["hits"].as_u64().unwrap(), 3);
    assert_eq!(json["cache"]["hit_rate"].as_str().unwrap(), "75.0%");

    let prom = m.render_prometheus();
    assert!(prom.contains("queries_total 1"));
    assert!(prom.contains("embed_latency_ms_count 1"));
}

#[test]
fn auth_end_to_end_key_validation() {
    use ai_graph_engine::middleware::auth::{ApiKeys, extract_key};
    use axum::http::{HeaderMap, HeaderValue};

    std::env::set_var("API_KEYS", "prod-key-abc,dev-key-xyz");
    let keys = ApiKeys::from_env();

    // valid keys
    assert!(keys.is_valid("prod-key-abc"));
    assert!(keys.is_valid("dev-key-xyz"));

    // invalid
    assert!(!keys.is_valid("hacker"));
    assert!(!keys.is_valid("prod-key-ab")); // one char short

    // header extraction
    let mut h = HeaderMap::new();
    h.insert("authorization", HeaderValue::from_static("Bearer prod-key-abc"));
    let extracted = extract_key(&h).unwrap();
    assert!(keys.is_valid(&extracted));
}
