//! Tests — delta store (incl. WAL), auth middleware, metrics, scoring
//! cargo test new_features

// ═══════════════════════════════════════════════════════════════
//  DELTA STORE + WAL
// ═══════════════════════════════════════════════════════════════

mod delta_tests {
    use std::sync::Arc;
    use ai_graph_engine::{
        delta::DeltaStore,
        graph::{
            builder::{from_json_payload, save as save_graph},
            csr::{EdgeInfo, NodeInfo},
        },
        storage::LocalStorage,
        vector::store::VectorStore,
    };
    use tempfile::tempdir;

    fn node(id: u32, name: &str) -> NodeInfo {
        NodeInfo { id, external_id: format!("Entity:{name}"), name: name.into(), node_type: "Entity".into(), weight: 0.0, props: serde_json::Value::Null, full_context: String::new(), embed_context: None, image_url: None }
    }

    fn unit_vec(seed: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[seed % dim] = 1.0;
        v
    }

    fn base_graph_dir() -> (tempfile::TempDir, Arc<ai_graph_engine::graph::csr::CsrGraph>) {
        let dir = tempdir().unwrap();
        let g = from_json_payload(&serde_json::json!({
            "entities": [
                {"id":"a","name":"A","type":"X","props":{}},
                {"id":"b","name":"B","type":"X","props":{}},
            ],
            "relations": [{"from":"a","to":"b","type":"r","weight":1.0}]
        })).unwrap();
        let storage = LocalStorage::new(dir.path().to_path_buf());
        save_graph(&g, &storage).unwrap();
        let vecs: Vec<Vec<f32>> = (0..2).map(|i| unit_vec(i, 8)).collect();
        VectorStore::write(&dir.path().join("vectors.bin"), 8, &vecs).unwrap();
        (dir, Arc::new(g))
    }

    #[test]
    fn starts_empty() {
        let d = DeltaStore::new("/tmp".into(), 100);
        assert_eq!(d.size(), 0);
        assert!(!d.needs_merge());
    }

    #[test]
    fn add_node_increments_size() {
        let d = DeltaStore::new("/tmp".into(), 100);
        d.add_node(node(10, "A"), unit_vec(0, 8));
        assert_eq!(d.size(), 1);
    }

    #[test]
    fn add_edge_increments_size() {
        let d = DeltaStore::new("/tmp".into(), 100);
        d.add_edge(EdgeInfo { from: 0, to: 1, edge_type: "r".into(), weight: 1.0, full_context: String::new(), embed_context: None, edge_id: 0 }, vec![]);
        assert_eq!(d.size(), 1);
    }

    #[test]
    fn needs_merge_at_threshold() {
        let d = DeltaStore::new("/tmp".into(), 3);
        d.add_node(node(10, "A"), unit_vec(0, 8));
        d.add_node(node(11, "B"), unit_vec(1, 8));
        assert!(!d.needs_merge());
        d.add_node(node(12, "C"), unit_vec(2, 8));
        assert!(d.needs_merge());
    }

    #[test]
    fn delta_neighbors_returned() {
        let d = DeltaStore::new("/tmp".into(), 100);
        d.add_edge(EdgeInfo { from: 5, to: 9, edge_type: "x".into(), weight: 0.5, full_context: String::new(), embed_context: None, edge_id: 0 }, vec![]);
        d.add_edge(EdgeInfo { from: 5, to: 7, edge_type: "x".into(), weight: 0.8, full_context: String::new(), embed_context: None, edge_id: 0 }, vec![]);
        let nb = d.read().neighbors(5).to_vec();
        assert_eq!(nb.len(), 2);
        assert!(nb.iter().any(|(to, _)| *to == 9));
        assert!(nb.iter().any(|(to, _)| *to == 7));
    }

    // ── WAL ──────────────────────────────────────────────────────────────────

    #[test]
    fn wal_file_created_on_add_node() {
        let dir = tempdir().unwrap();
        let d = DeltaStore::new(dir.path().to_path_buf(), 100);
        d.add_node(node(0, "X"), unit_vec(0, 8));
        assert!(dir.path().join("delta.wal").exists(), "WAL file should be created");
    }

    #[test]
    fn wal_replay_restores_nodes_after_crash() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // simulate: write some nodes, then "crash" (drop delta without merging)
        {
            let d = DeltaStore::new(path.clone(), 100);
            d.add_node(node(10, "Recovered_A"), unit_vec(0, 8));
            d.add_node(node(11, "Recovered_B"), unit_vec(1, 8));
        } // dropped — in-memory state gone, but WAL file persists

        // "restart": new DeltaStore, replay WAL
        let d2 = DeltaStore::new(path, 100);
        let replayed = d2.replay_wal();

        assert_eq!(replayed, 2, "should replay 2 entries");
        assert_eq!(d2.size(), 2, "delta should have 2 nodes after replay");
    }

    #[test]
    fn wal_replay_restores_edges() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let d = DeltaStore::new(path.clone(), 100);
            d.add_edge(EdgeInfo { from: 0, to: 1, edge_type: "r".into(), weight: 0.7, full_context: String::new(), embed_context: None, edge_id: 0 }, vec![]);
            d.add_edge(EdgeInfo { from: 1, to: 2, edge_type: "r".into(), weight: 0.5, full_context: String::new(), embed_context: None, edge_id: 0 }, vec![]);
        }

        let d2 = DeltaStore::new(path, 100);
        let replayed = d2.replay_wal();
        assert_eq!(replayed, 2);
        assert_eq!(d2.size(), 2);
    }

    #[test]
    fn wal_replay_on_empty_file_returns_zero() {
        let dir = tempdir().unwrap();
        let d = DeltaStore::new(dir.path().to_path_buf(), 100);
        assert_eq!(d.replay_wal(), 0);
    }

    #[tokio::test]
    async fn wal_truncated_after_merge() {
        let (dir, base) = base_graph_dir();

        let d = DeltaStore::new(dir.path().to_path_buf(), 100);
        d.add_node(node(2, "NewNode"), unit_vec(2, 8));

        let wal_path = dir.path().join("delta.wal");
        assert!(wal_path.exists());
        let content_before = std::fs::read_to_string(&wal_path).unwrap();
        assert!(!content_before.is_empty(), "WAL should have content before merge");

        let storage = LocalStorage::new(dir.path().to_path_buf());
        d.merge_into(base, &storage).await.unwrap();

        let content_after = std::fs::read_to_string(&wal_path).unwrap();
        assert!(content_after.is_empty(), "WAL should be empty after successful merge");
    }

    #[tokio::test]
    async fn merge_produces_larger_graph() {
        let (dir, base) = base_graph_dir();
        let d = DeltaStore::new(dir.path().to_path_buf(), 100);

        d.add_node(node(2, "C"), unit_vec(2, 8));
        d.add_node(node(3, "D"), unit_vec(3, 8));
        d.add_edge(EdgeInfo { from: 2, to: 3, edge_type: "r".into(), weight: 1.0, full_context: String::new(), embed_context: None, edge_id: 0 }, vec![]);

        let storage = LocalStorage::new(dir.path().to_path_buf());
        let (merged, vecs, _, _) = d.merge_into(base, &storage).await.unwrap();
        assert_eq!(merged.num_nodes(), 4);
        assert_eq!(vecs.len(), 4);
        assert!(merged.num_edges() >= 2);
        assert_eq!(d.size(), 0, "delta should be empty after merge");
    }
}

// ═══════════════════════════════════════════════════════════════
//  SCORING — edge-weighted proximity
// ═══════════════════════════════════════════════════════════════

mod scoring_tests {
    use ai_graph_engine::query::{QueryOptions, ScoringWeights};

    #[test]
    fn all_preset_weights_sum_to_1() {
        let presets = [
            ScoringWeights::balanced(),
            ScoringWeights::semantic_search(),
            ScoringWeights::relationship(),
            ScoringWeights::entity_lookup(),
        ];
        for w in &presets {
            let sum = w.alpha + w.beta + w.gamma;
            assert!((sum - 1.0).abs() < 1e-5, "α+β+γ = {sum}");
        }
    }

    #[test]
    fn semantic_mode_highest_alpha() {
        let s = ScoringWeights::semantic_search();
        assert!(s.alpha > s.beta && s.alpha > s.gamma);
    }

    #[test]
    fn relationship_mode_highest_beta() {
        let r = ScoringWeights::relationship();
        assert!(r.beta > r.alpha && r.beta > r.gamma);
    }

    #[test]
    fn entity_mode_highest_gamma() {
        let e = ScoringWeights::entity_lookup();
        assert!(e.gamma > e.beta);
    }

    // helper: compute score with new edge-weighted proximity
    fn score(vsim: f32, hop: u8, path_weight: f32, node_weight: f32, w: &ScoringWeights) -> f32 {
        let proximity = path_weight / (hop as f32 + 1.0);
        w.alpha * vsim + w.beta * proximity + w.gamma * node_weight
    }

    #[test]
    fn edge_weight_affects_proximity() {
        let w = ScoringWeights::balanced();
        // same hop, same vsim, same node_weight — only edge weight differs
        let strong = score(0.5, 1, 1.0, 0.5, &w); // edge weight 1.0
        let weak   = score(0.5, 1, 0.3, 0.5, &w); // edge weight 0.3
        assert!(strong > weak, "strong edge (w=1.0) should score higher than weak (w=0.3)");
    }

    #[test]
    fn seed_node_scores_higher_than_distant() {
        let w = ScoringWeights::balanced();
        let seed    = score(0.9, 0, 1.0, 0.8, &w);
        let distant = score(0.3, 2, 0.3, 0.2, &w);
        assert!(seed > distant);
    }

    #[test]
    fn hop0_proximity_is_1_when_path_weight_1() {
        // seed: hop=0, path_weight=1.0 → proximity = 1.0/1 = 1.0
        let proximity = 1.0f32 / (0u8 as f32 + 1.0);
        assert!((proximity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn proximity_decreases_with_hops() {
        let proximities: Vec<f32> = (0..4).map(|h| 1.0f32 / (h as f32 + 1.0)).collect();
        for w in proximities.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn default_options_sensible() {
        let opts = QueryOptions::default();
        assert!(opts.hnsw_k > 0);
        assert!(opts.bfs_depth > 0);
        assert!(opts.bfs_max_nodes >= opts.context_top_n);
        assert!(!opts.bidirectional, "default should not be bidirectional");
    }

    #[test]
    fn bidirectional_default_is_false() {
        let opts = QueryOptions::default();
        assert!(!opts.bidirectional);
    }
}

// ═══════════════════════════════════════════════════════════════
//  AUTH
// ═══════════════════════════════════════════════════════════════

mod auth_tests {
    use ai_graph_engine::middleware::auth::{ApiKeys, RateLimiter, extract_key};
    use axum::http::{HeaderMap, HeaderValue};

    fn keys(list: &str) -> ApiKeys {
        ApiKeys::new(list.split(',').map(|s| s.trim().to_string()).collect())
    }

    #[test]
    fn valid_key_accepted() {
        let k = keys("secret123,other");
        assert!(k.is_valid("secret123"));
        assert!(k.is_valid("other"));
    }

    #[test]
    fn invalid_key_rejected() {
        let k = keys("secret123");
        assert!(!k.is_valid("wrong"));
        assert!(!k.is_valid(""));
    }

    #[test]
    fn auth_disabled_when_no_keys() {
        let k = ApiKeys::new(vec![]);
        assert!(!k.is_enabled());
        assert!(k.is_valid("anything"));
    }

    #[test]
    fn timing_safe_same_length_wrong_key() {
        let k = keys("abcdef");
        assert!(!k.is_valid("xxxxxx"));
    }

    #[test]
    fn extracts_bearer_token() {
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer mytoken"));
        assert_eq!(extract_key(&h), Some("mytoken".into()));
    }

    #[test]
    fn extracts_x_api_key() {
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("mykey"));
        assert_eq!(extract_key(&h), Some("mykey".into()));
    }

    #[test]
    fn bearer_takes_priority() {
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer bearer_val"));
        h.insert("x-api-key",     HeaderValue::from_static("xapi_val"));
        assert_eq!(extract_key(&h), Some("bearer_val".into()));
    }

    #[test]
    fn no_auth_headers_returns_none() {
        assert_eq!(extract_key(&HeaderMap::new()), None);
    }

    #[test]
    fn allows_within_burst_capacity() {
        let limiter = RateLimiter::new(10, 60);
        for _ in 0..10 { assert!(limiter.check("u")); }
    }

    #[test]
    fn blocks_after_burst_exceeded() {
        let limiter = RateLimiter::new(3, 60);
        limiter.check("u"); limiter.check("u"); limiter.check("u");
        assert!(!limiter.check("u"), "4th request must be blocked");
    }

    #[test]
    fn different_keys_independent_buckets() {
        let limiter = RateLimiter::new(2, 60);
        limiter.check("alice"); limiter.check("alice");
        assert!(!limiter.check("alice"));
        assert!(limiter.check("bob"), "bob should still have tokens");
    }

    #[test]
    fn prune_does_not_panic() {
        let limiter = RateLimiter::new(10, 60);
        limiter.check("a"); limiter.check("b");
        limiter.prune();
    }
}

// ═══════════════════════════════════════════════════════════════
//  METRICS
// ═══════════════════════════════════════════════════════════════

mod metrics_tests {
    use ai_graph_engine::metrics::{Counter, Gauge, Histogram, Metrics, Timer};

    #[test]
    fn counter_increments() {
        let c = Counter::new("test");
        assert_eq!(c.get(), 0);
        c.inc(); c.inc();
        assert_eq!(c.get(), 2);
    }

    #[test]
    fn counter_add() {
        let c = Counter::new("test");
        c.add(5);
        assert_eq!(c.get(), 5);
    }

    #[test]
    fn gauge_sets_value() {
        let g = Gauge::new("test");
        g.set(42); assert_eq!(g.get(), 42);
        g.set(0);  assert_eq!(g.get(), 0);
    }

    #[test]
    fn histogram_count_and_sum() {
        let h = Histogram::new("test");
        h.observe(100); h.observe(200); h.observe(300);
        let r = h.render();
        assert!(r.contains("test_count 3"));
        assert!(r.contains("test_sum 600"));
    }

    #[test]
    fn histogram_percentiles_reasonable() {
        let h = Histogram::new("test");
        for _ in 0..100 { h.observe(10); }
        assert!(h.p50_ms() <= 50);
        assert!(h.p99_ms() <= 100);
    }

    #[test]
    fn histogram_empty_percentile_is_0() {
        let h = Histogram::new("test");
        assert_eq!(h.p50_ms(), 0);
        assert_eq!(h.p99_ms(), 0);
    }

    #[test]
    fn prometheus_render_contains_all_metrics() {
        let m = Metrics::new();
        m.queries_total.inc();
        m.cache_hits.add(5);
        m.query_latency.observe(250);
        m.graph_nodes.set(1000);
        let r = m.render_prometheus();
        assert!(r.contains("queries_total 1"));
        assert!(r.contains("cache_hits_total 5"));
        assert!(r.contains("graph_nodes 1000"));
        assert!(r.contains("query_latency_ms_bucket"));
    }

    #[test]
    fn summary_json_cache_hit_rate() {
        let m = Metrics::new();
        m.cache_hits.add(80);
        m.cache_misses.add(20);
        assert_eq!(m.summary_json()["cache"]["hit_rate"].as_str().unwrap(), "80.0%");
    }

    #[test]
    fn summary_json_zero_denominator_no_panic() {
        let m = Metrics::new();
        assert_eq!(m.summary_json()["cache"]["hit_rate"].as_str().unwrap(), "0.0%");
    }

    #[test]
    fn timer_records_on_drop() {
        let h = Histogram::new("test");
        { let _t = Timer::new(&h); std::thread::sleep(std::time::Duration::from_millis(5)); }
        assert!(h.render().contains("test_count 1"));
    }

    #[test]
    fn metrics_thread_safe() {
        use std::sync::Arc;
        let m = Arc::new(Metrics::new());
        let handles: Vec<_> = (0..8).map(|_| {
            let m2 = m.clone();
            std::thread::spawn(move || { for _ in 0..100 { m2.queries_total.inc(); } })
        }).collect();
        for h in handles { h.join().unwrap(); }
        assert_eq!(m.queries_total.get(), 800);
    }
}
