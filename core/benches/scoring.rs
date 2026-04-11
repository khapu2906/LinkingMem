//! Criterion benchmarks — scoring pipeline + LRU cache
//! cargo bench --bench scoring

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ai_graph_engine::{
    cache::EmbedCache,
    graph::csr::{CsrGraph, EdgeInfo, NodeInfo},
    query::ScoringWeights,
    vector::store::{VectorStore, cosine_sim},
};
use tempfile::tempdir;

const DIM: usize = 384;

fn make_graph(n: usize) -> CsrGraph {
    let nodes: Vec<NodeInfo> = (0..n as u32)
        .map(|id| NodeInfo { id, name: format!("n{id}"), node_type: "E".into(), weight: 0.0, props: serde_json::Value::Null, full_context: String::new(), embed_context: None })
        .collect();
    let edges: Vec<EdgeInfo> = (0..n)
        .flat_map(|i| (1..=4usize).map(move |d| EdgeInfo {
            from: i as u32, to: ((i + d) % n) as u32, edge_type: String::new(), weight: 1.0, full_context: String::new(), embed_context: None,
        }))
        .collect();
    CsrGraph::build(nodes, &edges)
}

fn make_store(n: usize) -> (tempfile::TempDir, VectorStore) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM).map(|j| ((i * DIM + j) as f32 * 0.0001).sin()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect();
    VectorStore::write(&path, DIM, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();
    (dir, store)
}

// ── scoring formula ───────────────────────────────────────────────────────────

fn bench_scoring_formula(c: &mut Criterion) {
    let (_dir, store) = make_store(10_000);
    let query: Vec<f32> = store.get(0).to_vec();
    let w = ScoringWeights::balanced();

    // simulate scoring 500 nodes (typical subgraph after BFS depth=2)
    let subgraph: Vec<(u32, u8)> = (0..500u32).map(|i| (i, (i % 3) as u8)).collect();

    c.benchmark_group("scoring")
        .throughput(Throughput::Elements(500))
        .bench_function("score_500_nodes", |b| {
            b.iter(|| {
                let mut total = 0.0f32;
                for &(node_id, hop) in &subgraph {
                    let vec = store.get(node_id);
                    let vsim = cosine_sim(vec, &query);
                    let proximity = 1.0 / (hop as f32 + 1.0);
                    let node_weight = 0.5f32; // mock
                    let score = w.alpha * vsim + w.beta * proximity + w.gamma * node_weight;
                    total += black_box(score);
                }
                total
            })
        });
}

// ── LRU cache ────────────────────────────────────────────────────────────────

fn bench_cache_hot(c: &mut Criterion) {
    let (_dir, store) = make_store(1_000);
    // capacity large enough that everything stays hot
    let cache = EmbedCache::new(10_000);

    // warm up: load all 1k nodes
    for i in 0..1_000u32 { cache.get_or_load(i, &store); }

    c.benchmark_group("lru_cache")
        .throughput(Throughput::Elements(100))
        .bench_function("hot_path_100_hits", |b| {
            b.iter(|| {
                for i in 0..100u32 {
                    black_box(cache.get_or_load(i % 1_000, &store));
                }
            })
        });
}

fn bench_cache_cold(c: &mut Criterion) {
    let (_dir, store) = make_store(10_000);
    // tiny cache → forces constant eviction + mmap reads
    let cache = EmbedCache::new(10);

    c.benchmark_group("lru_cache")
        .throughput(Throughput::Elements(100))
        .bench_function("cold_path_100_misses", |b| {
            b.iter(|| {
                cache.invalidate_all();
                for i in 0..100u32 {
                    // stride > cache size → every access is a miss
                    black_box(cache.get_or_load((i * 101) % 10_000, &store));
                }
            })
        });
}

// ── BFS + score combined (realistic pipeline slice) ──────────────────────────

fn bench_bfs_then_score(c: &mut Criterion) {
    let graph = make_graph(10_000);
    let (_dir, store) = make_store(10_000);
    let cache = EmbedCache::new(5_000);
    let query: Vec<f32> = store.get(0).to_vec();
    let w = ScoringWeights::balanced();
    let seeds = vec![0u32, 2500, 5000, 7500];

    c.benchmark_group("pipeline_slice")
        .throughput(Throughput::Elements(1))
        .bench_function("bfs_depth2_then_score_10k", |b| {
            b.iter(|| {
                // BFS
                let subgraph = graph.bfs_expand(&seeds, 2, 500, false);
                // score
                let mut best = 0.0f32;
                for &(node_id, hop, _path_w) in &subgraph {
                    let vec = cache.get_or_load(node_id, &store);
                    let vsim = cosine_sim(&vec, &query);
                    let proximity = 1.0 / (hop as f32 + 1.0);
                    let score = w.alpha * vsim + w.beta * proximity + w.gamma * 0.5;
                    if score > best { best = score; }
                }
                black_box(best)
            })
        });
}

criterion_group!(
    benches,
    bench_scoring_formula,
    bench_cache_hot,
    bench_cache_cold,
    bench_bfs_then_score,
);
criterion_main!(benches);
