//! FAISS-style recall@k benchmark.
//!
//! Measures both search speed AND accuracy of HNSW at different ef_construction
//! values. Recall@k = fraction of true top-k neighbours returned by HNSW.
//!
//! cargo bench --bench recall
//!
//! During setup (before timing), recall is printed to stderr:
//!
//!   [recall@10] M=16 ef_construction= 50  recall=87.4%  (n=10k, 200 queries)
//!   [recall@10] M=16 ef_construction=100  recall=94.1%  (n=10k, 200 queries)
//!   [recall@10] M=16 ef_construction=200  recall=97.8%  (n=10k, 200 queries)
//!   [recall@10] M=16 ef_construction=400  recall=99.1%  (n=10k, 200 queries)

use std::collections::HashSet;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::tempdir;
use ai_graph_engine::vector::{
    hnsw::{HnswIndex, normalise},
    store::{VectorStore, brute_search},
};

const DIM: usize = 384;

fn make_vecs(n: usize, offset: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let idx = i + offset;
            let mut v: Vec<f32> = (0..DIM)
                .map(|j| ((idx * DIM + j) as f32 * 0.0001).sin())
                .collect();
            normalise(&mut v);
            v
        })
        .collect()
}

fn write_store(vecs: &[Vec<f32>]) -> (tempfile::TempDir, VectorStore) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    VectorStore::write(&path, DIM, vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();
    (dir, store)
}

/// Recall@k = mean fraction of true top-k neighbours found by HNSW.
fn recall_at_k(index: &HnswIndex, store: &VectorStore, queries: &[Vec<f32>], k: usize) -> f64 {
    let total: f64 = queries.iter().map(|q| {
        let gt: HashSet<u32> = brute_search(store, q, k).into_iter().map(|(id, _)| id).collect();
        let approx: HashSet<u32> = index.search(q, k).into_iter().map(|(id, _)| id).collect();
        gt.intersection(&approx).count() as f64 / k as f64
    }).sum();
    total / queries.len() as f64
}

// ── recall@k vs ef_construction ───────────────────────────────────────────────

fn bench_recall_ef_tradeoff(c: &mut Criterion) {
    let n_train = 10_000;
    let n_query = 200;

    // train and query sets are disjoint (different offset avoids data leakage)
    let train   = make_vecs(n_train, 0);
    let queries = make_vecs(n_query, n_train);
    let ids: Vec<u32> = (0..n_train as u32).collect();
    let (_dir, store) = write_store(&train);

    let mut group = c.benchmark_group("recall_at_10");
    group.throughput(Throughput::Elements(1));

    for &ef in &[50usize, 100, 200, 400] {
        let index = HnswIndex::build(train.clone(), ids.clone(), 16, ef).unwrap();

        // recall computed once before Criterion timing loop — printed to stderr
        let recall = recall_at_k(&index, &store, &queries, 10);
        eprintln!(
            "[recall@10] M=16  ef_construction={ef:>3}  recall={:.1}%  (n=10k, {} queries)",
            recall * 100.0, n_query,
        );

        group.bench_with_input(
            BenchmarkId::new("M16_ef", ef),
            &(&index, &queries[0]),
            |b, (index, query)| {
                b.iter(|| black_box(index.search(query, 10)))
            },
        );
    }
    group.finish();
}

// ── recall@k vs corpus size ───────────────────────────────────────────────────

fn bench_recall_corpus_scale(c: &mut Criterion) {
    let n_query = 200;
    let ef = 200;

    let mut group = c.benchmark_group("recall_corpus_scale");
    group.throughput(Throughput::Elements(1));

    for &(n, label) in &[(1_000usize, "1k"), (10_000, "10k"), (100_000, "100k")] {
        let train   = make_vecs(n, 0);
        let queries = make_vecs(n_query, n);
        let ids: Vec<u32> = (0..n as u32).collect();
        let (_dir, store) = write_store(&train);
        let index = HnswIndex::build(train, ids, 16, ef).unwrap();

        let recall = recall_at_k(&index, &store, &queries, 10);
        eprintln!(
            "[recall@10] M=16  ef_construction={ef}  n={label:<6}  recall={:.1}%",
            recall * 100.0,
        );

        group.bench_with_input(
            BenchmarkId::new("M16_ef200", label),
            &(&index, &queries[0]),
            |b, (index, query)| {
                b.iter(|| black_box(index.search(query, 10)))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_recall_ef_tradeoff, bench_recall_corpus_scale);
criterion_main!(benches);
