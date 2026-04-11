//! Criterion benchmarks — vector search
//! cargo bench --bench vector_search
//!
//! Measures:
//!   - cosine_sim on single pair
//!   - brute-force top-k at 1k / 10k / 100k vectors
//!   - HNSW search at 1k / 10k / 100k vectors
//!   - HNSW build time
//!   - VectorStore mmap read latency (cold vs warm)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::tempdir;

use rayon::prelude::*;
use ai_graph_engine::vector::{
    hnsw::{HnswIndex, normalise},
    store::{VectorStore, brute_search, cosine_sim},
};

// ── helpers ───────────────────────────────────────────────────────────────────

const DIM: usize = 384; // all-MiniLM-L6-v2 output size

fn make_vecs(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM)
                .map(|j| ((i * DIM + j) as f32 * 0.0001).sin())
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

// ── benchmarks ────────────────────────────────────────────────────────────────

fn bench_cosine_sim(c: &mut Criterion) {
    let a = make_vecs(1).pop().unwrap();
    let b = make_vecs(1).pop().unwrap();

    c.benchmark_group("cosine_sim")
        .throughput(Throughput::Elements(1))
        .bench_function(format!("dim{DIM}"), |b_| {
            b_.iter(|| black_box(cosine_sim(black_box(&a), black_box(&b))))
        });
}

fn bench_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("brute_force_search");

    for &(size, label) in &[(1_000usize, "1k"), (10_000, "10k"), (50_000, "50k")] {
        let vecs = make_vecs(size);
        let (_dir, store) = write_store(&vecs);
        let query = &vecs[0];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("top10", label),
            &(&store, query),
            |b, (store, query)| {
                b.iter(|| black_box(brute_search(store, query, 10)))
            },
        );
    }
    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    for &(size, label) in &[(1_000usize, "1k"), (10_000, "10k"), (100_000, "100k")] {
        let vecs = make_vecs(size);
        let ids: Vec<u32> = (0..size as u32).collect();
        let index = HnswIndex::build(vecs.clone(), ids, 16, 200).unwrap();
        let query = &vecs[0];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("top10", label),
            &(&index, query),
            |b, (index, query)| {
                b.iter(|| black_box(index.search(query, 10)))
            },
        );
    }
    group.finish();
}

fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    // building is slow — use fewer samples
    group.sample_size(10);

    for &(size, label) in &[(1_000usize, "1k"), (10_000, "10k")] {
        let vecs = make_vecs(size);
        let ids: Vec<u32> = (0..size as u32).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("M16_ef200", label),
            &(vecs.clone(), ids.clone()),
            |b, (vecs, ids)| {
                b.iter(|| {
                    black_box(HnswIndex::build(vecs.clone(), ids.clone(), 16, 200).unwrap())
                })
            },
        );
    }
    group.finish();
}

fn bench_mmap_read(c: &mut Criterion) {
    let size = 10_000usize;
    let vecs = make_vecs(size);
    let (_dir, store) = write_store(&vecs);

    let mut group = c.benchmark_group("mmap_read");
    group.throughput(Throughput::Bytes((DIM * 4) as u64));

    // sequential — cache-friendly
    group.bench_function("sequential_10k", |b| {
        b.iter(|| {
            for id in 0..100u32 {
                black_box(store.get(id));
            }
        })
    });

    // random access — tests mmap page fault behaviour
    group.bench_function("random_access_10k", |b| {
        b.iter(|| {
            for i in 0..100u32 {
                let id = (i * 97 + 13) % size as u32; // pseudo-random
                black_box(store.get(id));
            }
        })
    });

    group.finish();
}

fn bench_hnsw_vs_brute(c: &mut Criterion) {
    // Side-by-side on same dataset so output is directly comparable
    let size = 10_000usize;
    let vecs = make_vecs(size);
    let ids: Vec<u32> = (0..size as u32).collect();
    let index = HnswIndex::build(vecs.clone(), ids, 16, 200).unwrap();
    let (_dir, store) = write_store(&vecs);
    let query = &vecs[42];

    let mut group = c.benchmark_group("hnsw_vs_brute_10k");
    group.throughput(Throughput::Elements(1));

    group.bench_function("brute_top10", |b| {
        b.iter(|| black_box(brute_search(&store, query, 10)))
    });
    group.bench_function("hnsw_top10", |b| {
        b.iter(|| black_box(index.search(query, 10)))
    });

    group.finish();
}

fn bench_multi_query_throughput(c: &mut Criterion) {
    use rayon::prelude::*;

    let size = 10_000usize;
    let vecs = make_vecs(size);
    let ids: Vec<u32> = (0..size as u32).collect();
    let index = HnswIndex::build(vecs.clone(), ids, 16, 200).unwrap();
    // query set disjoint from train
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            let mut v = make_vecs(1).pop().unwrap();
            // shift slightly so queries differ
            v.iter_mut().enumerate().for_each(|(j, x)| *x += (i * DIM + j) as f32 * 1e-5);
            normalise(&mut v);
            v
        })
        .collect();

    let mut group = c.benchmark_group("multi_query_throughput");
    group.throughput(Throughput::Elements(queries.len() as u64));

    group.bench_function("100q_sequential", |b| {
        b.iter(|| {
            queries.iter()
                .map(|q| index.search(q, 10))
                .collect::<Vec<_>>()
        })
    });

    group.bench_function("100q_rayon_par", |b| {
        b.iter(|| {
            queries.par_iter()
                .map(|q| index.search(q, 10))
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_sim,
    bench_brute_force,
    bench_hnsw_search,
    bench_hnsw_build,
    bench_mmap_read,
    bench_hnsw_vs_brute,
    bench_multi_query_throughput,
);
criterion_main!(benches);
