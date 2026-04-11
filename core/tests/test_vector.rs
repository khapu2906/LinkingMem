//! Unit tests — VectorStore, HNSW, cosine similarity
//! cargo test vector

use ai_graph_engine::vector::{
    hnsw::{HnswIndex, normalise},
    store::{VectorStore, brute_search, cosine_sim},
};
use tempfile::tempdir;

// ── helpers ──────────────────────────────────────────────────────────────────

fn unit_vec(values: &[f32]) -> Vec<f32> {
    let mut v = values.to_vec();
    normalise(&mut v);
    v
}

/// Build n random-ish deterministic vectors of given dim
fn make_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.1).sin())
                .collect();
            normalise(&mut v);
            v
        })
        .collect()
}

// ── cosine_sim ────────────────────────────────────────────────────────────────

#[test]
fn cosine_identical_vectors_is_1() {
    let v = unit_vec(&[1.0, 2.0, 3.0]);
    let sim = cosine_sim(&v, &v);
    assert!((sim - 1.0).abs() < 1e-5, "sim={sim}");
}

#[test]
fn cosine_orthogonal_vectors_is_0() {
    let a = unit_vec(&[1.0, 0.0, 0.0]);
    let b = unit_vec(&[0.0, 1.0, 0.0]);
    let sim = cosine_sim(&a, &b);
    assert!(sim.abs() < 1e-5, "sim={sim}");
}

#[test]
fn cosine_opposite_vectors_is_neg1() {
    let a = unit_vec(&[1.0, 2.0, 3.0]);
    let b: Vec<f32> = a.iter().map(|x| -x).collect();
    let sim = cosine_sim(&a, &b);
    assert!((sim + 1.0).abs() < 1e-5, "sim={sim}");
}

// ── normalise ────────────────────────────────────────────────────────────────

#[test]
fn normalise_gives_unit_length() {
    let mut v = vec![3.0f32, 4.0];
    normalise(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn normalise_zero_vector_no_panic() {
    let mut v = vec![0.0f32, 0.0, 0.0];
    normalise(&mut v); // should not panic or produce NaN
    assert!(v.iter().all(|x| !x.is_nan()));
}

// ── VectorStore ───────────────────────────────────────────────────────────────

#[test]
fn write_and_read_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vectors.bin");
    let vecs = make_vecs(10, 8);

    VectorStore::write(&path, 8, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();

    assert_eq!(store.num_vecs, 10);
    assert_eq!(store.dim, 8);

    for (i, original) in vecs.iter().enumerate() {
        let loaded = store.get(i as u32);
        for (a, b) in original.iter().zip(loaded.iter()) {
            assert!((a - b).abs() < 1e-6, "vec {i} mismatch: {a} vs {b}");
        }
    }
}

#[test]
fn store_rejects_wrong_size_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.bin");
    std::fs::write(&path, b"short").unwrap();
    assert!(VectorStore::open(&path).is_err());
}

// ── brute_search ──────────────────────────────────────────────────────────────

#[test]
fn brute_search_top1_finds_identical_vec() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    let vecs = make_vecs(20, 16);
    VectorStore::write(&path, 16, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();

    // query = exact copy of vec #5
    let query = store.get(5).to_vec();
    let results = brute_search(&store, &query, 1);

    assert_eq!(results[0].0, 5, "top-1 should be the query itself");
    assert!((results[0].1 - 1.0).abs() < 1e-4, "similarity should be ~1.0");
}

#[test]
fn brute_search_returns_k_results() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    let vecs = make_vecs(50, 8);
    VectorStore::write(&path, 8, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();

    let query = store.get(0).to_vec();
    let results = brute_search(&store, &query, 10);
    assert_eq!(results.len(), 10);
}

#[test]
fn brute_search_results_descending_similarity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    let vecs = make_vecs(30, 8);
    VectorStore::write(&path, 8, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();

    let query = store.get(0).to_vec();
    let results = brute_search(&store, &query, 10);

    for w in results.windows(2) {
        assert!(w[0].1 >= w[1].1, "results not sorted by descending similarity");
    }
}

// ── HnswIndex ─────────────────────────────────────────────────────────────────

#[test]
fn hnsw_top1_finds_nearest() {
    let vecs = make_vecs(100, 32);
    let ids: Vec<u32> = (0..100).collect();
    let index = HnswIndex::build(vecs.clone(), ids, 16, 100).unwrap();

    // query = exact copy of vec #42
    let results = index.search(&vecs[42], 1);
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 42, "top-1 should be node 42 (exact match)");
}

#[test]
fn hnsw_returns_k_or_fewer_results() {
    let vecs = make_vecs(50, 16);
    let ids: Vec<u32> = (0..50).collect();
    let index = HnswIndex::build(vecs.clone(), ids, 8, 100).unwrap();

    let results = index.search(&vecs[0], 10);
    assert!(results.len() <= 10);
    assert!(!results.is_empty());
}

#[test]
fn hnsw_recall_above_80pct_vs_brute_force() {
    // build 200-vector index, query 10 times, measure recall vs brute force
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    let vecs = make_vecs(200, 32);
    let ids: Vec<u32> = (0..200).collect();

    VectorStore::write(&path, 32, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();
    let index = HnswIndex::build(vecs.clone(), ids, 16, 200).unwrap();

    let mut hits = 0usize;
    let k = 10;
    let queries = [0usize, 17, 42, 88, 123, 150, 199, 3, 66, 101];

    for &qi in &queries {
        let q = &vecs[qi];
        let brute: std::collections::HashSet<u32> =
            brute_search(&store, q, k).into_iter().map(|(id, _)| id).collect();
        let hnsw: std::collections::HashSet<u32> =
            index.search(q, k).into_iter().map(|(id, _)| id).collect();
        hits += brute.intersection(&hnsw).count();
    }

    let recall = hits as f64 / (queries.len() * k) as f64;
    assert!(recall >= 0.80, "HNSW recall too low: {:.1}%", recall * 100.0);
}
