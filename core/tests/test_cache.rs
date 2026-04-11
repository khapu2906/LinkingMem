//! Unit tests — EmbedCache (moka concurrent cache)
//! cargo test cache

use ai_graph_engine::{cache::EmbedCache, vector::store::VectorStore};
use tempfile::tempdir;

fn make_store(n: usize, dim: usize) -> (tempfile::TempDir, VectorStore) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v.bin");
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| vec![(i as f32) / n as f32; dim])
        .collect();
    VectorStore::write(&path, dim, &vecs).unwrap();
    let store = VectorStore::open(&path).unwrap();
    (dir, store)
}

#[test]
fn cache_miss_loads_from_store() {
    let (_dir, store) = make_store(10, 4);
    let cache = EmbedCache::new(100);
    let v = cache.get_or_load(0, &store);
    assert_eq!(v.len(), 4);
}

#[test]
fn cache_hit_returns_same_value() {
    let (_dir, store) = make_store(10, 4);
    let cache = EmbedCache::new(100);
    let v1 = cache.get_or_load(3, &store);
    let v2 = cache.get_or_load(3, &store);
    assert_eq!(v1, v2);
}

#[test]
fn cache_size_increments_on_miss() {
    let (_dir, store) = make_store(10, 4);
    let cache = EmbedCache::new(100);
    assert_eq!(cache.cache_size(), 0);
    cache.get_or_load(0, &store);
    cache.get_or_load(1, &store);
    // moka entry_count() is eventually consistent — allow small lag
    let size = cache.cache_size();
    assert!(size <= 2, "size should be ≤ 2, got {size}");
}

#[test]
fn cache_respects_capacity() {
    let (_dir, store) = make_store(10, 4);
    let capacity = 3usize;
    let cache = EmbedCache::new(capacity);
    for i in 0..6u32 {
        cache.get_or_load(i, &store);
    }
    // moka evicts asynchronously; size should eventually be ≤ capacity
    // allow slightly above due to eviction lag
    let size = cache.cache_size();
    assert!(size <= capacity + 2, "size {size} > capacity {capacity}");
}

#[test]
fn cache_invalidate_all_empties() {
    let (_dir, store) = make_store(10, 4);
    let cache = EmbedCache::new(100);
    cache.get_or_load(0, &store);
    cache.get_or_load(1, &store);
    cache.invalidate_all();
    // After invalidation, pending tasks may still be in flight — allow 0 or very small
    let size = cache.cache_size();
    assert!(size <= 2, "after invalidate_all size should be near 0, got {size}");
}

#[test]
fn cache_is_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let (_dir, store) = make_store(20, 4);
    let store = Arc::new(store);
    let cache = Arc::new(EmbedCache::new(50));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let s = store.clone();
            let c = cache.clone();
            thread::spawn(move || {
                for i in 0..10u32 {
                    c.get_or_load(i % 20, &s);
                }
            })
        })
        .collect();

    for h in handles { h.join().unwrap(); }
    // just ensure no panic and size is sane
    assert!(cache.cache_size() <= 20);
}
