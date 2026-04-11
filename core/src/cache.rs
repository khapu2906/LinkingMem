use std::sync::Arc;
use moka::sync::Cache;
use crate::vector::store::VectorStore;

/// Thread-safe concurrent LRU cache for hot embeddings.
///
/// Uses moka's sharded concurrent cache — no global lock on reads.
/// Hot path: cache hit returns in μs without contention.
/// Cold path: load from mmap (~50μs), insert into cache.
///
/// Stores Arc<Vec<f32>> to avoid double allocation on cache miss:
/// one Arc is kept in the cache, the other is returned to the caller.
pub struct EmbedCache {
    cache: Cache<u32, Arc<Vec<f32>>>,
    capacity: usize,
}

impl EmbedCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Cache::new(capacity as u64),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get embedding — returns from cache or loads from mmap store.
    /// Lock-free on cache hit (moka uses internal sharding).
    /// Returns Arc to avoid cloning the vector on every access.
    pub fn get_or_load(&self, id: u32, store: &VectorStore) -> Arc<Vec<f32>> {
        if let Some(v) = self.cache.get(&id) {
            return v; // cache hit — Arc clone, no data copy
        }
        // cache miss — load from mmap, wrap in Arc once
        let arc = Arc::new(store.get(id).to_vec());
        self.cache.insert(id, arc.clone());
        arc
    }

    pub fn cache_size(&self) -> usize {
        self.cache.entry_count() as usize
    }

    pub fn invalidate_all(&self) {
        self.cache.invalidate_all();
    }
}
