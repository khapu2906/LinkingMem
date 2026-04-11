use instant_distance::{Builder, HnswMap, Point, Search};
use anyhow::Result;
use crate::vector::store::cosine_sim;
use crate::vector::store::VectorStore;

/// Newtype wrapper so we can implement instant_distance::Point
#[derive(Clone)]
pub struct FloatVec(pub Vec<f32>);

impl Point for FloatVec {
    /// Instant-distance minimises distance, so we return 1 - cosine_sim.
    /// Assumes unit-length vectors (normalise before inserting).
    fn distance(&self, other: &Self) -> f32 {
        1.0 - cosine_sim(&self.0, &other.0)
    }
}

/// Wrapper around instant-distance HnswMap.
///
/// Parameters to tune:
///   M              — connections per node (default 16). Higher = better recall, more RAM.
///   ef_construction — beam width during build (default 200). Higher = better recall, slower build.
pub struct HnswIndex {
    inner: HnswMap<FloatVec, u32>,
}

impl HnswIndex {
    /// Build HNSW index from vectors + node ids.
    pub fn build(vectors: Vec<Vec<f32>>, ids: Vec<u32>, _m: usize, ef: usize) -> Result<Self> {
        let points: Vec<FloatVec> = vectors.into_iter().map(FloatVec).collect();

        let inner = Builder::default()
            .ef_construction(ef)
            .build(points, ids);

        tracing::info!("built HNSW index: ef_construction={}", ef);
        Ok(Self { inner })
    }

    /// Approximate nearest neighbour search. Returns (node_id, distance) pairs.
    /// ef_search controls recall/speed trade-off at query time (default = k * 2).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let q = FloatVec(query.to_vec());
        let mut search = Search::default();
        self.inner
            .search(&q, &mut search)
            .take(k)
            .map(|item| (*item.value, item.distance))
            .collect()
    }
}

// ── edge HNSW ────────────────────────────────────────────────────────────────

/// HNSW index over edge embeddings.
///
/// Each edge has a `full_context` embedding. When queried, returns the
/// *endpoint node IDs* of matching edges so BFS can start from them.
///
/// `endpoints[i]` = (from_node_id, to_node_id) for edge index i.
/// The HNSW stores edge indices (0..M-1) as values.
pub struct EdgeHnswIndex {
    inner:     Option<HnswMap<FloatVec, u32>>,
    endpoints: Vec<(u32, u32)>,
}

impl EdgeHnswIndex {
    /// Build from pre-normalised edge embedding vectors and their endpoints.
    /// Returns an empty index (no-op search) when `vecs` is empty.
    pub fn build(vecs: Vec<Vec<f32>>, endpoints: Vec<(u32, u32)>, _m: usize, ef: usize) -> Result<Self> {
        assert_eq!(vecs.len(), endpoints.len(), "edge vecs and endpoints must be same length");
        if vecs.is_empty() {
            return Ok(Self { inner: None, endpoints: vec![] });
        }
        let ids:    Vec<u32>     = (0..vecs.len() as u32).collect();
        let points: Vec<FloatVec> = vecs.into_iter().map(FloatVec).collect();
        let inner = Builder::default()
            .ef_construction(ef)
            .build(points, ids);
        tracing::info!("built edge HNSW: {} edges (ef={})", endpoints.len(), ef);
        Ok(Self { inner: Some(inner), endpoints })
    }

    /// Search for edges whose `full_context` is semantically close to `query`.
    /// Returns `(from_node_id, to_node_id, distance)` for the top-k hits.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, u32, f32)> {
        let inner = match &self.inner { Some(i) => i, None => return vec![] };
        let q = FloatVec(query.to_vec());
        let mut search = Search::default();
        inner.search(&q, &mut search)
            .take(k)
            .filter_map(|item| {
                let idx = *item.value as usize;
                self.endpoints.get(idx).map(|&(from, to)| (from, to, item.distance))
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool { self.endpoints.is_empty() }
}

/// Load edge vectors + endpoints via storage backend and build the edge HNSW.
/// Gracefully returns an empty index when files are absent (first run / old data).
pub fn load_edge_hnsw(
    storage: &dyn crate::storage::StorageBackend,
    m:       usize,
    ef:      usize,
) -> Result<EdgeHnswIndex> {
    if !storage.exists("edge_vectors.bin") || !storage.exists("edge_endpoints.json") {
        tracing::info!("no edge_vectors.bin found — starting with empty edge HNSW");
        return EdgeHnswIndex::build(vec![], vec![], m, ef);
    }

    let store = VectorStore::open(&storage.local_path().join("edge_vectors.bin"))?;
    let mut vecs: Vec<Vec<f32>> = (0..store.num_vecs as u32)
        .map(|id| { let mut v = store.get(id).to_vec(); let _ = normalise(&mut v); v })
        .collect();

    let endpoints_json = storage.read_string("edge_endpoints.json")?;
    let endpoints: Vec<(u32, u32)> = serde_json::from_str(&endpoints_json)?;

    if vecs.len() != endpoints.len() {
        tracing::warn!(
            "edge_vectors.bin ({}) and edge_endpoints.json ({}) length mismatch — rebuilding empty",
            vecs.len(), endpoints.len()
        );
        vecs.clear();
        return EdgeHnswIndex::build(vec![], vec![], m, ef);
    }

    tracing::info!("loading edge HNSW ({} edges)…", vecs.len());
    EdgeHnswIndex::build(vecs, endpoints, m, ef)
}

/// Persist edge embeddings and endpoints via storage backend (called after merge).
pub fn save_edge_hnsw_data(
    storage:   &dyn crate::storage::StorageBackend,
    vecs:      &[Vec<f32>],
    endpoints: &[(u32, u32)],
) -> Result<()> {
    use crate::vector::store::VectorStore;
    let dim = vecs.first().map(|v| v.len()).unwrap_or(0);
    if dim > 0 {
        VectorStore::write(&storage.local_path().join("edge_vectors.bin"), dim, vecs)?;
    }
    storage.write_string("edge_endpoints.json", &serde_json::to_string(endpoints)?)?;
    Ok(())
}

// ── normalise ────────────────────────────────────────────────────────────────

/// Normalise a vector to unit length (required for cosine = dot product trick).
/// Returns false if the vector is near-zero (degenerate embedding) — caller should skip it.
pub fn normalise(v: &mut Vec<f32>) -> bool {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
        true
    } else {
        tracing::warn!("normalise: near-zero vector (norm={norm:.2e}), skipping normalisation");
        false
    }
}
