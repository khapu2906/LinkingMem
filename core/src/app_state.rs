/// Shared application state injected into every Axum handler via `State<AppState>`.
///
/// All fields are `Arc`-wrapped so `AppState` is cheaply cloneable.
/// `graph`, `hnsw`, and `store` are additionally wrapped in `RwLock` so they
/// can be hot-swapped after a delta merge without restarting the server.

use std::{path::PathBuf, sync::Arc};
use anyhow::Result;

use crate::{
    cache::EmbedCache,
    config::AppConfig,
    delta::DeltaStore,
    graph::{builder, csr::CsrGraph},
    metrics::Metrics,
    plugin::PluginClient,
    query::QueryEngine,
    storage::{LocalStorage, StorageBackend},
    vector::{
        hnsw::{EdgeHnswIndex, HnswIndex, load_edge_hnsw, normalise},
        store::VectorStore,
    },
};

// ── state ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub engine:     Arc<QueryEngine>,
    pub delta:      Arc<DeltaStore>,
    pub metrics:    Arc<Metrics>,
    pub plugin:     Arc<PluginClient>,
    pub storage:    Arc<dyn StorageBackend>,
    pub cfg:        Arc<AppConfig>,
    /// Monotonic clock recorded at startup — used by /health for uptime_secs.
    pub started_at: Arc<std::time::Instant>,
    // hot-swappable — QueryEngine holds the same Arcs, swapping here is
    // immediately visible to all in-flight and future queries
    pub graph:     Arc<tokio::sync::RwLock<Arc<CsrGraph>>>,
    pub hnsw:      Arc<tokio::sync::RwLock<Arc<HnswIndex>>>,
    pub store:     Arc<tokio::sync::RwLock<Arc<VectorStore>>>,
    pub edge_hnsw: Arc<tokio::sync::RwLock<Arc<EdgeHnswIndex>>>,
}

// ── bootstrap ─────────────────────────────────────────────────────────────────

pub async fn boot(cfg: &AppConfig) -> Result<AppState> {
    let storage: Arc<dyn StorageBackend> =
        Arc::new(LocalStorage::new(cfg.data.dir.clone()));

    let plugin = Arc::new(PluginClient::new(&cfg.plugins)?);

    for attempt in 1..=10 {
        if plugin.health().await { break; }
        if attempt == 10 {
            anyhow::bail!("plugin server not reachable at {}", cfg.plugins.embed_text.url());
        }
        tracing::warn!("waiting for plugin server ({attempt}/10)…");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    let graph_arc = Arc::new(builder::load(&*storage)?);
    tracing::info!("graph: {} nodes, {} edges", graph_arc.num_nodes(), graph_arc.num_edges());

    let store_arc = Arc::new(VectorStore::open(&storage.local_path().join("vectors.bin"))?);
    tracing::info!("building HNSW ({} vectors)…", store_arc.num_vecs);

    let vecs: Vec<Vec<f32>> = (0..store_arc.num_vecs as u32)
        .map(|id| { let mut v = store_arc.get(id).to_vec(); let _ = normalise(&mut v); v })
        .collect();
    let ids: Vec<u32> = (0..store_arc.num_vecs as u32).collect();
    // HNSW build is CPU-intensive — run on the blocking thread pool so we don't
    // stall the tokio async runtime during startup.
    let (hnsw_m, hnsw_ef) = (cfg.query.hnsw_m, cfg.query.hnsw_ef_construction);
    let hnsw_arc = Arc::new(
        tokio::task::spawn_blocking(move || HnswIndex::build(vecs, ids, hnsw_m, hnsw_ef))
            .await
            .map_err(|e| anyhow::anyhow!("HNSW build panicked: {e}"))??,
    );
    tracing::info!("HNSW built (M={}, ef_construction={})", cfg.query.hnsw_m, cfg.query.hnsw_ef_construction);

    let storage2 = storage.clone();
    let (hnsw_m2, hnsw_ef2) = (cfg.query.hnsw_m, cfg.query.hnsw_ef_construction);
    let edge_hnsw_arc = Arc::new(
        tokio::task::spawn_blocking(move || load_edge_hnsw(&*storage2, hnsw_m2, hnsw_ef2))
            .await
            .map_err(|e| anyhow::anyhow!("edge HNSW build panicked: {e}"))??,
    );
    tracing::info!("edge HNSW ready ({} edges)", if edge_hnsw_arc.is_empty() { 0 } else { 1 });

    let graph_rw: Arc<tokio::sync::RwLock<Arc<CsrGraph>>> =
        Arc::new(tokio::sync::RwLock::new(graph_arc));
    let hnsw_rw: Arc<tokio::sync::RwLock<Arc<HnswIndex>>> =
        Arc::new(tokio::sync::RwLock::new(hnsw_arc));
    let store_rw: Arc<tokio::sync::RwLock<Arc<VectorStore>>> =
        Arc::new(tokio::sync::RwLock::new(store_arc));
    let edge_hnsw_rw: Arc<tokio::sync::RwLock<Arc<EdgeHnswIndex>>> =
        Arc::new(tokio::sync::RwLock::new(edge_hnsw_arc));

    let cache   = Arc::new(EmbedCache::new(cfg.cache.embed_cache_size));
    let metrics = Metrics::new();

    {
        let g = graph_rw.read().await;
        metrics.graph_nodes.set(g.num_nodes() as u64);
        metrics.graph_edges.set(g.num_edges() as u64);
    }

    let data_dir: PathBuf = storage.local_path().to_path_buf();
    let delta = Arc::new(DeltaStore::new(data_dir, cfg.delta.merge_threshold));

    let replayed = delta.replay_wal();
    if replayed > 0 {
        tracing::info!("restored {replayed} delta entries from WAL");
        metrics.delta_size.set(delta.size() as u64);
    }
    // Initialize node ID allocator: base graph nodes + any delta nodes from WAL replay.
    // Must happen after both load and replay so the counter starts past all known IDs.
    {
        let g = graph_rw.read().await;
        delta.init_node_ids(g.num_nodes() as u32 + delta.read().nodes_len() as u32);
    }

    let engine = Arc::new(QueryEngine::new(
        graph_rw.clone(),
        hnsw_rw.clone(),
        store_rw.clone(),
        edge_hnsw_rw.clone(),
        cache,
        plugin.clone(),
    ));

    Ok(AppState {
        engine, delta, metrics, plugin, storage, cfg: Arc::new(cfg.clone()),
        started_at: Arc::new(std::time::Instant::now()),
        graph: graph_rw, hnsw: hnsw_rw, store: store_rw, edge_hnsw: edge_hnsw_rw,
    })
}

// ── merge helper ──────────────────────────────────────────────────────────────

/// Merge delta into the main graph and hot-swap all four indices atomically.
/// Spawned as a background task after each ingest that crosses the threshold.
pub async fn run_merge(s: AppState) {
    let base = s.graph.read().await.clone();
    let (new_graph, new_node_vecs, new_edge_vecs, new_edge_endpoints) =
        match s.delta.merge_into(base, &*s.storage).await {
            Ok(r)  => r,
            Err(e) => { tracing::error!("merge failed: {e}"); return; }
        };

    // Both HNSW builds are CPU-intensive — offload to blocking thread pool so
    // in-flight query tasks keep running on the async runtime during rebuild.
    let node_ids = (0..new_node_vecs.len() as u32).collect::<Vec<_>>();
    let (hnsw_m, hnsw_ef) = (s.cfg.query.hnsw_m, s.cfg.query.hnsw_ef_construction);

    let new_hnsw = match tokio::task::spawn_blocking(move || {
        HnswIndex::build(new_node_vecs, node_ids, hnsw_m, hnsw_ef)
    }).await {
        Ok(Ok(h))  => h,
        Ok(Err(e)) => { tracing::error!("HNSW rebuild failed: {e}"); return; }
        Err(e)     => { tracing::error!("HNSW rebuild panicked: {e}"); return; }
    };

    let new_edge_hnsw = match tokio::task::spawn_blocking(move || {
        EdgeHnswIndex::build(new_edge_vecs, new_edge_endpoints, hnsw_m, hnsw_ef)
    }).await {
        Ok(Ok(h))  => h,
        Ok(Err(e)) => { tracing::error!("edge HNSW rebuild failed: {e}"); return; }
        Err(e)     => { tracing::error!("edge HNSW rebuild panicked: {e}"); return; }
    };

    let new_store = match VectorStore::open(&s.storage.local_path().join("vectors.bin")) {
        Ok(st) => st,
        Err(e) => { tracing::error!("VectorStore reload failed: {e}"); return; }
    };

    // Acquire all four write locks before swapping any —
    // prevents queries from observing partially-swapped state.
    let mut g  = s.graph.write().await;
    let mut h  = s.hnsw.write().await;
    let mut eh = s.edge_hnsw.write().await;
    let mut st = s.store.write().await;
    *g  = Arc::new(new_graph);
    *h  = Arc::new(new_hnsw);
    *eh = Arc::new(new_edge_hnsw);
    *st = Arc::new(new_store);
    drop(g); drop(h); drop(eh); drop(st);
    s.engine.invalidate_query_cache();
    tracing::info!("hot-swap complete (graph + hnsw + edge_hnsw + store)");
}
