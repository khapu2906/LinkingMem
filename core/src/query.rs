use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use moka::sync::Cache;
use serde::{Deserialize, Serialize};

use crate::cache::EmbedCache;
use crate::graph::csr::{CsrGraph, NodeInfo};
use crate::plugin::{LlmHints, PluginClient};
use crate::vector::hnsw::{EdgeHnswIndex, HnswIndex, normalise};
use crate::vector::store::{VectorStore, cosine_sim};

// ─── scoring weights ────────────────────────────────────────────────────────

/// Controls the blend of three scoring signals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// cosine similarity with the query embedding
    pub alpha: f32,
    /// edge-weighted graph proximity — rewards nodes close to seeds via strong edges
    pub beta: f32,
    /// combined in+out degree importance
    pub gamma: f32,
}

impl ScoringWeights {
    pub fn balanced()        -> Self { Self { alpha: 0.5, beta: 0.3, gamma: 0.2 } }
    pub fn semantic_search() -> Self { Self { alpha: 0.7, beta: 0.2, gamma: 0.1 } }
    pub fn relationship()    -> Self { Self { alpha: 0.3, beta: 0.6, gamma: 0.1 } }
    pub fn entity_lookup()   -> Self { Self { alpha: 0.4, beta: 0.2, gamma: 0.4 } }
}

impl Default for ScoringWeights {
    fn default() -> Self { Self::balanced() }
}

// ─── query options ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryOptions {
    /// top-k seed nodes from HNSW
    pub hnsw_k: usize,
    /// BFS expansion depth from seeds
    pub bfs_depth: u8,
    /// max nodes to collect during BFS
    pub bfs_max_nodes: usize,
    /// top-n nodes passed to LLM as context
    pub context_top_n: usize,
    /// minimum score to include a node in LLM context (0.0 = no cutoff)
    pub context_min_score: f32,
    pub weights: ScoringWeights,
    /// traverse both forward and reverse edges in BFS (relationship mode)
    #[serde(default)]
    pub bidirectional: bool,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            hnsw_k: 20,
            bfs_depth: 2,
            bfs_max_nodes: 500,
            context_top_n: 50,
            context_min_score: 0.3,
            weights: ScoringWeights::default(),
            bidirectional: false,
        }
    }
}

// ─── result types ────────────────────────────────────────────────────────────

/// A node in the result subgraph with relevance scores.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScoredNode {
    pub id:           u32,
    pub name:         String,
    #[serde(rename = "type")]
    pub node_type:    String,
    pub props:        serde_json::Value,
    pub full_context: String,
    pub score:        f32,
    pub vector_sim:   f32,
    pub hop:          u8,
    /// true = found directly by HNSW; false = reached via BFS expansion
    pub is_seed:      bool,
}

/// A directed edge between two nodes in the result subgraph.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SubgraphEdge {
    pub from:      String,
    pub to:        String,
    pub weight:    f32,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub edge_type: String,
}

/// The subgraph returned to the caller — nodes + edges for graph visualisation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Subgraph {
    pub nodes: Vec<ScoredNode>,
    pub edges: Vec<SubgraphEdge>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueryResult {
    pub answer:   String,
    pub subgraph: Subgraph,
    pub stats:    QueryStats,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueryStats {
    pub seed_nodes:     usize,
    pub subgraph_nodes: usize,
    pub context_nodes:  usize,
    /// Wall-clock time for this request (near-zero on cache hit)
    pub total_ms:   u64,
    pub embed_ms:   u64,
    pub search_ms:  u64,
    pub bfs_ms:     u64,
    pub score_ms:   u64,
    pub llm_ms:     u64,
    /// true = served from cache; component timings are zeroed
    pub cache_hit:  bool,
}

// ─── query result cache ───────────────────────────────────────────────────────

pub struct QueryCache {
    inner: Cache<String, QueryResult>,
}

impl QueryCache {
    pub fn new(capacity: u64, ttl_secs: u64) -> Self {
        Self {
            inner: Cache::builder()
                .max_capacity(capacity)
                .time_to_live(Duration::from_secs(ttl_secs))
                .build(),
        }
    }

    /// Returns None if opts cannot be serialized (e.g. NaN weights) — callers
    /// must skip the cache in that case rather than risk a key collision.
    pub fn make_key(prefix: &str, id: &str, opts: &QueryOptions, llm_generate: bool) -> Option<String> {
        let opts_str = serde_json::to_string(opts).ok()?;
        Some(format!("{}||{}||{}||{}", prefix, id, llm_generate, opts_str))
    }

    pub fn get_by_key(&self, key: &str) -> Option<QueryResult> {
        self.inner.get(key)
    }

    pub fn insert_by_key(&self, key: String, result: QueryResult) {
        self.inner.insert(key, result);
    }

    pub fn invalidate_all(&self) {
        self.inner.invalidate_all();
    }
}

// ─── engine ──────────────────────────────────────────────────────────────────

pub struct QueryEngine {
    graph:       Arc<tokio::sync::RwLock<Arc<CsrGraph>>>,
    hnsw:        Arc<tokio::sync::RwLock<Arc<HnswIndex>>>,
    edge_hnsw:   Arc<tokio::sync::RwLock<Arc<EdgeHnswIndex>>>,
    pub store:   Arc<tokio::sync::RwLock<Arc<VectorStore>>>,
    cache:       Arc<EmbedCache>,
    plugin:      Arc<PluginClient>,
    query_cache: QueryCache,
}

impl QueryEngine {
    pub fn new(
        graph:     Arc<tokio::sync::RwLock<Arc<CsrGraph>>>,
        hnsw:      Arc<tokio::sync::RwLock<Arc<HnswIndex>>>,
        store:     Arc<tokio::sync::RwLock<Arc<VectorStore>>>,
        edge_hnsw: Arc<tokio::sync::RwLock<Arc<EdgeHnswIndex>>>,
        cache:     Arc<EmbedCache>,
        plugin:    Arc<PluginClient>,
    ) -> Self {
        let query_cache = QueryCache::new(10_000, 300);
        Self { graph, hnsw, edge_hnsw, store, cache, plugin, query_cache }
    }

    pub fn invalidate_query_cache(&self) {
        self.query_cache.invalidate_all();
    }

    /// Backward-compatible entry point — text query with LLM generation.
    pub async fn query(&self, q: &str, opts: QueryOptions) -> Result<QueryResult> {
        self.query_text(q, opts, true, None).await
    }

    // ── Public entry points ───────────────────────────────────────────────────

    /// Text query: embed → HNSW search → BFS → score → [LLM]
    pub async fn query_text(&self, q: &str, opts: QueryOptions, llm_generate: bool, hints: Option<LlmHints>) -> Result<QueryResult> {
        let t0 = std::time::Instant::now();

        // Skip cache when caller provided hints — different hints produce different answers
        let cache_key = if hints.is_none() {
            QueryCache::make_key("text", q.trim(), &opts, llm_generate)
        } else {
            None
        };
        if let Some(ref key) = cache_key {
            if let Some(cached) = self.query_cache.get_by_key(key) {
                return Ok(Self::mark_cache_hit(cached, t0));
            }
        }

        // Step 1: embed
        let mut qvec = self.plugin.embed_one(q).await?;
        let _ = normalise(&mut qvec);
        let embed_ms = t0.elapsed().as_millis() as u64;

        // Step 2: HNSW search
        let t1 = std::time::Instant::now();
        let (seed_hits, edge_seed_hits) = self.search_hnsw(&qvec, opts.hnsw_k).await;
        let seed_ids  = Self::merge_seeds(&seed_hits, &edge_seed_hits);
        let search_ms = t1.elapsed().as_millis() as u64;

        let result = self.run_pipeline(q, qvec, seed_ids, seed_hits, &opts, embed_ms, search_ms, t0, llm_generate, hints).await?;

        if let Some(key) = cache_key {
            self.query_cache.insert_by_key(key, result.clone());
        }
        Ok(result)
    }

    /// Vector query: HNSW search → BFS → score → [LLM] (embed step skipped)
    pub async fn query_with_vector(&self, mut qvec: Vec<f32>, q: &str, opts: QueryOptions, llm_generate: bool, hints: Option<LlmHints>) -> Result<QueryResult> {
        let t0 = std::time::Instant::now();
        let _ = normalise(&mut qvec);

        // Step 2: HNSW search (step 1 skipped)
        let t1 = std::time::Instant::now();
        let (seed_hits, edge_seed_hits) = self.search_hnsw(&qvec, opts.hnsw_k).await;
        let seed_ids  = Self::merge_seeds(&seed_hits, &edge_seed_hits);
        let search_ms = t1.elapsed().as_millis() as u64;

        // No cache for vector queries — vectors are not practical cache keys
        self.run_pipeline(q, qvec, seed_ids, seed_hits, &opts, 0, search_ms, t0, llm_generate, hints).await
    }

    /// Node query: BFS from node_id → score → [LLM] (no embed, no HNSW)
    ///
    /// Seed node is given vsim=1.0 so the alpha term keeps it top-ranked.
    /// Expanded nodes are scored purely by graph proximity + node_weight.
    /// Callers should set `context_min_score=0.0` in opts for full exploration.
    pub async fn query_from_node(&self, node_id: u32, q: &str, opts: QueryOptions, llm_generate: bool, hints: Option<LlmHints>) -> Result<QueryResult> {
        let t0 = std::time::Instant::now();

        {
            let graph = self.graph.read().await;
            if node_id as usize >= graph.num_nodes() {
                return Err(anyhow::anyhow!("node {node_id} not found"));
            }
        }

        let cache_key = if hints.is_none() {
            QueryCache::make_key("node", &node_id.to_string(), &opts, llm_generate)
        } else {
            None
        };
        if let Some(ref key) = cache_key {
            if let Some(cached) = self.query_cache.get_by_key(key) {
                return Ok(Self::mark_cache_hit(cached, t0));
            }
        }

        // Seed node with distance=0.0 → vsim = 1.0 - 0.0 = 1.0
        let seed_hits = vec![(node_id, 0.0_f32)];
        let seed_ids  = vec![node_id];
        // query_vec is empty — cosine_sim on empty slices returns 0.0 safely
        let result = self.run_pipeline(q, vec![], seed_ids, seed_hits, &opts, 0, 0, t0, llm_generate, hints).await?;

        if let Some(key) = cache_key {
            self.query_cache.insert_by_key(key, result.clone());
        }
        Ok(result)
    }

    // ── Multi-hop reasoning ───────────────────────────────────────────────────

    /// Multi-hop query: iterative graph reasoning.
    ///
    /// Pipeline:
    ///   1. embed → HNSW → BFS → score  (same as regular query)
    ///   2. call plugin /reason — LLM decides if it needs more context
    ///   3. if done=false: embed follow-up entity names → HNSW → BFS, merge context
    ///   4. repeat up to `max_hops` times; final /generate on last iteration
    pub async fn query_multihop(
        &self,
        q:        &str,
        opts:     QueryOptions,
        max_hops: u8,
        hints:    Option<LlmHints>,
    ) -> Result<QueryResult> {
        let t0 = std::time::Instant::now();

        // ── step 1: initial retrieval ─────────────────────────────────────
        let mut qvec = self.plugin.embed_one(q).await?;
        let _ = normalise(&mut qvec);
        let embed_ms = t0.elapsed().as_millis() as u64;

        let t1 = std::time::Instant::now();
        let (seed_hits, edge_seed_hits) = self.search_hnsw(&qvec, opts.hnsw_k).await;
        let seed_ids = Self::merge_seeds(&seed_hits, &edge_seed_hits);
        let search_ms = t1.elapsed().as_millis() as u64;

        let (mut node_infos, mut raw_edges) =
            self.retrieve_context(&qvec, seed_ids, &seed_hits, &opts).await?;

        // ── step 2: reason loop ───────────────────────────────────────────
        let mut hops_used: u8 = 0;
        let final_answer: String;

        loop {
            let reason = self.plugin
                .reason(&node_infos, &raw_edges, q, hops_used as u32, max_hops as u32, hints.clone())
                .await?;

            if reason.done || reason.follow_ups.is_empty() || hops_used >= max_hops {
                final_answer = reason.answer;
                break;
            }

            hops_used += 1;

            // embed follow-up entity names, search HNSW, expand context
            let extra_ids = self.resolve_entity_names(&reason.follow_ups, opts.hnsw_k).await;
            if extra_ids.is_empty() {
                // nothing new to explore — generate with what we have
                final_answer = self.plugin.generate(&node_infos, &raw_edges, q, hints.clone()).await?;
                break;
            }

            let extra_hits: Vec<(u32, f32)> = extra_ids.iter().map(|&id| (id, 0.0_f32)).collect();
            let (extra_nodes, extra_edges) =
                self.retrieve_context(&qvec, extra_ids, &extra_hits, &opts).await?;

            QueryEngine::merge_context(&mut node_infos, extra_nodes);
            QueryEngine::merge_edges(&mut raw_edges, extra_edges);

            // keep context within bounds
            node_infos.sort_by(|a, b| b.weight.partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal));
            node_infos.truncate(opts.context_top_n);
        }

        let llm_ms = t0.elapsed().as_millis() as u64 - embed_ms - search_ms;

        // rebuild ScoredNode list for the response
        let scored: Vec<crate::query::ScoredNode> = node_infos.iter().map(|n| ScoredNode {
            id: n.id, name: n.name.clone(), node_type: n.node_type.clone(),
            props: n.props.clone(), full_context: n.full_context.clone(),
            score: n.weight, vector_sim: 0.0, hop: 0, is_seed: false,
        }).collect();

        let id_to_name: HashMap<u32, &str> =
            node_infos.iter().map(|n| (n.id, n.name.as_str())).collect();
        let subgraph_edges: Vec<SubgraphEdge> = raw_edges.iter()
            .filter_map(|(from, to, w, et)| Some(SubgraphEdge {
                from: id_to_name.get(from)?.to_string(),
                to:   id_to_name.get(to)?.to_string(),
                weight: *w, edge_type: et.clone(),
            }))
            .collect();

        Ok(QueryResult {
            answer: final_answer,
            subgraph: Subgraph { nodes: scored, edges: subgraph_edges },
            stats: QueryStats {
                seed_nodes: 0,
                subgraph_nodes: node_infos.len(),
                context_nodes: node_infos.len(),
                total_ms: t0.elapsed().as_millis() as u64,
                embed_ms, search_ms, bfs_ms: 0, score_ms: 0, llm_ms,
                cache_hit: false,
            },
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn mark_cache_hit(mut r: QueryResult, t0: std::time::Instant) -> QueryResult {
        r.stats.cache_hit  = true;
        r.stats.total_ms   = t0.elapsed().as_millis() as u64;
        r.stats.embed_ms   = 0;
        r.stats.search_ms  = 0;
        r.stats.bfs_ms     = 0;
        r.stats.score_ms   = 0;
        r.stats.llm_ms     = 0;
        r
    }

    async fn search_hnsw(&self, qvec: &[f32], k: usize) -> (Vec<(u32, f32)>, Vec<(u32, u32, f32)>) {
        let hnsw_guard      = self.hnsw.read();
        let edge_hnsw_guard = self.edge_hnsw.read();
        let (hnsw, edge_hnsw) = tokio::join!(hnsw_guard, edge_hnsw_guard);
        let s = hnsw.search(qvec, k);
        let e = edge_hnsw.search(qvec, k);
        (s, e)
    }

    fn merge_seeds(seed_hits: &[(u32, f32)], edge_seed_hits: &[(u32, u32, f32)]) -> Vec<u32> {
        let mut seen: HashSet<u32> = seed_hits.iter().map(|(id, _)| *id).collect();
        let mut seed_ids: Vec<u32> = seen.iter().copied().collect();
        for (from, to, _) in edge_seed_hits {
            if seen.insert(*from) { seed_ids.push(*from); }
            if seen.insert(*to)   { seed_ids.push(*to); }
        }
        seed_ids
    }

    /// Shared pipeline from BFS onwards.
    async fn run_pipeline(
        &self,
        q:            &str,
        query_vec:    Vec<f32>,        // empty = no vector scoring (node query)
        seed_ids:     Vec<u32>,
        seed_hits:    Vec<(u32, f32)>, // (node_id, distance) for vsim lookup
        opts:         &QueryOptions,
        embed_ms:     u64,
        search_ms:    u64,
        t0:           std::time::Instant,
        llm_generate: bool,
        hints:        Option<LlmHints>,
    ) -> Result<QueryResult> {
        // Step 3: BFS expansion
        let t2       = std::time::Instant::now();
        let graph    = self.graph.read().await;
        let subgraph = graph.bfs_expand(&seed_ids, opts.bfs_depth, opts.bfs_max_nodes, opts.bidirectional);
        let bfs_ms   = t2.elapsed().as_millis() as u64;

        // Step 4: score + rank
        let t3    = std::time::Instant::now();
        let store = self.store.read().await;
        let mut scored = self.score_nodes(&graph, &store, &subgraph, &query_vec, &seed_hits, &opts.weights);
        drop(store);
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        if opts.context_min_score > 0.0 {
            scored.retain(|n| n.score >= opts.context_min_score);
        }
        scored.truncate(opts.context_top_n);

        let context_ids: HashSet<u32> = scored.iter().map(|sn| sn.id).collect();
        let raw_edges = graph.edges_between(&context_ids);
        drop(graph);
        let score_ms = t3.elapsed().as_millis() as u64;

        let id_to_name: HashMap<u32, &str> =
            scored.iter().map(|sn| (sn.id, sn.name.as_str())).collect();
        let subgraph_edges: Vec<SubgraphEdge> = raw_edges.iter()
            .filter_map(|(from, to, w, et)| Some(SubgraphEdge {
                from:      id_to_name.get(from)?.to_string(),
                to:        id_to_name.get(to)?.to_string(),
                weight:    *w,
                edge_type: et.clone(),
            }))
            .collect();

        let node_infos: Vec<NodeInfo> = scored.iter()
            .map(|sn| NodeInfo {
                id:            sn.id,
                external_id:   String::new(),
                name:          sn.name.clone(),
                node_type:     sn.node_type.clone(),
                weight:        sn.score,
                props:         sn.props.clone(),
                full_context:  sn.full_context.clone(),
                embed_context: None,
                image_url:     None,
            })
            .collect();

        // Step 5: LLM generate
        let t4     = std::time::Instant::now();
        let answer = if llm_generate {
            self.plugin.generate(&node_infos, &raw_edges, q, hints).await?
        } else {
            String::new()
        };
        let llm_ms = t4.elapsed().as_millis() as u64;

        Ok(QueryResult {
            answer,
            subgraph: Subgraph { nodes: scored, edges: subgraph_edges },
            stats: QueryStats {
                seed_nodes:     seed_ids.len(),
                subgraph_nodes: subgraph.len(),
                context_nodes:  node_infos.len(),
                total_ms:  t0.elapsed().as_millis() as u64,
                embed_ms,
                search_ms,
                bfs_ms,
                score_ms,
                llm_ms,
                cache_hit: false,
            },
        })
    }

    /// Run BFS → score → filter → return (NodeInfo list, raw edges).
    /// Used by both `run_pipeline` and multi-hop expansion.
    async fn retrieve_context(
        &self,
        query_vec: &[f32],
        seed_ids:  Vec<u32>,
        seed_hits: &[(u32, f32)],
        opts:      &QueryOptions,
    ) -> Result<(Vec<NodeInfo>, Vec<(u32, u32, f32, String)>)> {
        let graph    = self.graph.read().await;
        let subgraph = graph.bfs_expand(&seed_ids, opts.bfs_depth, opts.bfs_max_nodes, opts.bidirectional);

        let store = self.store.read().await;
        let mut scored = self.score_nodes(&graph, &store, &subgraph, query_vec, seed_hits, &opts.weights);
        drop(store);

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        if opts.context_min_score > 0.0 {
            scored.retain(|n| n.score >= opts.context_min_score);
        }
        scored.truncate(opts.context_top_n);

        let context_ids: HashSet<u32> = scored.iter().map(|sn| sn.id).collect();
        let raw_edges = graph.edges_between(&context_ids);
        drop(graph);

        let node_infos = scored.iter().map(|sn| NodeInfo {
            id:            sn.id,
            external_id:   String::new(),
            name:          sn.name.clone(),
            node_type:     sn.node_type.clone(),
            weight:        sn.score,
            props:         sn.props.clone(),
            full_context:  sn.full_context.clone(),
            embed_context: None,
            image_url:     None,
        }).collect();

        Ok((node_infos, raw_edges))
    }

    /// Embed entity names returned by multi-hop reasoning and find matching
    /// node IDs via HNSW search.  Falls back to empty vec on error (non-fatal).
    async fn resolve_entity_names(&self, names: &[String], k: usize) -> Vec<u32> {
        let mut found: HashSet<u32> = HashSet::new();
        let graph = self.graph.read().await;

        for name in names {
            let vec = match self.plugin.embed_one(name).await {
                Ok(mut v) => { let _ = normalise(&mut v); v }
                Err(e) => { tracing::warn!("resolve '{name}': embed failed: {e}"); continue; }
            };

            let hits = self.hnsw.read().await.search(&vec, k.min(5));
            for (id, _dist) in hits {
                // Accept the hit only if the node name is a substring match
                // (prevents completely unrelated nodes from polluting context).
                if let Some(node) = graph.nodes.get(id as usize) {
                    let name_lower = name.to_lowercase();
                    let node_lower = node.name.to_lowercase();
                    if node_lower.contains(&name_lower) || name_lower.contains(&node_lower) {
                        found.insert(id);
                        break;
                    }
                }
            }
        }
        found.into_iter().collect()
    }

    /// Merge `extra` into `base`, deduplicating by node ID.
    fn merge_context(base: &mut Vec<NodeInfo>, extra: Vec<NodeInfo>) {
        let existing: HashSet<u32> = base.iter().map(|n| n.id).collect();
        for node in extra {
            if !existing.contains(&node.id) {
                base.push(node);
            }
        }
    }

    /// Merge `extra` edges into `base`, deduplicating by (from, to).
    fn merge_edges(
        base:  &mut Vec<(u32, u32, f32, String)>,
        extra: Vec<(u32, u32, f32, String)>,
    ) {
        let existing: HashSet<(u32, u32)> = base.iter().map(|e| (e.0, e.1)).collect();
        for edge in extra {
            if !existing.contains(&(edge.0, edge.1)) {
                base.push(edge);
            }
        }
    }

    fn score_nodes(
        &self,
        graph:     &CsrGraph,
        store:     &VectorStore,
        subgraph:  &[(u32, u8, f32)],
        query_vec: &[f32],            // empty = skip cosine sim (node query)
        seeds:     &[(u32, f32)],
        w:         &ScoringWeights,
    ) -> Vec<ScoredNode> {
        let seed_sim: HashMap<u32, f32> =
            seeds.iter().map(|(id, dist)| (*id, 1.0 - dist)).collect();
        let seed_set: HashSet<u32> =
            seeds.iter().map(|(id, _)| *id).collect();

        subgraph.iter().filter_map(|(node_id, hop, path_weight)| {
            let node = match graph.nodes.get(*node_id as usize) {
                Some(n) => n,
                None => {
                    tracing::warn!("score_nodes: node_id={node_id} out of range (graph has {} nodes) — skipping", graph.num_nodes());
                    return None;
                }
            };

            let vsim = if query_vec.is_empty() {
                // No query vector (node query) — seed has 1.0, expanded nodes 0.0
                seed_sim.get(node_id).copied().unwrap_or(0.0)
            } else {
                seed_sim.get(node_id).copied().unwrap_or_else(|| {
                    let embedding = self.cache.get_or_load(*node_id, store);
                    cosine_sim(&embedding, query_vec)
                })
            };

            let proximity = if *hop == 0 {
                vsim.max(0.0)
            } else {
                path_weight / (*hop as f32 + 1.0)
            };
            let score = w.alpha * vsim + w.beta * proximity + w.gamma * node.weight;

            Some(ScoredNode {
                id:           node.id,
                name:         node.name.clone(),
                node_type:    node.node_type.clone(),
                props:        node.props.clone(),
                full_context: node.full_context.clone(),
                score,
                vector_sim:   vsim,
                hop:          *hop,
                is_seed:      seed_set.contains(node_id),
            })
        }).collect()
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScoringWeights presets ────────────────────────────────────────────────

    #[test]
    fn balanced_weights_sum_to_one() {
        let w = ScoringWeights::balanced();
        let sum = w.alpha + w.beta + w.gamma;
        assert!((sum - 1.0).abs() < 1e-6, "balanced weights must sum to 1, got {sum}");
    }

    #[test]
    fn semantic_weights_alpha_dominant() {
        let w = ScoringWeights::semantic_search();
        assert!(w.alpha > w.beta && w.alpha > w.gamma);
    }

    #[test]
    fn relationship_weights_beta_dominant() {
        let w = ScoringWeights::relationship();
        assert!(w.beta > w.alpha && w.beta > w.gamma);
    }

    #[test]
    fn entity_weights_gamma_not_minimal() {
        // entity_lookup: alpha=0.4, beta=0.2, gamma=0.4 — gamma ties alpha, both beat beta
        let w = ScoringWeights::entity_lookup();
        assert!(w.gamma >= w.alpha, "gamma should be at least as large as alpha");
        assert!(w.gamma > w.beta,   "gamma should dominate beta");
    }

    // ── QueryCache key ────────────────────────────────────────────────────────

    #[test]
    fn cache_key_same_inputs_match() {
        let opts = QueryOptions::default();
        let k1 = QueryCache::make_key("text", "hello", &opts, true);
        let k2 = QueryCache::make_key("text", "hello", &opts, true);
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_differs_by_query() {
        let opts = QueryOptions::default();
        let k1 = QueryCache::make_key("text", "hello", &opts, true);
        let k2 = QueryCache::make_key("text", "world", &opts, true);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_by_llm_flag() {
        let opts = QueryOptions::default();
        let k1 = QueryCache::make_key("text", "hello", &opts, true);
        let k2 = QueryCache::make_key("text", "hello", &opts, false);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_differs_by_prefix() {
        let opts = QueryOptions::default();
        let k1 = QueryCache::make_key("text", "1", &opts, true);
        let k2 = QueryCache::make_key("node", "1", &opts, true);
        assert_ne!(k1, k2);
    }

    // ── merge helpers ─────────────────────────────────────────────────────────

    #[test]
    fn merge_context_deduplicates() {
        let make_node = |id: u32| NodeInfo {
            id, external_id: String::new(), name: id.to_string(), node_type: "T".into(),
            weight: 0.5, props: serde_json::Value::Null,
            full_context: String::new(), embed_context: None, image_url: None,
        };
        let mut base = vec![make_node(1), make_node(2)];
        QueryEngine::merge_context(&mut base, vec![make_node(2), make_node(3)]);
        assert_eq!(base.len(), 3);
        assert!(base.iter().any(|n| n.id == 3));
    }

    #[test]
    fn merge_edges_deduplicates() {
        let mut base: Vec<(u32, u32, f32, String)> = vec![(1, 2, 0.9, "rel".into())];
        QueryEngine::merge_edges(&mut base, vec![(1, 2, 0.9, "rel".into()), (2, 3, 0.5, "".into())]);
        assert_eq!(base.len(), 2);
    }
}
