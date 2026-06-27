use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum_valid::Valid;

use crate::{
    api::{
        dto::query::{QueryImageReq, QueryMultihopReq, QueryNodeReq, QueryReq, QueryResponse, QueryTextReq, QueryVectorReq},
        ApiError,
    },
    app_state::AppState,
    query::{QueryOptions, QueryResult, ScoringWeights},
    vector::hnsw::normalise,
};

// ── shared helpers ────────────────────────────────────────────────────────────

fn resolve_weights(mode: Option<&str>) -> (ScoringWeights, bool) {
    match mode {
        Some("semantic")     => (ScoringWeights::semantic_search(), false),
        Some("relationship") => (ScoringWeights::relationship(),    true),
        Some("entity")       => (ScoringWeights::entity_lookup(),   false),
        _                    => (ScoringWeights::balanced(),         false),
    }
}

fn record_metrics(s: &AppState, t: std::time::Instant, result: &QueryResult) {
    let total_ms = t.elapsed().as_millis() as u64;
    s.metrics.query_latency.observe(total_ms);
    s.metrics.embed_latency.observe(result.stats.embed_ms);
    s.metrics.llm_latency.observe(result.stats.llm_ms);
    if result.stats.cache_hit {
        s.metrics.cache_hits.inc();
    } else {
        s.metrics.cache_misses.inc();
    }
}

// ── POST /query/text ──────────────────────────────────────────────────────────

pub async fn handle_query_text(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<QueryTextReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.queries_total.inc();

    // Embed is always needed for text; check plugin unconditionally
    if !s.plugin.check_ready().await {
        s.metrics.queries_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let llm_generate = req.pipeline.as_ref().map_or(true, |p| p.llm_generate);
    let (weights, bidirectional) = resolve_weights(req.mode.as_deref());
    let qcfg = &s.cfg.query;
    let opts = req.options.unwrap_or(QueryOptions {
        weights,
        bidirectional,
        hnsw_k:            qcfg.hnsw_k,
        bfs_depth:         qcfg.bfs_depth,
        bfs_max_nodes:     qcfg.bfs_max_nodes,
        context_top_n:     qcfg.context_top_n,
        context_min_score: qcfg.context_min_score,
    });

    let hints  = req.pipeline.as_ref().and_then(|p| p.hints.clone());
    let result = s.engine.query_text(&req.query, opts, llm_generate, hints).await.map_err(|e| {
        s.metrics.queries_failed_total.inc();
        ApiError::Internal(format!("query failed: {e}"))
    })?;

    record_metrics(&s, t, &result);
    let profile = req.response.unwrap_or_default();
    Ok((StatusCode::OK, Json(QueryResponse::from_result(result, &profile))))
}

// ── POST /query/vector ────────────────────────────────────────────────────────

pub async fn handle_query_vector(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<QueryVectorReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.queries_total.inc();

    let llm_generate = req.pipeline.as_ref().map_or(true, |p| p.llm_generate);
    // Plugin needed only for LLM generation — embed step is skipped
    if llm_generate && !s.plugin.check_ready().await {
        s.metrics.queries_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let (weights, bidirectional) = resolve_weights(req.mode.as_deref());
    let qcfg = &s.cfg.query;
    let opts = req.options.unwrap_or(QueryOptions {
        weights,
        bidirectional,
        hnsw_k:            qcfg.hnsw_k,
        bfs_depth:         qcfg.bfs_depth,
        bfs_max_nodes:     qcfg.bfs_max_nodes,
        context_top_n:     qcfg.context_top_n,
        context_min_score: qcfg.context_min_score,
    });

    let q      = req.pipeline.as_ref().and_then(|p| p.prompt.as_deref()).unwrap_or("");
    let hints  = req.pipeline.as_ref().and_then(|p| p.hints.clone());
    let result = s.engine.query_with_vector(req.vector, q, opts, llm_generate, hints).await.map_err(|e| {
        s.metrics.queries_failed_total.inc();
        ApiError::Internal(format!("query failed: {e}"))
    })?;

    record_metrics(&s, t, &result);
    let profile = req.response.unwrap_or_default();
    Ok((StatusCode::OK, Json(QueryResponse::from_result(result, &profile))))
}

// ── POST /query/node ──────────────────────────────────────────────────────────

pub async fn handle_query_node(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<QueryNodeReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.queries_total.inc();

    // Node query = pure graph exploration, no plugin needed
    let (weights, bidirectional) = resolve_weights(req.mode.as_deref());
    let qcfg = &s.cfg.query;
    let opts = req.options.unwrap_or(QueryOptions {
        weights,
        bidirectional,
        hnsw_k:            qcfg.hnsw_k,
        bfs_depth:         qcfg.bfs_depth,
        bfs_max_nodes:     qcfg.bfs_max_nodes,
        context_top_n:     qcfg.context_top_n,
        context_min_score: 0.0,
    });

    let result = s.engine.query_from_node(req.node_id, "", opts, false, None).await.map_err(|e| {
        s.metrics.queries_failed_total.inc();
        if e.to_string().contains("not found") {
            return ApiError::NotFound(e.to_string());
        }
        ApiError::Internal(format!("query failed: {e}"))
    })?;

    record_metrics(&s, t, &result);
    let profile = req.response.unwrap_or_default();
    Ok((StatusCode::OK, Json(QueryResponse::from_result(result, &profile))))
}

// ── POST /query/multihop (Phase 10) ──────────────────────────────────────────

pub async fn handle_query_multihop(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<QueryMultihopReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.queries_total.inc();

    if !s.plugin.check_ready().await {
        s.metrics.queries_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let (weights, bidirectional) = resolve_weights(req.mode.as_deref());
    let qcfg = &s.cfg.query;
    let opts = req.options.unwrap_or(QueryOptions {
        weights,
        bidirectional,
        hnsw_k:            qcfg.hnsw_k,
        bfs_depth:         qcfg.bfs_depth,
        bfs_max_nodes:     qcfg.bfs_max_nodes,
        context_top_n:     qcfg.context_top_n,
        context_min_score: qcfg.context_min_score,
    });

    let result = s.engine.query_multihop(&req.query, opts, req.max_hops, None).await.map_err(|e| {
        s.metrics.queries_failed_total.inc();
        ApiError::Internal(format!("multihop query failed: {e}"))
    })?;

    record_metrics(&s, t, &result);
    let profile = req.response.unwrap_or_default();
    Ok((StatusCode::OK, Json(QueryResponse::from_result(result, &profile))))
}

// ── POST /query (legacy) ──────────────────────────────────────────────────────

pub async fn handle_query(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<QueryReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.queries_total.inc();

    let needs_embed    = req.vector.is_none();
    let needs_generate = req.pipeline.as_ref().map_or(true, |p| p.llm_generate);
    if (needs_embed || needs_generate) && !s.plugin.check_ready().await {
        s.metrics.queries_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let (weights, bidirectional) = resolve_weights(req.mode.as_deref());
    let qcfg = &s.cfg.query;
    let opts = req.options.unwrap_or(QueryOptions {
        weights,
        bidirectional,
        hnsw_k:            qcfg.hnsw_k,
        bfs_depth:         qcfg.bfs_depth,
        bfs_max_nodes:     qcfg.bfs_max_nodes,
        context_top_n:     qcfg.context_top_n,
        context_min_score: qcfg.context_min_score,
    });

    let q     = req.query.as_deref().unwrap_or("");
    let hints = req.pipeline.as_ref().and_then(|p| p.hints.clone());
    let result = if let Some(vec) = req.vector {
        s.engine.query_with_vector(vec, q, opts, needs_generate, hints).await
    } else {
        s.engine.query_text(q, opts, needs_generate, hints).await
    }.map_err(|e| {
        s.metrics.queries_failed_total.inc();
        ApiError::Internal(format!("query failed: {e}"))
    })?;

    record_metrics(&s, t, &result);
    let profile = req.response.unwrap_or_default();
    Ok((StatusCode::OK, Json(QueryResponse::from_result(result, &profile))))
}

// ── POST /query/image ─────────────────────────────────────────────────────────

/// Query the graph using an image as input.
/// The plugin generates a caption via Vision LLM, embeds it in the shared
/// text vector space, then runs the standard HNSW → BFS → score → LLM pipeline.
pub async fn handle_query_image(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<QueryImageReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.queries_total.inc();

    if !s.plugin.check_ready().await {
        s.metrics.queries_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let mut vec = s.plugin.embed_image(&req.image_url).await.map_err(|e| {
        s.metrics.queries_failed_total.inc();
        ApiError::plugin("embed_image", e)
    })?;
    let _ = normalise(&mut vec);

    let llm_generate = req.pipeline.as_ref().map_or(true, |p| p.llm_generate);
    let prompt       = req.pipeline.as_ref().and_then(|p| p.prompt.clone()).unwrap_or_default();
    let hints        = req.pipeline.as_ref().and_then(|p| p.hints.clone());

    let (weights, bidirectional) = resolve_weights(req.mode.as_deref());
    let qcfg = &s.cfg.query;
    let opts = req.options.unwrap_or(QueryOptions {
        weights,
        bidirectional,
        hnsw_k:            qcfg.hnsw_k,
        bfs_depth:         qcfg.bfs_depth,
        bfs_max_nodes:     qcfg.bfs_max_nodes,
        context_top_n:     qcfg.context_top_n,
        context_min_score: qcfg.context_min_score,
    });

    let result = s.engine.query_with_vector(vec, &prompt, opts, llm_generate, hints)
        .await
        .map_err(|e| {
            s.metrics.queries_failed_total.inc();
            ApiError::Internal(format!("query failed: {e}"))
        })?;

    record_metrics(&s, t, &result);
    let profile = req.response.unwrap_or_default();
    Ok((StatusCode::OK, Json(QueryResponse::from_result(result, &profile))))
}
