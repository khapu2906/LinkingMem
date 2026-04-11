use axum::{extract::{Path, Query, State}, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};

use crate::{api::ApiError, app_state::{AppState, run_merge}};

/// GET /health
pub async fn health(State(s): State<AppState>) -> impl IntoResponse {
    let graph = s.graph.read().await;
    s.metrics.graph_nodes.set(graph.num_nodes() as u64);
    s.metrics.graph_edges.set(graph.num_edges() as u64);
    s.metrics.delta_size.set(s.delta.size() as u64);
    drop(graph);

    let plugin_ok    = s.plugin.check_ready().await;
    let uptime_secs  = s.started_at.elapsed().as_secs();

    Json(serde_json::json!({
        "status":       "ok",
        "uptime_secs":  uptime_secs,
        "plugin":       { "reachable": plugin_ok },
        "metrics":      s.metrics.summary_json(),
    }))
}

/// GET /metrics — Prometheus text format
pub async fn metrics_endpoint(State(s): State<AppState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        s.metrics.render_prometheus(),
    )
}

/// POST /delta/merge — force an immediate merge (admin endpoint)
pub async fn force_merge(State(s): State<AppState>) -> impl IntoResponse {
    run_merge(s).await;
    Json(serde_json::json!({ "status": "merge complete" }))
}

// ── graph stats ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct GraphStats {
    pub nodes:              usize,
    pub edges:              usize,
    pub delta_nodes:        usize,
    pub delta_edges:        usize,
    pub delta_pending_merge: bool,
}

/// GET /graph/stats — current graph + delta counts as JSON
pub async fn graph_stats(State(s): State<AppState>) -> impl IntoResponse {
    let graph = s.graph.read().await;
    Json(GraphStats {
        nodes:               graph.num_nodes(),
        edges:               graph.num_edges(),
        delta_nodes:         s.delta.read().nodes_len(),
        delta_edges:         s.delta.read().edges_len(),
        delta_pending_merge: s.delta.needs_merge(),
    })
}

// ── node lookup ───────────────────────────────────────────────────────────────

/// GET /nodes/:id — fetch a single node by numeric ID
pub async fn get_node(
    State(s): State<AppState>,
    Path(id): Path<u32>,
) -> Result<impl IntoResponse, ApiError> {
    let graph = s.graph.read().await;
    if let Some(node) = graph.nodes.get(id as usize) {
        return Ok(Json(node.clone()));
    }
    drop(graph);

    // fall back to delta buffer
    let delta = s.delta.read();
    if let Some(node) = delta.get_node(id) {
        return Ok(Json(node.clone()));
    }

    Err(ApiError::NotFound(format!("node {id} not found")))
}

// ── node search ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct NodeSearchParams {
    /// Substring match on node name (case-insensitive)
    pub q:          Option<String>,
    /// Filter by exact entity type
    #[serde(rename = "type")]
    pub node_type:  Option<String>,
    /// Return only nodes connected to this node ID
    pub related_to: Option<u32>,
    /// Edge direction for related_to: "out" (default) | "in" | "both"
    pub direction:  Option<String>,
    /// Filter by edge type (use with related_to for fast lookup, or alone for full scan)
    pub edge_type:  Option<String>,
    /// Max results to return (default 50, max 500)
    pub limit:      Option<usize>,
}

/// GET /nodes?q=alice&type=Person&limit=20
/// GET /nodes?related_to=5&direction=out&edge_type=works_at
pub async fn search_nodes(
    State(s): State<AppState>,
    Query(params): Query<NodeSearchParams>,
) -> Result<impl IntoResponse, ApiError> {
    let limit    = params.limit.unwrap_or(50).min(500);
    let q_lower  = params.q.as_deref().map(|s| s.to_lowercase());
    let etype    = params.node_type.as_deref();
    let et_edge  = params.edge_type.as_deref();

    let graph = s.graph.read().await;

    // ── Branch 1: related_to — collect neighbours ─────────────────────────────
    if let Some(root_id) = params.related_to {
        if root_id as usize >= graph.num_nodes() {
            return Err(ApiError::NotFound(format!("node {root_id} not found")));
        }

        let direction = params.direction.as_deref().unwrap_or("out");
        let mut candidate_ids: Vec<u32> = Vec::new();

        match direction {
            "in" => {
                for &from_id in graph.rev_neighbors(root_id) {
                    if edge_type_matches(&graph, from_id, root_id, et_edge) {
                        candidate_ids.push(from_id);
                    }
                }
            }
            "both" => {
                // outgoing
                let nbrs  = graph.neighbors(root_id);
                let types = graph.neighbor_types(root_id);
                for (i, &nb) in nbrs.iter().enumerate() {
                    if et_edge.map_or(true, |et| types[i].eq_ignore_ascii_case(et)) {
                        candidate_ids.push(nb);
                    }
                }
                // incoming (deduplicated)
                let out_set: std::collections::HashSet<u32> = candidate_ids.iter().copied().collect();
                for &from_id in graph.rev_neighbors(root_id) {
                    if !out_set.contains(&from_id) && edge_type_matches(&graph, from_id, root_id, et_edge) {
                        candidate_ids.push(from_id);
                    }
                }
            }
            _ => {
                // "out" — default
                let nbrs  = graph.neighbors(root_id);
                let types = graph.neighbor_types(root_id);
                for (i, &nb) in nbrs.iter().enumerate() {
                    if et_edge.map_or(true, |et| types[i].eq_ignore_ascii_case(et)) {
                        candidate_ids.push(nb);
                    }
                }
            }
        }

        let mut results: Vec<_> = candidate_ids.iter()
            .filter_map(|&id| graph.nodes.get(id as usize))
            .filter(|n| {
                q_lower.as_deref().map_or(true, |q| n.name.to_lowercase().contains(q)) &&
                etype.map_or(true, |t| n.node_type.eq_ignore_ascii_case(t))
            })
            .take(limit)
            .cloned()
            .collect();

        drop(graph);

        // delta: scan new_edges for related_to
        if results.len() < limit {
            let delta = s.delta.read();
            let room  = limit - results.len();
            let from_delta: Vec<_> = match direction {
                "in"   => delta.new_edges.iter().filter(|e| e.to == root_id).map(|e| e.from).collect(),
                "both" => {
                    let mut ids: Vec<u32> = delta.new_edges.iter().filter(|e| e.from == root_id).map(|e| e.to).collect();
                    let out_set: std::collections::HashSet<u32> = ids.iter().copied().collect();
                    for e in delta.new_edges.iter().filter(|e| e.to == root_id && !out_set.contains(&e.from)) {
                        ids.push(e.from);
                    }
                    ids
                }
                _ => delta.new_edges.iter().filter(|e| e.from == root_id).map(|e| e.to).collect(),
            };

            let extra: Vec<_> = from_delta.iter()
                .filter_map(|&id| delta.get_node(id).or_else(|| {
                    // node might be in main graph already (re-read after drop is safe here since we use RwLock)
                    None // just skip — main graph nodes were already collected above
                }))
                .filter(|n| {
                    et_edge.map_or(true, |et| {
                        delta.new_edges.iter().any(|e| {
                            (e.from == root_id && e.to == n.id || e.to == root_id && e.from == n.id)
                            && e.edge_type.eq_ignore_ascii_case(et)
                        })
                    }) &&
                    q_lower.as_deref().map_or(true, |q| n.name.to_lowercase().contains(q)) &&
                    etype.map_or(true, |t| n.node_type.eq_ignore_ascii_case(t))
                })
                .take(room)
                .cloned()
                .collect();
            results.extend(extra);
        }

        let count = results.len();
        return Ok(Json(serde_json::json!({ "nodes": results, "count": count })));
    }

    // ── Branch 2: edge_type scan (no related_to) ─────────────────────────────
    if let Some(et) = et_edge {
        let mut seen = std::collections::HashSet::new();
        let mut results: Vec<_> = (0..graph.num_nodes() as u32)
            .filter(|&nid| {
                graph.neighbor_types(nid).iter().any(|t| t.eq_ignore_ascii_case(et))
            })
            .filter_map(|nid| graph.nodes.get(nid as usize))
            .filter(|n| {
                q_lower.as_deref().map_or(true, |q| n.name.to_lowercase().contains(q)) &&
                etype.map_or(true, |t| n.node_type.eq_ignore_ascii_case(t)) &&
                seen.insert(n.id)
            })
            .take(limit)
            .cloned()
            .collect();

        drop(graph);

        if results.len() < limit {
            let delta = s.delta.read();
            let room  = limit - results.len();
            let extra: Vec<_> = delta.new_nodes.iter()
                .filter(|n| {
                    !seen.contains(&n.id) &&
                    delta.new_edges.iter().any(|e| e.from == n.id && e.edge_type.eq_ignore_ascii_case(et)) &&
                    q_lower.as_deref().map_or(true, |q| n.name.to_lowercase().contains(q)) &&
                    etype.map_or(true, |t| n.node_type.eq_ignore_ascii_case(t))
                })
                .take(room)
                .cloned()
                .collect();
            results.extend(extra);
        }

        let count = results.len();
        return Ok(Json(serde_json::json!({ "nodes": results, "count": count })));
    }

    // ── Branch 3: plain name/type search (original behavior) ─────────────────
    let mut results: Vec<_> = graph.nodes.iter()
        .filter(|n| {
            q_lower.as_deref().map_or(true, |q| n.name.to_lowercase().contains(q)) &&
            etype.map_or(true, |t| n.node_type.eq_ignore_ascii_case(t))
        })
        .take(limit)
        .cloned()
        .collect();
    drop(graph);

    if results.len() < limit {
        let delta  = s.delta.read();
        let room   = limit - results.len();
        let from_delta: Vec<_> = delta.new_nodes.iter()
            .filter(|n| {
                q_lower.as_deref().map_or(true, |q| n.name.to_lowercase().contains(q)) &&
                etype.map_or(true, |t| n.node_type.eq_ignore_ascii_case(t))
            })
            .take(room)
            .cloned()
            .collect();
        results.extend(from_delta);
    }

    let count = results.len();
    Ok(Json(serde_json::json!({ "nodes": results, "count": count })))
}

/// Check if there is an edge from_id → to_id matching the optional edge_type filter.
fn edge_type_matches(graph: &crate::graph::csr::CsrGraph, from_id: u32, to_id: u32, et_filter: Option<&str>) -> bool {
    if et_filter.is_none() { return true; }
    let et = et_filter.unwrap();
    let nbrs  = graph.neighbors(from_id);
    let types = graph.neighbor_types(from_id);
    nbrs.iter().enumerate()
        .any(|(i, &nb)| nb == to_id && types[i].eq_ignore_ascii_case(et))
}

// ── edge search ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct EdgeSearchParams {
    /// Filter edges originating from this node ID
    pub from:  Option<u32>,
    /// Filter edges pointing to this node ID
    pub to:    Option<u32>,
    /// Filter by edge type (case-insensitive)
    #[serde(rename = "type")]
    pub edge_type: Option<String>,
    /// Max results (default 100, max 1000)
    pub limit: Option<usize>,
}

#[derive(Serialize, Clone)]
pub struct EdgeEntry {
    pub from_id:   u32,
    pub from_name: String,
    pub to_id:     u32,
    pub to_name:   String,
    pub weight:    f32,
    pub edge_type: String,
}

/// GET /edges?from=5&type=works_at
/// GET /edges?to=5
/// GET /edges?type=manages
pub async fn search_edges(
    State(s): State<AppState>,
    Query(params): Query<EdgeSearchParams>,
) -> Result<impl IntoResponse, ApiError> {
    if params.from.is_none() && params.to.is_none() && params.edge_type.is_none() {
        return Err(ApiError::BadRequest(
            "at least one filter required: from, to, or type".into()
        ));
    }

    let limit    = params.limit.unwrap_or(100).min(1_000);
    let et_filter = params.edge_type.as_deref();
    let graph    = s.graph.read().await;

    let node_name = |id: u32| -> String {
        graph.nodes.get(id as usize).map(|n| n.name.clone()).unwrap_or_default()
    };

    let mut results: Vec<EdgeEntry> = Vec::new();

    match (params.from, params.to) {
        (Some(from_id), None) => {
            if from_id as usize >= graph.num_nodes() {
                return Err(ApiError::NotFound(format!("node {from_id} not found")));
            }
            let nbrs  = graph.neighbors(from_id);
            let wts   = graph.neighbor_weights(from_id);
            let types = graph.neighbor_types(from_id);
            for i in 0..nbrs.len() {
                if results.len() >= limit { break; }
                if et_filter.map_or(true, |et| types[i].eq_ignore_ascii_case(et)) {
                    results.push(EdgeEntry {
                        from_id,
                        from_name: node_name(from_id),
                        to_id:     nbrs[i],
                        to_name:   node_name(nbrs[i]),
                        weight:    wts[i],
                        edge_type: types[i].clone(),
                    });
                }
            }
        }

        (None, Some(to_id)) => {
            if to_id as usize >= graph.num_nodes() {
                return Err(ApiError::NotFound(format!("node {to_id} not found")));
            }
            for &from_id in graph.rev_neighbors(to_id) {
                if results.len() >= limit { break; }
                let nbrs  = graph.neighbors(from_id);
                let wts   = graph.neighbor_weights(from_id);
                let types = graph.neighbor_types(from_id);
                if let Some(idx) = nbrs.iter().position(|&n| n == to_id) {
                    if et_filter.map_or(true, |et| types[idx].eq_ignore_ascii_case(et)) {
                        results.push(EdgeEntry {
                            from_id,
                            from_name: node_name(from_id),
                            to_id,
                            to_name:   node_name(to_id),
                            weight:    wts[idx],
                            edge_type: types[idx].clone(),
                        });
                    }
                }
            }
        }

        (Some(from_id), Some(to_id)) => {
            let nbrs  = graph.neighbors(from_id);
            let wts   = graph.neighbor_weights(from_id);
            let types = graph.neighbor_types(from_id);
            if let Some(idx) = nbrs.iter().position(|&n| n == to_id) {
                if et_filter.map_or(true, |et| types[idx].eq_ignore_ascii_case(et)) {
                    results.push(EdgeEntry {
                        from_id,
                        from_name: node_name(from_id),
                        to_id,
                        to_name:   node_name(to_id),
                        weight:    wts[idx],
                        edge_type: types[idx].clone(),
                    });
                }
            }
        }

        (None, None) => {
            // edge_type-only scan — O(E)
            let et = et_filter.unwrap(); // guaranteed Some by earlier check
            'outer: for nid in 0..graph.num_nodes() as u32 {
                let nbrs  = graph.neighbors(nid);
                let wts   = graph.neighbor_weights(nid);
                let types = graph.neighbor_types(nid);
                for i in 0..nbrs.len() {
                    if results.len() >= limit { break 'outer; }
                    if types[i].eq_ignore_ascii_case(et) {
                        results.push(EdgeEntry {
                            from_id:   nid,
                            from_name: node_name(nid),
                            to_id:     nbrs[i],
                            to_name:   node_name(nbrs[i]),
                            weight:    wts[i],
                            edge_type: types[i].clone(),
                        });
                    }
                }
            }
        }
    }

    drop(graph);

    // delta edges — acquire graph2 first (async), then delta (sync), never hold sync lock over await
    if results.len() < limit {
        let graph2  = s.graph.read().await;
        let delta   = s.delta.read();
        let resolve = |id: u32| -> String {
            graph2.nodes.get(id as usize)
                .map(|n| n.name.clone())
                .or_else(|| delta.get_node(id).map(|n| n.name.clone()))
                .unwrap_or_default()
        };

        for edge in delta.new_edges.iter() {
            if results.len() >= limit { break; }
            let matches_from = params.from.map_or(true, |f| edge.from == f);
            let matches_to   = params.to.map_or(true,   |t| edge.to   == t);
            let matches_et   = et_filter.map_or(true, |et| edge.edge_type.eq_ignore_ascii_case(et));
            if matches_from && matches_to && matches_et {
                results.push(EdgeEntry {
                    from_id:   edge.from,
                    from_name: resolve(edge.from),
                    to_id:     edge.to,
                    to_name:   resolve(edge.to),
                    weight:    edge.weight,
                    edge_type: edge.edge_type.clone(),
                });
            }
        }
    }

    let count = results.len();
    Ok(Json(serde_json::json!({ "edges": results, "count": count })))
}
