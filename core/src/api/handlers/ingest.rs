use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum_valid::Valid;

use crate::{
    api::{
        dto::ingest::{IngestJsonReq, IngestResponse, IngestTextReq, ResolutionOverride},
        ApiError,
    },
    app_state::{run_merge, AppState},
    config::IngestConfig,
    entity_resolution::{resolve, resolve_in_batch, ResolutionConfig, ResolutionMode, ResolveResult},
    graph::csr::{EdgeInfo, NodeInfo},
    vector::hnsw::normalise,
};

/// Max texts per embed HTTP request.
/// Keeps requests within timeout even for large imports.
/// Python plugin internally batches by EMBED_BATCH_SIZE (default 32).
const EMBED_CHUNK: usize = 256;

// ── shared resolution helper ──────────────────────────────────────────────────

pub fn build_resolution_cfg(
    defaults:  &IngestConfig,
    override_: Option<&ResolutionOverride>,
) -> ResolutionConfig {
    let mode = override_
        .and_then(|o| o.mode.as_deref())
        .unwrap_or(&defaults.resolution_mode);

    ResolutionConfig {
        mode: match mode {
            "none" => ResolutionMode::None,
            _      => ResolutionMode::Embedding,
        },
        threshold:  override_.and_then(|o| o.threshold).unwrap_or(defaults.resolution_threshold),
        match_type: override_.and_then(|o| o.match_type).unwrap_or(defaults.resolution_match_type),
    }
}

// ── /ingest/text ─────────────────────────────────────────────────────────────

/// Picks the text to embed for a node.
/// Priority: embed_context → full_context → name.
fn node_embed_text(entity: &serde_json::Value) -> String {
    for key in &["embed_context", "full_context", "name"] {
        let v = entity[key].as_str().unwrap_or("").trim();
        if !v.is_empty() { return v.to_string(); }
    }
    String::new()
}

/// Picks the text to embed for an edge.
/// Priority: embed_context → full_context → type.
fn edge_embed_text(relation: &serde_json::Value) -> String {
    for key in &["embed_context", "full_context", "type"] {
        let v = relation[key].as_str().unwrap_or("").trim();
        if !v.is_empty() { return v.to_string(); }
    }
    String::new()
}

/// POST /ingest/text — raw text → extract → entity resolution → delta
pub async fn ingest_text(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<IngestTextReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.ingest_total.inc();

    if !s.plugin.check_ready().await {
        s.metrics.ingest_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let extracted = s.plugin.extract(&req.text, req.hints).await
        .map_err(|e| { s.metrics.ingest_failed_total.inc(); ApiError::plugin("extract", e) })?;

    let embed_texts: Vec<String> = extracted.entities
        .iter()
        .map(node_embed_text)
        .collect();

    // keep names for response reporting (still used for id_map key lookup)
    let names: Vec<String> = extracted.entities
        .iter()
        .filter_map(|e| e["name"].as_str().map(|s| s.to_string()))
        .collect();

    if names.is_empty() {
        return Ok((StatusCode::OK, Json(IngestResponse {
            ingested:        0,
            new_nodes:       0,
            resolved:        0,
            delta_size:      s.delta.size(),
            merge_triggered: false,
            message:         Some("no entities found".into()),
        })));
    }

    // embed nodes and edges concurrently
    let rel_embed_texts: Vec<String> = extracted.relations
        .iter()
        .map(edge_embed_text)
        .filter(|t| !t.is_empty())
        .collect();

    let (vecs_result, rel_vecs_result) = tokio::join!(
        s.plugin.embed_chunked(embed_texts, EMBED_CHUNK),
        async {
            if rel_embed_texts.is_empty() {
                Ok(vec![])
            } else {
                s.plugin.embed_chunked(rel_embed_texts, EMBED_CHUNK).await
            }
        }
    );
    let vecs     = vecs_result    .map_err(|e| { s.metrics.ingest_failed_total.inc(); ApiError::plugin("embed", e) })?;
    let rel_vecs = rel_vecs_result.map_err(|e| { s.metrics.ingest_failed_total.inc(); ApiError::plugin("embed relations", e) })?;

    let res_cfg = build_resolution_cfg(&s.cfg.ingest, req.resolution.as_ref());

    // ── entity resolution (read-only snapshot) ────────────────────────────
    let graph      = s.graph.read().await;
    let hnsw       = s.hnsw.read().await;
    let delta_snap = s.delta.read();

    let mut id_map:        std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut to_add:        Vec<(NodeInfo, Vec<f32>)> = Vec::new();
    let mut resolved_count = 0usize;

    for (entity, mut vec) in extracted.entities.iter().zip(vecs) {
        let str_id        = entity["id"].as_str().unwrap_or("").to_string();
        let name          = entity["name"].as_str().unwrap_or("").to_string();
        let ent_type      = entity["type"].as_str().unwrap_or("Entity");
        let full_context  = entity["full_context"].as_str().unwrap_or("").to_string();
        let embed_context = entity["embed_context"].as_str()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string());

        let _ = normalise(&mut vec);

        let resolved_id = match resolve(ent_type, &vec, &graph, &delta_snap, &hnsw, &res_cfg) {
            ResolveResult::Existing(id) => {
                tracing::info!("entity_resolution: '{}' → existing id={}", name, id);
                resolved_count += 1;
                id
            }
            ResolveResult::New => {
                if let Some(id) = resolve_in_batch(ent_type, &vec, &to_add, &res_cfg) {
                    tracing::info!("entity_resolution: '{}' → batch dedup id={}", name, id);
                    resolved_count += 1;
                    id
                } else {
                    let new_id = s.delta.alloc_node_id()
                        .map_err(|e| ApiError::Internal(e.to_string()))?;
                    to_add.push((NodeInfo {
                        id:           new_id,
                        external_id:  if str_id.is_empty() {
                                          format!("{}:{}", ent_type, name)
                                      } else {
                                          str_id.clone()
                                      },
                        name:         name.clone(),
                        node_type:    ent_type.to_string(),
                        weight:       0.0,
                        props:        entity["props"].clone(),
                        full_context,
                        embed_context,
                        image_url:    None,
                    }, vec));
                    new_id
                }
            }
        };
        id_map.insert(str_id, resolved_id);
    }

    drop(delta_snap);
    drop(hnsw);
    drop(graph);

    // ── commit to delta (batch — one WAL fsync each) ─────────────────────
    let new_node_count = to_add.len();
    s.delta.add_nodes_batch(to_add);

    let mut edges_to_add: Vec<(EdgeInfo, Vec<f32>)> = Vec::new();
    let mut rel_vec_idx = 0usize;
    for rel in &extracted.relations {
        let from_str = rel["from"].as_str().unwrap_or("");
        let to_str   = rel["to"].as_str().unwrap_or("");
        let et = edge_embed_text(rel);
        let edge_vec = if et.is_empty() {
            vec![]
        } else {
            let v = rel_vecs.get(rel_vec_idx).cloned().unwrap_or_default();
            rel_vec_idx += 1;
            v
        };
        if let (Some(&from), Some(&to)) = (id_map.get(from_str), id_map.get(to_str)) {
            edges_to_add.push((EdgeInfo {
                from,
                to,
                edge_type:     rel["type"].as_str().unwrap_or("").to_string(),
                weight:        rel["weight"].as_f64().unwrap_or(1.0) as f32,
                full_context:  rel["full_context"].as_str().unwrap_or("").to_string(),
                embed_context: rel["embed_context"].as_str()
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.to_string()),
                edge_id:       s.delta.alloc_edge_id(),
            }, edge_vec));
        }
    }
    s.delta.add_edges_batch(edges_to_add);

    let delta_size      = s.delta.size();
    let merge_triggered = s.delta.needs_merge();
    s.metrics.delta_size.set(delta_size as u64);
    s.metrics.nodes_ingested_total.add(new_node_count as u64);
    s.metrics.ingest_latency.observe(t.elapsed().as_millis() as u64);

    if merge_triggered {
        let s2 = s.clone();
        tokio::spawn(async move { run_merge(s2).await });
    }

    Ok((StatusCode::OK, Json(IngestResponse {
        ingested:   names.len(),
        new_nodes:  new_node_count,
        resolved:   resolved_count,
        delta_size,
        merge_triggered,
        message:    None,
    })))
}

// ── /ingest/json ──────────────────────────────────────────────────────────────

/// POST /ingest/json — pre-structured payload → entity resolution → delta
pub async fn ingest_json(
    State(s): State<AppState>,
    Valid(Json(req)): Valid<Json<IngestJsonReq>>,
) -> Result<impl IntoResponse, ApiError> {
    let t = std::time::Instant::now();
    s.metrics.ingest_total.inc();

    if !s.plugin.check_ready().await {
        s.metrics.ingest_failed_total.inc();
        return Err(ApiError::ServiceUnavailable("plugin not available".into()));
    }

    let payload = serde_json::json!({
        "entities":  req.entities,
        "relations": req.relations,
    });

    let partial_graph = crate::graph::builder::from_json_payload(&payload)
        .map_err(|e| ApiError::unprocessable(e.to_string()))?;

    let names: Vec<String> = partial_graph.nodes.iter().map(|n| n.name.clone()).collect();
    if names.is_empty() {
        return Ok((StatusCode::OK, Json(IngestResponse {
            ingested:        0,
            new_nodes:       0,
            resolved:        0,
            delta_size:      s.delta.size(),
            merge_triggered: false,
            message:         Some("no entities".into()),
        })));
    }

    // Separate nodes into text-embeddable and image-embeddable groups.
    // Image nodes go through /embed/image (vision → caption → embed);
    // all others go through /embed/text in a single chunked batch.
    let mut node_embed_texts: Vec<String>          = Vec::with_capacity(partial_graph.nodes.len());
    let mut text_node_indices: Vec<usize>          = Vec::new();
    let mut image_node_indices: Vec<usize>         = Vec::new();

    for (i, n) in partial_graph.nodes.iter().enumerate() {
        if n.image_url.is_some() {
            image_node_indices.push(i);
        } else {
            node_embed_texts.push(
                n.embed_context.as_deref()
                    .filter(|s| !s.is_empty())
                    .or_else(|| if n.full_context.is_empty() { None } else { Some(n.full_context.as_str()) })
                    .unwrap_or(&n.name)
                    .to_string()
            );
            text_node_indices.push(i);
        }
    }

    let rel_embed_texts: Vec<String> = req.relations.iter()
        .map(edge_embed_text)
        .filter(|t| !t.is_empty())
        .collect();

    // Embed text nodes and relations concurrently; image nodes sequentially after.
    let (text_vecs_result, rel_vecs_result) = tokio::join!(
        async {
            if node_embed_texts.is_empty() { Ok(vec![]) }
            else { s.plugin.embed_chunked(node_embed_texts, EMBED_CHUNK).await }
        },
        async {
            if rel_embed_texts.is_empty() { Ok(vec![]) }
            else { s.plugin.embed_chunked(rel_embed_texts, EMBED_CHUNK).await }
        }
    );
    let text_vecs = text_vecs_result .map_err(|e| { s.metrics.ingest_failed_total.inc(); ApiError::plugin("embed", e) })?;
    let rel_vecs  = rel_vecs_result  .map_err(|e| { s.metrics.ingest_failed_total.inc(); ApiError::plugin("embed relations", e) })?;

    // If auto_store is enabled, persist each image (base64 or URL) to the image
    // plugin before embedding. The returned stable URL replaces the original input.
    // stored_image_urls[i] maps image_node_indices[i] → stable URL to use for embed.
    let mut stored_image_urls: Vec<String> = Vec::with_capacity(image_node_indices.len());
    for &ni in &image_node_indices {
        let raw_url = partial_graph.nodes[ni].image_url.as_deref().unwrap_or("");
        let embed_url = if s.cfg.image.auto_store {
            match s.plugin.store_image(raw_url).await {
                Ok(stored) => stored,
                Err(e) => {
                    tracing::warn!("auto_store failed for node {}: {e} — using original url", ni);
                    raw_url.to_string()
                }
            }
        } else {
            raw_url.to_string()
        };
        stored_image_urls.push(embed_url);
    }

    // Embed image nodes (one request each — images aren't batchable cheaply).
    let mut image_vecs: Vec<Vec<f32>> = Vec::with_capacity(image_node_indices.len());
    for (slot, &ni) in image_node_indices.iter().enumerate() {
        let _ = ni; // ni accessed via stored_image_urls[slot]
        let url = &stored_image_urls[slot];
        let v = s.plugin.embed_image(url).await
            .map_err(|e| { s.metrics.ingest_failed_total.inc(); ApiError::plugin("embed_image", e) })?;
        image_vecs.push(v);
    }

    // Merge back into a single ordered vec indexed by original node position.
    let mut vecs: Vec<Vec<f32>> = vec![vec![]; partial_graph.nodes.len()];
    for (slot, &ni) in text_node_indices.iter().enumerate() {
        vecs[ni] = text_vecs.get(slot).cloned().unwrap_or_default();
    }
    for (slot, &ni) in image_node_indices.iter().enumerate() {
        vecs[ni] = image_vecs.get(slot).cloned().unwrap_or_default();
    }

    let res_cfg = build_resolution_cfg(&s.cfg.ingest, req.resolution.as_ref());

    // ── entity resolution ─────────────────────────────────────────────────
    let graph      = s.graph.read().await;
    let hnsw       = s.hnsw.read().await;
    let delta_snap = s.delta.read();

    let mut idx_to_id:     Vec<u32> = Vec::with_capacity(names.len());
    let mut to_add:        Vec<(NodeInfo, Vec<f32>)> = Vec::new();
    let mut resolved_count = 0usize;

    for (_current_node_idx, (node, mut vec)) in partial_graph.nodes.iter().zip(vecs).enumerate() {
        let _ = normalise(&mut vec);

        let resolved_id = match resolve(&node.node_type, &vec, &graph, &delta_snap, &hnsw, &res_cfg) {
            ResolveResult::Existing(id) => {
                tracing::info!("entity_resolution: '{}' → existing id={}", node.name, id);
                resolved_count += 1;
                id
            }
            ResolveResult::New => {
                if let Some(id) = resolve_in_batch(&node.node_type, &vec, &to_add, &res_cfg) {
                    tracing::info!("entity_resolution: '{}' → batch dedup id={}", node.name, id);
                    resolved_count += 1;
                    id
                } else {
                    let new_id = s.delta.alloc_node_id()
                        .map_err(|e| ApiError::Internal(e.to_string()))?;
                    let mut n = node.clone();
                    n.id = new_id;
                    if n.external_id.is_empty() {
                        n.external_id = format!("{}:{}", n.node_type, n.name);
                    }
                    // Replace image_url with the stable stored URL when auto_store ran.
                    if s.cfg.image.auto_store {
                        if let Some(img_slot) = image_node_indices.iter().position(|&x| x == _current_node_idx) {
                            if let Some(stable_url) = stored_image_urls.get(img_slot) {
                                n.image_url = Some(stable_url.clone());
                            }
                        }
                    }
                    to_add.push((n, vec));
                    new_id
                }
            }
        };
        idx_to_id.push(resolved_id);
    }

    drop(delta_snap);
    drop(hnsw);
    drop(graph);

    let new_node_count = to_add.len();
    s.delta.add_nodes_batch(to_add);

    // Rebuild str_id → partial_graph_idx mapping to resolve relation endpoints.
    let str_to_partial: std::collections::HashMap<String, u32> = req.entities.iter()
        .enumerate()
        .filter_map(|(i, e)| e["id"].as_str().map(|id| (id.to_string(), i as u32)))
        .collect();

    let mut edges_to_add: Vec<(EdgeInfo, Vec<f32>)> = Vec::new();
    let mut rel_vec_idx = 0usize;
    for rel in &req.relations {
        let from_str = rel["from"].as_str().unwrap_or("");
        let to_str   = rel["to"].as_str().unwrap_or("");
        let et = edge_embed_text(rel);
        let edge_vec = if et.is_empty() {
            vec![]
        } else {
            let v = rel_vecs.get(rel_vec_idx).cloned().unwrap_or_default();
            rel_vec_idx += 1;
            v
        };
        if let (Some(&pi_from), Some(&pi_to)) = (str_to_partial.get(from_str), str_to_partial.get(to_str)) {
            if let (Some(&from), Some(&to)) = (idx_to_id.get(pi_from as usize), idx_to_id.get(pi_to as usize)) {
                edges_to_add.push((EdgeInfo {
                    from,
                    to,
                    edge_type:     rel["type"].as_str().unwrap_or("").to_string(),
                    weight:        rel["weight"].as_f64().unwrap_or(1.0) as f32,
                    full_context:  rel["full_context"].as_str().unwrap_or("").to_string(),
                    embed_context: rel["embed_context"].as_str()
                        .filter(|s| !s.trim().is_empty())
                        .map(|s| s.to_string()),
                    edge_id:       s.delta.alloc_edge_id(),
                }, edge_vec));
            }
        }
    }
    s.delta.add_edges_batch(edges_to_add);

    let delta_size      = s.delta.size();
    let merge_triggered = s.delta.needs_merge();
    s.metrics.delta_size.set(delta_size as u64);
    s.metrics.nodes_ingested_total.add(new_node_count as u64);
    s.metrics.ingest_latency.observe(t.elapsed().as_millis() as u64);

    if merge_triggered {
        let s2 = s.clone();
        tokio::spawn(async move { run_merge(s2).await });
    }

    Ok((StatusCode::OK, Json(IngestResponse {
        ingested:   names.len(),
        new_nodes:  new_node_count,
        resolved:   resolved_count,
        delta_size,
        merge_triggered,
        message:    None,
    })))
}
