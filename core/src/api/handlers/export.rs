/// Export and Import handlers.
///
/// Endpoints:
///   GET  /export/graph?format=json    — download full graph as JSON attachment
///   GET  /export/graph?format=ndjson  — stream graph as newline-delimited JSON
///   POST /export/graph                — same as GET but format/options in JSON body
///   POST /import/graph                — import via JSON body
///   POST /import/graph/upload         — import via multipart file upload

use axum::{
    body::Body,
    extract::{Multipart, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use crate::{
    api::dto::export::{ExportEdge, ExportNode, ExportStats, GraphExport, NdjsonLine},
    app_state::AppState,
};

// ── query params / body ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ExportParams {
    /// `json` (default) or `ndjson`
    #[serde(default = "default_format")]
    pub format: String,
}

/// Body accepted by `POST /export/graph`.
#[derive(Debug, Deserialize, Default)]
pub struct ExportBody {
    /// `"json"` (default) or `"ndjson"`
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    "json".into()
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Collect nodes + edges from the live graph (base CSR) and pending delta.
async fn collect_export_data(s: &AppState) -> (Vec<ExportNode>, Vec<ExportEdge>) {
    let graph = s.graph.read().await.clone();

    // Build a name lookup: node_id → name
    let id_to_name: Vec<String> = graph.nodes.iter().map(|n| n.name.clone()).collect();

    // Nodes from CSR
    let mut nodes: Vec<ExportNode> = graph
        .nodes
        .iter()
        .map(|n| ExportNode {
            name: n.name.clone(),
            node_type: n.node_type.clone(),
            props: n.props.clone(),
            full_context: n.full_context.clone(),
            embed_context: n.embed_context.clone(),
        })
        .collect();

    // Edges from CSR
    let csr_edges = graph.all_edges();
    let mut edges: Vec<ExportEdge> = csr_edges
        .iter()
        .map(|e| ExportEdge {
            from: id_to_name
                .get(e.from as usize)
                .cloned()
                .unwrap_or_default(),
            to: id_to_name
                .get(e.to as usize)
                .cloned()
                .unwrap_or_default(),
            edge_type: e.edge_type.clone(),
            weight: e.weight,
            full_context: e.full_context.clone(),
            embed_context: e.embed_context.clone(),
        })
        .collect();

    // Include delta (not yet merged into CSR)
    {
        let delta = s.delta.read();

        // Delta nodes — IDs start after CSR node count
        let base_count = graph.num_nodes();
        let delta_id_to_name: Vec<String> = delta
            .new_nodes
            .iter()
            .map(|n| n.name.clone())
            .collect();

        for n in &delta.new_nodes {
            nodes.push(ExportNode {
                name: n.name.clone(),
                node_type: n.node_type.clone(),
                props: n.props.clone(),
                full_context: n.full_context.clone(),
                embed_context: n.embed_context.clone(),
            });
        }

        // Delta edges — resolve IDs using both CSR names and delta names
        let resolve_id = |id: u32| -> String {
            if (id as usize) < base_count {
                id_to_name.get(id as usize).cloned().unwrap_or_default()
            } else {
                delta_id_to_name
                    .get(id as usize - base_count)
                    .cloned()
                    .unwrap_or_default()
            }
        };

        for e in &delta.new_edges {
            edges.push(ExportEdge {
                from: resolve_id(e.from),
                to: resolve_id(e.to),
                edge_type: e.edge_type.clone(),
                weight: e.weight,
                full_context: e.full_context.clone(),
                embed_context: e.embed_context.clone(),
            });
        }
    }

    (nodes, edges)
}

fn now_rfc3339() -> String {
    // Use std time — no chrono dependency required.
    // Format: "YYYY-MM-DDTHH:MM:SSZ" (UTC, second precision)
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = epoch_to_ymd_hms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn epoch_to_ymd_hms(mut secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let s = (secs % 60) as u32; secs /= 60;
    let mi = (secs % 60) as u32; secs /= 60;
    let h = (secs % 24) as u32; secs /= 24;
    // days since 1970-01-01
    let mut days = secs;
    let mut y = 1970u32;
    loop {
        let dy = if is_leap(y) { 366 } else { 365 };
        if days < dy { break; }
        days -= dy;
        y += 1;
    }
    let months = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut mo = 1u32;
    for &dm in &months {
        if days < dm { break; }
        days -= dm;
        mo += 1;
    }
    (y, mo, days as u32 + 1, h, mi, s)
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

// ── GET /export/graph ─────────────────────────────────────────────────────────

pub async fn handle_export_graph(
    State(s): State<AppState>,
    Query(params): Query<ExportParams>,
) -> Response {
    let (nodes, edges) = collect_export_data(&s).await;
    let exported_at = now_rfc3339();
    match params.format.as_str() {
        "ndjson" => export_ndjson(nodes, edges, exported_at),
        _ => export_json(nodes, edges, exported_at),
    }
}

/// POST /export/graph — same as GET but accepts format in JSON body.
/// Useful when the client prefers POST or wants to extend with filters later.
pub async fn handle_export_graph_post(
    State(s): State<AppState>,
    body: Option<Json<ExportBody>>,
) -> Response {
    let format = body.map(|Json(b)| b.format).unwrap_or_else(default_format);
    let (nodes, edges) = collect_export_data(&s).await;
    let exported_at = now_rfc3339();
    match format.as_str() {
        "ndjson" => export_ndjson(nodes, edges, exported_at),
        _ => export_json(nodes, edges, exported_at),
    }
}

fn export_json(nodes: Vec<ExportNode>, edges: Vec<ExportEdge>, exported_at: String) -> Response {
    let payload = GraphExport {
        version: "1".into(),
        exported_at,
        stats: ExportStats {
            nodes: nodes.len(),
            edges: edges.len(),
        },
        nodes,
        edges,
    };

    match serde_json::to_vec_pretty(&payload) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"graph_export.json\"",
            )
            .body(Body::from(bytes))
            .unwrap(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("serialization error: {e}"),
        )
            .into_response(),
    }
}

fn export_ndjson(nodes: Vec<ExportNode>, edges: Vec<ExportEdge>, exported_at: String) -> Response {
    let mut lines = Vec::new();

    let meta = NdjsonLine::Meta {
        version: "1".into(),
        exported_at,
        nodes: nodes.len(),
        edges: edges.len(),
    };
    if let Ok(s) = serde_json::to_string(&meta) {
        lines.push(s);
    }

    for n in nodes {
        let line = NdjsonLine::Node {
            name: n.name,
            node_type: n.node_type,
            props: n.props,
            full_context: n.full_context,
            embed_context: n.embed_context,
        };
        if let Ok(s) = serde_json::to_string(&line) {
            lines.push(s);
        }
    }

    for e in edges {
        let line = NdjsonLine::Edge {
            from: e.from,
            to: e.to,
            edge_type: e.edge_type,
            weight: e.weight,
            full_context: e.full_context,
            embed_context: e.embed_context,
        };
        if let Ok(s) = serde_json::to_string(&line) {
            lines.push(s);
        }
    }

    let body = lines.join("\n") + "\n";

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"graph_export.ndjson\"",
        )
        .body(Body::from(body))
        .unwrap()
}

// ── POST /import/graph ────────────────────────────────────────────────────────

/// Import response — summary of what was added.
#[derive(serde::Serialize)]
pub struct ImportResult {
    pub nodes_added: usize,
    pub edges_added: usize,
    pub skipped_edges: usize,
    pub delta_size: usize,
}

/// Import via JSON body (Content-Type: application/json).
pub async fn handle_import_json(
    State(s): State<AppState>,
    Json(payload): Json<GraphExport>,
) -> impl IntoResponse {
    apply_import(s, payload).await
}

/// Import via multipart file upload (Content-Type: multipart/form-data, field name: `file`).
/// Accepts both JSON and NDJSON files.
pub async fn handle_import_multipart(
    State(s): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Extract the first field named "file"
    let mut raw: Option<Vec<u8>> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            match field.bytes().await {
                Ok(b) => { raw = Some(b.to_vec()); break; }
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({ "error": format!("failed to read file: {e}") })),
                    ).into_response();
                }
            }
        }
    }

    let bytes = match raw {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "missing field: file" })),
            ).into_response();
        }
    };

    // Detect format: if first non-whitespace byte is `{` it may be JSON or NDJSON
    let text = match std::str::from_utf8(&bytes) {
        Ok(t) => t,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "file is not valid UTF-8" })),
            ).into_response();
        }
    };

    // Try full JSON first, then NDJSON
    let payload = if let Ok(p) = serde_json::from_str::<GraphExport>(text) {
        p
    } else {
        match parse_ndjson(text) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("could not parse file: {e}") })),
                ).into_response();
            }
        }
    };

    apply_import(s, payload).await.into_response()
}

/// Parse NDJSON text into a `GraphExport`.
fn parse_ndjson(text: &str) -> Result<GraphExport, String> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut version = "1".to_string();
    let mut exported_at = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let record: NdjsonLine = serde_json::from_str(line)
            .map_err(|e| format!("invalid NDJSON line: {e}"))?;

        match record {
            NdjsonLine::Meta { version: v, exported_at: ea, .. } => {
                version = v;
                exported_at = ea;
            }
            NdjsonLine::Node { name, node_type, props, full_context, embed_context } => {
                nodes.push(ExportNode { name, node_type, props, full_context, embed_context });
            }
            NdjsonLine::Edge { from, to, edge_type, weight, full_context, embed_context } => {
                edges.push(ExportEdge { from, to, edge_type, weight, full_context, embed_context });
            }
        }
    }

    Ok(GraphExport {
        version,
        exported_at,
        stats: ExportStats { nodes: nodes.len(), edges: edges.len() },
        nodes,
        edges,
    })
}

/// Apply a parsed `GraphExport` to the live delta store.
async fn apply_import(s: AppState, payload: GraphExport) -> impl IntoResponse {
    use crate::graph::csr::{EdgeInfo, NodeInfo};

    let mut nodes_added = 0usize;
    let mut edges_added = 0usize;
    let mut skipped_edges = 0usize;

    // We must resolve node names → IDs when adding edges.
    // Collect existing names from CSR + any names we just inserted.
    let mut name_to_id: std::collections::HashMap<String, u32> = {
        let graph = s.graph.read().await;
        graph
            .nodes
            .iter()
            .map(|n| (n.name.clone(), n.id))
            .collect()
    };

    // Add nodes
    for n in payload.nodes {
        if name_to_id.contains_key(&n.name) {
            // Node already exists — skip silently
            continue;
        }
        let id = match s.delta.alloc_node_id() {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!("import: ID space exhausted: {e}");
                break;
            }
        };
        let info = NodeInfo {
            id,
            name: n.name.clone(),
            node_type: n.node_type,
            weight: 0.0,
            props: n.props,
            full_context: n.full_context,
            embed_context: n.embed_context,
        };
        s.delta.add_node(info, vec![]);
        name_to_id.insert(n.name, id);
        nodes_added += 1;
    }

    // Add edges
    for e in payload.edges {
        let from_id = match name_to_id.get(&e.from) {
            Some(&id) => id,
            None => { skipped_edges += 1; continue; }
        };
        let to_id = match name_to_id.get(&e.to) {
            Some(&id) => id,
            None => { skipped_edges += 1; continue; }
        };
        let info = EdgeInfo {
            from: from_id,
            to: to_id,
            edge_type: e.edge_type,
            weight: e.weight,
            full_context: e.full_context,
            embed_context: e.embed_context,
        };
        s.delta.add_edge(info, vec![]);
        edges_added += 1;
    }

    s.metrics.delta_size.set(s.delta.size() as u64);

    (
        StatusCode::OK,
        Json(ImportResult {
            nodes_added,
            edges_added,
            skipped_edges,
            delta_size: s.delta.size(),
        }),
    )
        .into_response()
}
