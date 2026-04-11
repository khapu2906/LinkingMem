use anyhow::Result;
use serde_json;
use crate::graph::csr::{CsrGraph, EdgeInfo, NodeInfo};
use crate::storage::StorageBackend;

/// Save graph snapshot files via the given storage backend.
///
/// Writes:
///   nodes.json          — node metadata (name, type, props, full_context)
///   edges.bin           — [from: u32, to: u32, weight: f32] * num_edges  (12 bytes each)
///   edge_types.json     — edge type labels (parallel to edges.bin)
///   edge_contexts.json  — edge full_context strings (parallel to edges.bin)
pub fn save(graph: &CsrGraph, storage: &dyn StorageBackend) -> Result<()> {
    let nodes_json = serde_json::to_string_pretty(&graph.nodes)?;
    storage.write_bytes("nodes.json", nodes_json.as_bytes())?;

    let mut edge_buf:           Vec<u8>           = Vec::with_capacity(graph.num_edges() * 12);
    let mut edge_types:         Vec<String>        = Vec::with_capacity(graph.num_edges());
    let mut edge_contexts:      Vec<String>        = Vec::with_capacity(graph.num_edges());
    let mut edge_embed_ctxs:    Vec<Option<String>> = Vec::with_capacity(graph.num_edges());
    for node_id in 0..graph.num_nodes() {
        let neighbors   = graph.neighbors(node_id as u32);
        let weights     = graph.neighbor_weights(node_id as u32);
        let types       = graph.neighbor_types(node_id as u32);
        let contexts    = graph.neighbor_full_contexts(node_id as u32);
        let embed_ctxs  = graph.neighbor_embed_contexts(node_id as u32);
        for i in 0..neighbors.len() {
            edge_buf.extend_from_slice(&(node_id as u32).to_le_bytes());
            edge_buf.extend_from_slice(&neighbors[i].to_le_bytes());
            edge_buf.extend_from_slice(&weights[i].to_le_bytes());
            edge_types.push(types[i].clone());
            edge_contexts.push(contexts[i].clone());
            edge_embed_ctxs.push(embed_ctxs.get(i).cloned().unwrap_or(None));
        }
    }
    storage.write_bytes("edges.bin", &edge_buf)?;
    storage.write_string("edge_types.json",         &serde_json::to_string(&edge_types)?)?;
    storage.write_string("edge_contexts.json",      &serde_json::to_string(&edge_contexts)?)?;
    storage.write_string("edge_embed_contexts.json",&serde_json::to_string(&edge_embed_ctxs)?)?;

    tracing::info!(
        "saved graph: {} nodes, {} edges → {}",
        graph.num_nodes(), graph.num_edges(), storage.local_path().display()
    );
    Ok(())
}

/// Load graph from snapshot files via the given storage backend.
pub fn load(storage: &dyn StorageBackend) -> Result<CsrGraph> {
    let nodes_json = storage.read_string("nodes.json")?;
    let nodes: Vec<NodeInfo> = serde_json::from_str(&nodes_json)?;

    let edge_bytes = storage.read_bytes("edges.bin")?;
    let num_edges  = edge_bytes.len() / 12;

    // edge_types.json is optional — older graphs without it get empty strings
    let edge_types: Vec<String> = storage.read_string("edge_types.json")
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    // edge_contexts.json is optional — older graphs without it get empty strings
    let edge_contexts: Vec<String> = storage.read_string("edge_contexts.json")
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    // edge_embed_contexts.json is optional
    let edge_embed_ctxs: Vec<Option<String>> = storage.read_string("edge_embed_contexts.json")
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let mut edges: Vec<EdgeInfo> = Vec::with_capacity(num_edges);
    for (i, chunk) in edge_bytes.chunks_exact(12).enumerate() {
        let from = u32::from_le_bytes(chunk[0..4].try_into()?);
        let to   = u32::from_le_bytes(chunk[4..8].try_into()?);
        let w    = f32::from_le_bytes(chunk[8..12].try_into()?);
        edges.push(EdgeInfo {
            from, to,
            edge_type:     edge_types.get(i).cloned().unwrap_or_default(),
            full_context:  edge_contexts.get(i).cloned().unwrap_or_default(),
            embed_context: edge_embed_ctxs.get(i).cloned().unwrap_or(None),
            weight: w,
        });
    }

    tracing::info!("loaded graph: {} nodes, {} edges", nodes.len(), edges.len());
    Ok(CsrGraph::build(nodes, &edges))
}

/// Build a CsrGraph from a JSON ingestion payload.
/// Expected format:
/// {
///   "entities": [{"id": "e1", "name": "...", "type": "...", "props": {...}}],
///   "relations": [{"from": "e1", "to": "e2", "type": "...", "weight": 1.0}]
/// }
pub fn from_json_payload(payload: &serde_json::Value) -> Result<CsrGraph> {
    use std::collections::HashMap;

    let entities = payload["entities"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing 'entities' array"))?;

    let relations = payload["relations"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing 'relations' array"))?;

    let mut id_map: HashMap<String, u32> = HashMap::new();
    let mut nodes: Vec<NodeInfo> = Vec::new();

    for (idx, e) in entities.iter().enumerate() {
        let str_id = e["id"].as_str().unwrap_or("").to_string();
        id_map.insert(str_id.clone(), idx as u32);
        let node_type = e["type"].as_str().unwrap_or("").trim().to_string();
        nodes.push(NodeInfo {
            id:           idx as u32,
            name:         e["name"].as_str().unwrap_or("").trim().to_string(),
            node_type:    if node_type.is_empty() { "Entity".to_string() } else { node_type },
            weight:       0.0,
            props:        e["props"].clone(),
            full_context:  e["full_context"].as_str().unwrap_or("").to_string(),
            embed_context: e["embed_context"].as_str()
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.to_string()),
        });
    }

    let mut edge_list: Vec<EdgeInfo> = Vec::new();
    for r in relations {
        let from_str = r["from"].as_str().unwrap_or("");
        let to_str   = r["to"].as_str().unwrap_or("");
        if let (Some(&from), Some(&to)) = (id_map.get(from_str), id_map.get(to_str)) {
            edge_list.push(EdgeInfo {
                from,
                to,
                edge_type:     r["type"].as_str().unwrap_or("").to_string(),
                weight:        r["weight"].as_f64().unwrap_or(1.0) as f32,
                full_context:  r["full_context"].as_str().unwrap_or("").to_string(),
                embed_context: r["embed_context"].as_str()
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.to_string()),
            });
        }
    }

    Ok(CsrGraph::build(nodes, &edge_list))
}
