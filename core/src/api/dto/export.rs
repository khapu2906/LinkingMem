use serde::{Deserialize, Serialize};

/// A single node in the exported payload.
/// Uses human-readable names instead of numeric IDs so the file is
/// self-contained and can be re-imported into a fresh instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportNode {
    pub name: String,
    pub node_type: String,
    pub props: serde_json::Value,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub full_context: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// A single edge in the exported payload.
/// `from` / `to` are node *names* (matching `ExportNode::name`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportEdge {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub weight: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub full_context: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_context: Option<String>,
}

/// Stats section inside the export file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStats {
    pub nodes: usize,
    pub edges: usize,
}

/// Top-level JSON export format.
///
/// ```json
/// {
///   "version": "1",
///   "exported_at": "2024-01-01T00:00:00Z",
///   "stats": { "nodes": 1240, "edges": 4832 },
///   "nodes": [...],
///   "edges": [...]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExport {
    pub version: String,
    pub exported_at: String,
    pub stats: ExportStats,
    pub nodes: Vec<ExportNode>,
    pub edges: Vec<ExportEdge>,
}

/// One line in the NDJSON streaming format.
///
/// Each line is an independent JSON object with a `type` discriminant:
/// - `meta`  — header record (one per stream, always first)
/// - `node`  — a node record
/// - `edge`  — an edge record
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NdjsonLine {
    Meta {
        version: String,
        exported_at: String,
        nodes: usize,
        edges: usize,
    },
    Node {
        name: String,
        node_type: String,
        props: serde_json::Value,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        full_context: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        embed_context: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    Edge {
        from: String,
        to: String,
        edge_type: String,
        weight: f32,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        full_context: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        embed_context: Option<String>,
    },
}
