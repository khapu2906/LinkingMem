use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::plugin::LlmHints;

// ── resolution override ───────────────────────────────────────────────────────

#[derive(Deserialize, Default, Clone, Validate)]
pub struct ResolutionOverride {
    /// "none" | "embedding"
    #[serde(default)]
    pub mode: Option<String>,

    /// Cosine similarity threshold [0.0–1.0]
    #[validate(range(min = 0.0, max = 1.0, message = "threshold must be between 0.0 and 1.0"))]
    #[serde(default)]
    pub threshold: Option<f32>,

    /// Require same entity type to match
    #[serde(default)]
    pub match_type: Option<bool>,
}

// ── /ingest/text ─────────────────────────────────────────────────────────────

#[derive(Deserialize, Validate)]
pub struct IngestTextReq {
    #[validate(length(min = 1, max = 100_000, message = "text must be 1–100k characters"))]
    pub text: String,

    #[validate(nested)]
    #[serde(default)]
    pub resolution: Option<ResolutionOverride>,

    /// Per-request LLM extraction customization (rules, system prompt override).
    #[serde(default)]
    pub hints: Option<LlmHints>,
}

// ── /ingest/json ──────────────────────────────────────────────────────────────

#[derive(Deserialize, Validate)]
pub struct IngestJsonReq {
    #[validate(length(min = 1, max = 10_000, message = "entities must have 1–10k items"))]
    pub entities: Vec<serde_json::Value>,

    #[serde(default)]
    pub relations: Vec<serde_json::Value>,

    #[validate(nested)]
    #[serde(default)]
    pub resolution: Option<ResolutionOverride>,
}

// ── response ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct IngestResponse {
    pub ingested:        usize,
    pub new_nodes:       usize,
    pub resolved:        usize,
    pub delta_size:      usize,
    pub merge_triggered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message:         Option<String>,
}
