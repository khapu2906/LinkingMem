use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::plugin::LlmHints;
use crate::query::{QueryOptions, QueryResult, SubgraphEdge, ScoredNode, QueryStats};

fn default_max_hops() -> u8 { 2 }

fn validate_max_hops(v: u8) -> Result<(), validator::ValidationError> {
    if v == 0 || v > 5 {
        let mut e = validator::ValidationError::new("invalid_max_hops");
        e.message = Some("max_hops must be 1–5".into());
        return Err(e);
    }
    Ok(())
}

// ── pipeline control ──────────────────────────────────────────────────────────

/// Controls which pipeline stages to execute.
#[derive(Deserialize, Clone, Debug, Default)]
pub struct PipelineControl {
    /// When false, skip LLM generation and return only the subgraph.
    /// Default: true for /query/text and /query/vector; false for /query/node.
    #[serde(default = "default_true")]
    pub llm_generate: bool,

    /// LLM prompt text. Only used when `llm_generate=true`.
    /// Relevant for /query/vector and /query/node where there is no query string to embed.
    /// When omitted, LLM summarises the context subgraph without a specific question.
    #[serde(default)]
    pub prompt: Option<String>,

    /// Per-request LLM customization: override/extend system prompt, add rules,
    /// or inject extra context snippets into the generation prompt.
    #[serde(default)]
    pub hints: Option<LlmHints>,
}

fn default_true() -> bool { true }

fn validate_mode(mode: &String) -> Result<(), validator::ValidationError> {
    match mode.as_str() {
        "semantic" | "relationship" | "entity" | "balanced" => Ok(()),
        m => {
            let mut e = validator::ValidationError::new("invalid_mode");
            e.message = Some(format!("unknown mode '{m}'; valid: semantic, relationship, entity, balanced").into());
            Err(e)
        }
    }
}

// ── request types ─────────────────────────────────────────────────────────────

/// POST /query/text — natural-language query string.
#[derive(Deserialize, Validate)]
pub struct QueryTextReq {
    /// Query text. Required, 1–4096 characters.
    #[validate(length(min = 1, max = 4_096, message = "query must be 1–4096 characters"))]
    pub query: String,

    #[serde(default)]
    pub pipeline: Option<PipelineControl>,

    #[validate(custom(function = "validate_mode"))]
    #[serde(default)]
    pub mode: Option<String>,

    #[serde(default)]
    pub options: Option<QueryOptions>,

    #[serde(default)]
    pub response: Option<ResponseProfile>,
}

/// POST /query/vector — pre-computed embedding, skips the embed step.
#[derive(Deserialize, Validate)]
pub struct QueryVectorReq {
    /// Pre-computed query embedding. Must match the server's embedding dimension.
    #[validate(length(min = 1, message = "vector must not be empty"))]
    pub vector: Vec<f32>,

    #[serde(default)]
    pub pipeline: Option<PipelineControl>,

    #[validate(custom(function = "validate_mode"))]
    #[serde(default)]
    pub mode: Option<String>,

    #[serde(default)]
    pub options: Option<QueryOptions>,

    #[serde(default)]
    pub response: Option<ResponseProfile>,
}

/// POST /query/multihop — iterative graph reasoning (Phase 10).
///
/// Same as /query/text but the LLM can request additional graph expansion
/// for up to `max_hops` iterations before producing its final answer.
#[derive(Deserialize, Validate)]
pub struct QueryMultihopReq {
    #[validate(length(min = 1, max = 4_096, message = "query must be 1–4096 characters"))]
    pub query: String,

    /// Maximum reasoning iterations (1–5, default 2).
    #[serde(default = "default_max_hops")]
    #[validate(custom(function = "validate_max_hops"))]
    pub max_hops: u8,

    #[validate(custom(function = "validate_mode"))]
    #[serde(default)]
    pub mode: Option<String>,

    #[serde(default)]
    pub options: Option<QueryOptions>,

    #[serde(default)]
    pub response: Option<ResponseProfile>,
}

/// POST /query/node — start BFS from a known node ID, no HNSW search, no LLM.
/// Pure graph exploration: subgraph only.
#[derive(Deserialize, Validate)]
pub struct QueryNodeReq {
    /// Numeric node ID to start exploration from.
    pub node_id: u32,

    #[validate(custom(function = "validate_mode"))]
    #[serde(default)]
    pub mode: Option<String>,

    #[serde(default)]
    pub options: Option<QueryOptions>,

    #[serde(default)]
    pub response: Option<ResponseProfile>,
}

/// POST /query/image — query by image URL; engine embeds the image via plugin.
#[derive(Deserialize, Validate)]
pub struct QueryImageReq {
    /// URL or base64 data-URI of the query image.
    #[validate(length(min = 1, max = 4_096, message = "image_url must be 1–4096 characters"))]
    pub image_url: String,

    #[serde(default)]
    pub pipeline: Option<PipelineControl>,

    #[validate(custom(function = "validate_mode"))]
    #[serde(default)]
    pub mode: Option<String>,

    #[serde(default)]
    pub options: Option<QueryOptions>,

    #[serde(default)]
    pub response: Option<ResponseProfile>,
}

// ── legacy request (kept for backward compat on POST /query) ──────────────────

#[derive(Deserialize, Validate)]
pub struct QueryReq {
    #[validate(length(min = 1, max = 4_096, message = "query must be 1–4096 characters"))]
    #[serde(default)]
    pub query: Option<String>,

    #[serde(default)]
    pub vector: Option<Vec<f32>>,

    #[serde(default)]
    pub pipeline: Option<PipelineControl>,

    #[validate(custom(function = "validate_mode"))]
    #[serde(default)]
    pub mode: Option<String>,

    #[serde(default)]
    pub options: Option<QueryOptions>,

    #[serde(default)]
    pub response: Option<ResponseProfile>,
}

// ── response profile ──────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize, Clone, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseProfile {
    /// Only `answer` and `stats.total_ms`.
    Minimal,
    /// `answer` + `subgraph.nodes` (no edges) + full `stats`. Default.
    #[default]
    Standard,
    /// Everything — nodes, edges, and all stats.
    Full,
}

// ── response shapes ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct QueryResponse {
    pub answer: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub subgraph: Option<SubgraphResponse>,

    pub stats: StatsResponse,
}

#[derive(Serialize)]
pub struct SubgraphResponse {
    pub nodes: Vec<ScoredNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edges: Option<Vec<SubgraphEdge>>,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_ms:       u64,
    pub cache_hit:      bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_nodes:     Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subgraph_nodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_nodes:  Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_ms:       Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_ms:      Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bfs_ms:         Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_ms:       Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_ms:         Option<u64>,
}

// ── profile projection ────────────────────────────────────────────────────────

impl QueryResponse {
    pub fn from_result(result: QueryResult, profile: &ResponseProfile) -> Self {
        let subgraph = match profile {
            ResponseProfile::Minimal => None,
            ResponseProfile::Standard => Some(SubgraphResponse {
                nodes: result.subgraph.nodes,
                edges: None,
            }),
            ResponseProfile::Full => Some(SubgraphResponse {
                nodes: result.subgraph.nodes,
                edges: Some(result.subgraph.edges),
            }),
        };

        let stats = build_stats(result.stats, profile);

        QueryResponse { answer: result.answer, subgraph, stats }
    }
}

pub(crate) fn build_stats(s: QueryStats, profile: &ResponseProfile) -> StatsResponse {
    match profile {
        ResponseProfile::Minimal => StatsResponse {
            total_ms:       s.total_ms,
            cache_hit:      s.cache_hit,
            seed_nodes:     None,
            subgraph_nodes: None,
            context_nodes:  None,
            embed_ms:       None,
            search_ms:      None,
            bfs_ms:         None,
            score_ms:       None,
            llm_ms:         None,
        },
        ResponseProfile::Standard => StatsResponse {
            total_ms:       s.total_ms,
            cache_hit:      s.cache_hit,
            seed_nodes:     Some(s.seed_nodes),
            subgraph_nodes: Some(s.subgraph_nodes),
            context_nodes:  Some(s.context_nodes),
            embed_ms:       Some(s.embed_ms),
            search_ms:      None,
            bfs_ms:         None,
            score_ms:       None,
            llm_ms:         Some(s.llm_ms),
        },
        ResponseProfile::Full => StatsResponse {
            total_ms:       s.total_ms,
            cache_hit:      s.cache_hit,
            seed_nodes:     Some(s.seed_nodes),
            subgraph_nodes: Some(s.subgraph_nodes),
            context_nodes:  Some(s.context_nodes),
            embed_ms:       Some(s.embed_ms),
            search_ms:      Some(s.search_ms),
            bfs_ms:         Some(s.bfs_ms),
            score_ms:       Some(s.score_ms),
            llm_ms:         Some(s.llm_ms),
        },
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{QueryResult, Subgraph, QueryStats};
    use validator::Validate;

    fn make_stats() -> QueryStats {
        QueryStats {
            seed_nodes: 5, subgraph_nodes: 20, context_nodes: 10,
            total_ms: 100, embed_ms: 10, search_ms: 5,
            bfs_ms: 15, score_ms: 8, llm_ms: 62,
            cache_hit: false,
        }
    }

    fn make_result() -> QueryResult {
        QueryResult {
            answer:   "test answer".into(),
            subgraph: Subgraph { nodes: vec![], edges: vec![] },
            stats:    make_stats(),
        }
    }

    // ── response profile projection ───────────────────────────────────────────

    #[test]
    fn minimal_profile_strips_subgraph_and_stats() {
        let r = QueryResponse::from_result(make_result(), &ResponseProfile::Minimal);
        assert!(r.subgraph.is_none());
        assert!(r.stats.seed_nodes.is_none());
        assert!(r.stats.llm_ms.is_none());
        assert_eq!(r.stats.total_ms, 100);
    }

    #[test]
    fn standard_profile_has_nodes_no_edges() {
        let r = QueryResponse::from_result(make_result(), &ResponseProfile::Standard);
        let sub = r.subgraph.expect("standard must include subgraph");
        assert!(sub.edges.is_none());
        assert!(r.stats.seed_nodes.is_some());
        assert!(r.stats.search_ms.is_none(), "standard omits search_ms");
    }

    #[test]
    fn full_profile_has_all_fields() {
        let r = QueryResponse::from_result(make_result(), &ResponseProfile::Full);
        let sub = r.subgraph.expect("full must include subgraph");
        assert!(sub.edges.is_some());
        assert!(r.stats.search_ms.is_some());
        assert!(r.stats.bfs_ms.is_some());
        assert!(r.stats.score_ms.is_some());
        assert!(r.stats.llm_ms.is_some());
    }

    // ── max_hops validator ────────────────────────────────────────────────────

    #[test]
    fn max_hops_zero_is_invalid() {
        assert!(validate_max_hops(0).is_err());
    }

    #[test]
    fn max_hops_five_is_valid() {
        assert!(validate_max_hops(5).is_ok());
    }

    #[test]
    fn max_hops_six_is_invalid() {
        assert!(validate_max_hops(6).is_err());
    }

    // ── mode validator ────────────────────────────────────────────────────────

    #[test]
    fn valid_modes_pass() {
        for m in &["semantic", "relationship", "entity", "balanced"] {
            assert!(validate_mode(&m.to_string()).is_ok(), "mode '{m}' should be valid");
        }
    }

    #[test]
    fn unknown_mode_fails() {
        assert!(validate_mode(&"hybrid".to_string()).is_err());
    }

    // ── LlmHints default / is_empty ───────────────────────────────────────────

    #[test]
    fn llm_hints_default_is_empty() {
        let h = LlmHints::default();
        assert!(h.is_empty());
    }

    #[test]
    fn llm_hints_with_rules_not_empty() {
        let h = LlmHints { rules: vec!["Answer in French".into()], ..Default::default() };
        assert!(!h.is_empty());
    }

    // ── QueryTextReq validation ───────────────────────────────────────────────

    #[test]
    fn empty_query_text_fails_validation() {
        let req = QueryTextReq {
            query:    String::new(),
            pipeline: None, mode: None, options: None, response: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn valid_query_text_passes() {
        let req = QueryTextReq {
            query:    "Who is Alice?".into(),
            pipeline: None, mode: None, options: None, response: None,
        };
        assert!(req.validate().is_ok());
    }
}
