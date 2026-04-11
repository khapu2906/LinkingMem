//! Application configuration.
//!
//! Loading order (later overrides earlier):
//!   1. Built-in defaults
//!   2. `plugins.toml` file (path from CONFIG_FILE env var, default: `../plugins.toml`)
//!   3. Environment variables (always win)
//!
//! Example:
//! ```no_run
//! use ai_graph_engine::config::AppConfig;
//! let cfg = AppConfig::load().unwrap();
//! println!("{}", cfg.server.bind_addr);
//! println!("{}", cfg.plugins.embed_text.url());
//! ```

use std::path::{Path, PathBuf};
use anyhow::Result;
use serde::Deserialize;

// ── Top-level ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server:  ServerConfig,
    pub data:    DataConfig,
    pub plugins: PluginsConfig,
    pub cache:   CacheConfig,
    pub delta:   DeltaConfig,
    pub auth:    AuthConfig,
    pub ingest:  IngestConfig,
    pub query:   QueryConfig,
}

impl AppConfig {
    /// Load config from file + env vars.
    pub fn load() -> Result<Self> {
        let config_path = std::env::var("PLUGIN_CONFIG_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_base_dir().join("plugins.toml"));

        let file_cfg = FileConfig::from_file_or_default(&config_path);

        Ok(Self {
            server:  ServerConfig::resolve(&file_cfg.server),
            data:    DataConfig::resolve(&file_cfg.data),
            plugins: PluginsConfig::resolve(&file_cfg.plugins),
            cache:   CacheConfig::resolve(&file_cfg.cache),
            delta:   DeltaConfig::resolve(&file_cfg.delta),
            auth:    AuthConfig::resolve(),
            ingest:  IngestConfig::resolve(&file_cfg.ingest),
            query:   QueryConfig::resolve(&file_cfg.query),
        })
    }
}

// ── Server ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind the HTTP server. Env: BIND_ADDR
    pub bind_addr: String,
    /// Log level filter. Env: RUST_LOG
    pub log_level: String,
}

impl ServerConfig {
    fn resolve(file: &FileServerConfig) -> Self {
        Self {
            bind_addr: std::env::var("BIND_ADDR")
                .unwrap_or_else(|_| file.bind_addr.clone()
                    .unwrap_or_else(|| "0.0.0.0:8000".into())),
            log_level: std::env::var("RUST_LOG")
                .unwrap_or_else(|_| file.log_level.clone()
                    .unwrap_or_else(|| "info".into())),
        }
    }
}

// ── Data ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Directory containing graph.bin, vectors.bin, delta.wal. Env: DATA_DIR
    pub dir: PathBuf,
}

impl DataConfig {
    fn resolve(file: &FileDataConfig) -> Self {
        Self {
            dir: std::env::var("DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| file.dir.as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| default_base_dir().join("data"))),
        }
    }
}

// ── Plugins ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PluginsConfig {
    pub embed_text: PluginEndpoint,
    pub extract:    PluginEndpoint,
    pub generate:   PluginEndpoint,
}

impl PluginsConfig {
    fn resolve(file: &FilePluginsConfig) -> Self {
        Self {
            embed_text: PluginEndpoint::resolve(&file.embed_text, "PLUGIN_EMBED_TEXT_URL", "PLUGIN_URL"),
            extract:    PluginEndpoint::resolve(&file.extract,    "PLUGIN_EXTRACT_URL",    "PLUGIN_URL"),
            generate:   PluginEndpoint::resolve(&file.generate,   "PLUGIN_GENERATE_URL",   "PLUGIN_URL"),
        }
    }

    /// Convenience constructor for CLIs that only need a single plugin URL with no auth.
    pub fn from_single_url(url: &str) -> Self {
        let ep = PluginEndpoint::Http { url: url.to_string(), auth_token: None };
        Self { embed_text: ep.clone(), extract: ep.clone(), generate: ep }
    }
}

#[derive(Debug, Clone)]
pub enum PluginEndpoint {
    /// HTTP/HTTPS transport. `auth_token` is sent as `Authorization: Bearer <token>`
    /// when present. Leave `None` if the plugin server has no auth.
    Http  { url: String, auth_token: Option<String> },
    /// Unix domain socket. Auth is skipped — socket permissions are the security boundary.
    Unix  { socket: PathBuf },
}

impl PluginEndpoint {
    /// Return the HTTP base URL (for Unix sockets this is a placeholder the
    /// hyper connector intercepts; the actual path is in the variant).
    pub fn url(&self) -> &str {
        match self {
            Self::Http { url, .. } => url,
            Self::Unix { .. }      => "http://localhost",
        }
    }

    /// Bearer token to send in the `Authorization` header.
    /// Always `None` for Unix sockets — the socket permissions handle access control.
    pub fn auth_token(&self) -> Option<&str> {
        match self {
            Self::Http { auth_token, .. } => auth_token.as_deref(),
            Self::Unix { .. }             => None,
        }
    }

    /// Unix socket path, or `None` for HTTP endpoints.
    pub fn socket(&self) -> Option<&std::path::Path> {
        match self {
            Self::Unix { socket } => Some(socket.as_path()),
            Self::Http { .. }     => None,
        }
    }

    fn resolve(file: &FilePluginEntry, specific_env: &str, fallback_env: &str) -> Self {
        // 1. specific env var wins (e.g. PLUGIN_EMBED_URL)
        if let Ok(url) = std::env::var(specific_env) {
            return Self::Http { url, auth_token: None };
        }
        // 2. generic PLUGIN_URL fallback
        if let Ok(url) = std::env::var(fallback_env) {
            return Self::Http { url, auth_token: None };
        }
        // 3. file config
        if file.transport.as_deref() == Some("unix") {
            if let Some(socket) = &file.socket {
                return Self::Unix { socket: PathBuf::from(socket) };
            }
        }
        if let Some(url) = &file.url {
            return Self::Http { url: url.clone(), auth_token: file.auth_token.clone() };
        }
        // 4. built-in default
        Self::Http { url: "http://localhost:8001".into(), auth_token: None }
    }
}

// ── Cache ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Max entries in the embedding vector cache. Env: EMBED_CACHE_SIZE
    pub embed_cache_size: usize,
    /// Max entries in the query result cache. Env: QUERY_CACHE_SIZE
    pub query_cache_size: usize,
    /// TTL for query result cache entries in seconds. Env: QUERY_CACHE_TTL_SECS
    pub query_cache_ttl_secs: u64,
}

impl CacheConfig {
    fn resolve(file: &FileCacheConfig) -> Self {
        Self {
            embed_cache_size: env_parse("EMBED_CACHE_SIZE")
                .unwrap_or(file.embed_cache_size.unwrap_or(50_000)),
            query_cache_size: env_parse("QUERY_CACHE_SIZE")
                .unwrap_or(file.query_cache_size.unwrap_or(10_000)),
            query_cache_ttl_secs: env_parse("QUERY_CACHE_TTL_SECS")
                .unwrap_or(file.query_cache_ttl_secs.unwrap_or(300)),
        }
    }
}

// ── Delta ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeltaConfig {
    /// Trigger async merge when delta exceeds this many entries. Env: DELTA_MERGE_THRESHOLD
    pub merge_threshold: usize,
}

impl DeltaConfig {
    fn resolve(file: &FileDeltaConfig) -> Self {
        Self {
            merge_threshold: env_parse("DELTA_MERGE_THRESHOLD")
                .unwrap_or(file.merge_threshold.unwrap_or(500)),
        }
    }
}

// ── Auth ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Comma-separated list of allowed CORS origins. Env: CORS_ALLOWED_ORIGINS
    /// Empty = permissive (all origins allowed — dev mode only).
    pub cors_origins: Vec<String>,
}

impl AuthConfig {
    fn resolve() -> Self {
        let cors_origins = std::env::var("CORS_ALLOWED_ORIGINS")
            .map(|s| s.split(',').map(|o| o.trim().to_string()).filter(|o| !o.is_empty()).collect())
            .unwrap_or_default();
        Self { cors_origins }
    }
}

// ── Query ─────────────────────────────────────────────────────────────────────

/// Default query pipeline parameters, all overridable per-request via `QueryOptions`.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Top-k seed nodes from HNSW vector search. Env: HNSW_K
    pub hnsw_k: usize,
    /// BFS expansion depth from seeds. Env: BFS_DEPTH
    pub bfs_depth: u8,
    /// Max nodes collected during BFS. Env: BFS_MAX_NODES
    pub bfs_max_nodes: usize,
    /// Top-n scored nodes passed to the LLM. Env: CONTEXT_TOP_N
    pub context_top_n: usize,
    /// Minimum score threshold to include a node in LLM context. Env: CONTEXT_MIN_SCORE
    pub context_min_score: f32,
    /// HNSW M parameter — max connections per node. Env: HNSW_M
    pub hnsw_m: usize,
    /// HNSW ef_construction — beam width during index build. Env: HNSW_EF_CONSTRUCTION
    pub hnsw_ef_construction: usize,
}

impl QueryConfig {
    fn resolve(file: &FileQueryConfig) -> Self {
        Self {
            hnsw_k:               env_parse("HNSW_K")               .unwrap_or(file.hnsw_k               .unwrap_or(20)),
            bfs_depth:            env_parse("BFS_DEPTH")             .unwrap_or(file.bfs_depth             .unwrap_or(2)),
            bfs_max_nodes:        env_parse("BFS_MAX_NODES")         .unwrap_or(file.bfs_max_nodes         .unwrap_or(500)),
            context_top_n:        env_parse("CONTEXT_TOP_N")         .unwrap_or(file.context_top_n         .unwrap_or(50)),
            context_min_score:    env_parse("CONTEXT_MIN_SCORE")     .unwrap_or(file.context_min_score     .unwrap_or(0.3)),
            hnsw_m:               env_parse("HNSW_M")               .unwrap_or(file.hnsw_m                .unwrap_or(16)),
            hnsw_ef_construction: env_parse("HNSW_EF_CONSTRUCTION")  .unwrap_or(file.hnsw_ef_construction  .unwrap_or(200)),
        }
    }
}

// ── Ingest ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Default entity resolution mode. Env: RESOLUTION_MODE ("none" | "embedding")
    pub resolution_mode: String,
    /// Cosine similarity threshold for embedding-based resolution. Env: RESOLUTION_THRESHOLD
    pub resolution_threshold: f32,
    /// Require matching entity type when resolving. Env: RESOLUTION_MATCH_TYPE
    pub resolution_match_type: bool,
}

impl IngestConfig {
    fn resolve(file: &FileIngestConfig) -> Self {
        Self {
            resolution_mode: std::env::var("RESOLUTION_MODE")
                .unwrap_or_else(|_| file.resolution_mode.clone()
                    .unwrap_or_else(|| "embedding".into())),
            resolution_threshold: env_parse("RESOLUTION_THRESHOLD")
                .unwrap_or(file.resolution_threshold.unwrap_or(0.92)),
            resolution_match_type: env_parse::<bool>("RESOLUTION_MATCH_TYPE")
                .unwrap_or(file.resolution_match_type.unwrap_or(false)),
        }
    }
}

// ── File format (plugins.toml) ────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    #[serde(default)] server:  FileServerConfig,
    #[serde(default)] data:    FileDataConfig,
    #[serde(default)] plugins: FilePluginsConfig,
    #[serde(default)] cache:   FileCacheConfig,
    #[serde(default)] delta:   FileDeltaConfig,
    #[serde(default)] ingest:  FileIngestConfig,
    #[serde(default)] query:   FileQueryConfig,
}

#[derive(Debug, Default, Deserialize)]
struct FileQueryConfig {
    hnsw_k:               Option<usize>,
    bfs_depth:            Option<u8>,
    bfs_max_nodes:        Option<usize>,
    context_top_n:        Option<usize>,
    context_min_score:    Option<f32>,
    hnsw_m:               Option<usize>,
    hnsw_ef_construction: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct FileServerConfig {
    bind_addr: Option<String>,
    log_level: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct FileDataConfig {
    dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct FilePluginsConfig {
    #[serde(default)] embed_text: FilePluginEntry,
    #[serde(default)] extract:    FilePluginEntry,
    #[serde(default)] generate:   FilePluginEntry,
}

#[derive(Debug, Default, Deserialize)]
struct FilePluginEntry {
    transport:  Option<String>,  // "http" | "unix"
    url:        Option<String>,
    socket:     Option<String>,
    auth_token: Option<String>,  // Bearer token (HTTP only; ignored for unix socket)
}

#[derive(Debug, Default, Deserialize)]
struct FileCacheConfig {
    embed_cache_size:     Option<usize>,
    query_cache_size:     Option<usize>,
    query_cache_ttl_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct FileDeltaConfig {
    merge_threshold: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct FileIngestConfig {
    resolution_mode:       Option<String>,
    resolution_threshold:  Option<f32>,
    resolution_match_type: Option<bool>,
}

impl FileConfig {
    fn from_file_or_default(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(cfg) => cfg,
                Err(e) => {
                    // Parse error is likely a misconfiguration — log as error so it's visible,
                    // but fall back to defaults so the engine can still start.
                    tracing::error!(
                        "failed to parse config file {}: {e}\n\
                         Fix the file or unset PLUGIN_CONFIG_FILE to suppress this error.\n\
                         Falling back to built-in defaults.",
                        path.display()
                    );
                    Self::default()
                }
            },
            Err(_) => {
                tracing::debug!("config file {} not found — using defaults", path.display());
                Self::default()
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.parse().ok()
}

/// Returns the "home" directory for default file paths.
///
/// Resolution order:
///   1. Directory containing the running binary  (production: /opt/engine/)
///   2. Cargo workspace root via CARGO_MANIFEST_DIR (dev: cargo run)
///   3. Current working directory as last resort
///
/// This means:
///   - `data/`        lives next to the binary in production
///   - `plugins.toml` lives next to the binary in production
///   - Both live at the project root during `cargo run`
fn default_base_dir() -> PathBuf {
    // 1. next to the binary (production)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // skip Cargo's target/debug or target/release dirs
            let is_cargo_target = dir.ends_with("debug") || dir.ends_with("release");
            if !is_cargo_target {
                return dir.to_path_buf();
            }
        }
    }
    // 2. workspace root during `cargo run`
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest).parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
    }
    // 3. cwd fallback
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
