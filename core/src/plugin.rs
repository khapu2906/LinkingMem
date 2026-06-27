use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::config::PluginsConfig;
use crate::graph::csr::NodeInfo;

const HEALTH_TTL: Duration = Duration::from_secs(5);

/// HTTP/Unix client that talks to the Python plugin server.
/// All AI work (embed, extract, generate, reason) lives in Python.
/// Rust stays pure compute — no AI dependencies.
///
/// Each operation (embed / extract / generate / reason) can point to a different
/// URL or unix socket path, each with its own optional Bearer auth token.
pub struct PluginClient {
    /// shared HTTP client for all HTTP endpoints
    client: reqwest::Client,

    embed_text_url:  String,
    embed_image_url: String,
    image_store_url: String,
    extract_url:     String,
    generate_url:    String,

    embed_text_token:  Option<String>,
    embed_image_token: Option<String>,
    image_store_token: Option<String>,
    extract_token:     Option<String>,
    generate_token:    Option<String>,

    /// Some = unix socket endpoint; None = use HTTP URL above
    embed_text_socket:  Option<PathBuf>,
    embed_image_socket: Option<PathBuf>,
    image_store_socket: Option<PathBuf>,
    extract_socket:     Option<PathBuf>,
    generate_socket:    Option<PathBuf>,

    health_cache: std::sync::Mutex<Option<(bool, std::time::Instant)>>,

    embed_timeout:    Duration,
    extract_timeout:  Duration,
    generate_timeout: Duration,
    health_timeout:   Duration,
}

// ─── LLM hint types ────────────────────────────────────────────────────────

/// Per-request LLM behaviour overrides.  Mirrors the Python `LlmHints` schema.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct LlmHints {
    /// Fully replace the operation's default system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Extra rules appended after the default rules block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<String>,

    /// Extra text snippets injected into the prompt context (generate/reason only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extend_context: Vec<String>,
}

impl LlmHints {
    pub fn is_empty(&self) -> bool {
        self.system_prompt.is_none() && self.rules.is_empty() && self.extend_context.is_empty()
    }
}

// ─── request/response types ────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    vectors: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct EmbedImageRequest {
    image_url: String,
}

#[derive(Deserialize)]
struct EmbedImageResponse {
    vector: Vec<f32>,
}

#[derive(Serialize)]
struct StoreImageRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    url:  Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
}

#[derive(Deserialize)]
struct StoreImageResponse {
    url: String,
}

#[derive(Serialize)]
struct ExtractRequest {
    text:  String,
    hints: LlmHints,
}

#[derive(Deserialize)]
pub struct ExtractResponse {
    pub entities: Vec<serde_json::Value>,
    pub relations: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct GenerateRequest {
    context:   Vec<ContextNode>,
    relations: Vec<ContextEdge>,
    query:     String,
    hints:     LlmHints,
}

#[derive(Serialize)]
struct ContextNode {
    id:           u32,
    name:         String,
    node_type:    String,
    props:        serde_json::Value,
    full_context: String,
    score:        f32,
}

#[derive(Serialize)]
struct ContextEdge {
    from_node: String,
    to_node:   String,
    weight:    f32,
    #[serde(skip_serializing_if = "String::is_empty")]
    edge_type: String,
}

#[derive(Deserialize)]
struct GenerateResponse {
    answer: String,
}

/// Multi-hop reasoning request (mirrors Python ReasonRequest schema)
#[derive(Serialize)]
struct ReasonRequest {
    context:        Vec<ContextNode>,
    relations:      Vec<ContextEdge>,
    query:          String,
    iteration:      u32,
    max_iterations: u32,
    hints:          LlmHints,
}

/// Multi-hop reasoning response (mirrors Python ReasonResponse schema)
#[derive(Deserialize, Debug)]
pub struct ReasonResponse {
    pub answer:     String,
    pub follow_ups: Vec<String>,
    pub done:       bool,
}

// ─── helpers ───────────────────────────────────────────────────────────────

fn env_duration(var: &str, default_secs: u64) -> Duration {
    Duration::from_secs(
        std::env::var(var)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(default_secs),
    )
}

fn build_context(context: &[NodeInfo]) -> Vec<ContextNode> {
    context.iter().map(|n| ContextNode {
        id:           n.id,
        name:         n.name.clone(),
        node_type:    n.node_type.clone(),
        props:        n.props.clone(),
        full_context: n.full_context.clone(),
        score:        n.weight,
    }).collect()
}

fn build_edges(
    context: &[NodeInfo],
    edges: &[(u32, u32, f32, String)],
) -> Vec<ContextEdge> {
    let id_to_name: std::collections::HashMap<u32, &str> =
        context.iter().map(|n| (n.id, n.name.as_str())).collect();
    edges.iter()
        .filter_map(|(from, to, w, et)| Some(ContextEdge {
            from_node: id_to_name.get(from)?.to_string(),
            to_node:   id_to_name.get(to)?.to_string(),
            weight:    *w,
            edge_type: et.clone(),
        }))
        .collect()
}

// ─── unix socket transport ────────────────────────────────────────────────
//
// Opens a new connection per request (no pooling needed — unix socket
// connections to a local process are extremely cheap).

async fn unix_post<T, R>(
    socket_path: &Path,
    path: &str,
    body: &T,
    token: Option<&str>,
    timeout: Duration,
) -> Result<R>
where
    T: Serialize,
    R: serde::de::DeserializeOwned,
{
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::Request;
    use hyper_util::rt::TokioIo;
    use tokio::net::UnixStream;

    let stream = tokio::time::timeout(timeout, UnixStream::connect(socket_path))
        .await
        .map_err(|_| anyhow::anyhow!("unix connect timed out: {}", socket_path.display()))?
        .map_err(|e| anyhow::anyhow!("unix connect failed: {e}"))?;

    let io = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http1::Builder::new()
        .handshake(io)
        .await?;
    tokio::spawn(async move { let _ = conn.await; });

    let json = serde_json::to_vec(body)?;
    let mut builder = Request::builder()
        .uri(format!("http://localhost{path}"))
        .method("POST")
        .header("content-type", "application/json")
        .header("content-length", json.len().to_string())
        .header("host", "localhost");
    if let Some(t) = token {
        builder = builder.header("authorization", format!("Bearer {t}"));
    }
    let req = builder.body(Full::new(Bytes::from(json)))?;

    let resp = tokio::time::timeout(timeout, sender.send_request(req))
        .await
        .map_err(|_| anyhow::anyhow!("unix request timed out"))?
        .map_err(|e| anyhow::anyhow!("unix request failed: {e}"))?;

    let status = resp.status();
    let body_bytes = resp.into_body().collect().await
        .map_err(|e| anyhow::anyhow!("unix response read failed: {e}"))?
        .to_bytes();

    if !status.is_success() {
        let msg = String::from_utf8_lossy(&body_bytes);
        anyhow::bail!("plugin returned {status}: {msg}");
    }

    serde_json::from_slice(&body_bytes)
        .map_err(|e| anyhow::anyhow!("unix response parse failed: {e}"))
}

// ─── PluginClient ──────────────────────────────────────────────────────────

impl PluginClient {
    pub fn new(cfg: &PluginsConfig) -> anyhow::Result<Self> {
        let connect_timeout = env_duration("PLUGIN_CONNECT_TIMEOUT_SECS", 5);

        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;

        let embed_timeout    = env_duration("PLUGIN_EMBED_TIMEOUT_SECS",     30);
        let extract_timeout  = env_duration("PLUGIN_EXTRACT_TIMEOUT_SECS",   60);
        let generate_timeout = env_duration("PLUGIN_GENERATE_TIMEOUT_SECS", 120);
        let health_timeout   = env_duration("PLUGIN_HEALTH_TIMEOUT_SECS",     3);

        tracing::info!(
            "plugin timeouts — connect={connect_timeout:?} embed={embed_timeout:?} \
             extract={extract_timeout:?} generate={generate_timeout:?}"
        );
        tracing::info!(
            "plugin endpoints — embed_text={} embed_image={} image_store={} extract={} generate={}",
            cfg.embed_text.url(), cfg.embed_image.url(), cfg.image_store.url(),
            cfg.extract.url(), cfg.generate.url(),
        );

        Ok(Self {
            client,
            embed_text_url:    cfg.embed_text.url().trim_end_matches('/').to_string(),
            embed_image_url:   cfg.embed_image.url().trim_end_matches('/').to_string(),
            image_store_url:   cfg.image_store.url().trim_end_matches('/').to_string(),
            extract_url:       cfg.extract.url().trim_end_matches('/').to_string(),
            generate_url:      cfg.generate.url().trim_end_matches('/').to_string(),
            embed_text_token:  cfg.embed_text.auth_token().map(str::to_string),
            embed_image_token: cfg.embed_image.auth_token().map(str::to_string),
            image_store_token: cfg.image_store.auth_token().map(str::to_string),
            extract_token:     cfg.extract.auth_token().map(str::to_string),
            generate_token:    cfg.generate.auth_token().map(str::to_string),
            embed_text_socket:  cfg.embed_text.socket().map(|p| p.to_path_buf()),
            embed_image_socket: cfg.embed_image.socket().map(|p| p.to_path_buf()),
            image_store_socket: cfg.image_store.socket().map(|p| p.to_path_buf()),
            extract_socket:     cfg.extract.socket().map(|p| p.to_path_buf()),
            generate_socket:    cfg.generate.socket().map(|p| p.to_path_buf()),
            health_cache: std::sync::Mutex::new(None),
            embed_timeout,
            extract_timeout,
            generate_timeout,
            health_timeout,
        })
    }

    /// Dispatch a POST request to either HTTP or Unix socket depending on config.
    async fn post<T, R>(
        &self,
        base_url: &str,
        socket: Option<&Path>,
        token: Option<&str>,
        path: &str,
        body: &T,
        timeout: Duration,
    ) -> Result<R>
    where
        T: Serialize,
        R: serde::de::DeserializeOwned,
    {
        if let Some(sock) = socket {
            return unix_post(sock, path, body, token, timeout).await;
        }
        let resp: R = self
            .client
            .post(format!("{base_url}{path}"))
            .header("content-type", "application/json")
            .bearer_auth_opt(token)
            .timeout(timeout)
            .json(body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp)
    }

    pub async fn check_ready(&self) -> bool {
        {
            let cached = match self.health_cache.lock() { Ok(g) => g, Err(e) => e.into_inner() };
            if let Some((ready, checked_at)) = *cached {
                if checked_at.elapsed() < HEALTH_TTL {
                    return ready;
                }
            }
        }
        let ready = self.health().await;
        let mut cached = match self.health_cache.lock() { Ok(g) => g, Err(e) => e.into_inner() };
        *cached = Some((ready, std::time::Instant::now()));
        ready
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let resp: EmbedResponse = self
            .post(
                &self.embed_text_url,
                self.embed_text_socket.as_deref(),
                self.embed_text_token.as_deref(),
                "/embed/text",
                &EmbedRequest { texts },
                self.embed_timeout,
            )
            .await?;
        Ok(resp.vectors)
    }

    /// Embed texts in chunks of `chunk_size` to avoid HTTP timeout on large payloads.
    /// Each chunk is a separate request; results are concatenated in order.
    pub async fn embed_chunked(&self, texts: Vec<String>, chunk_size: usize) -> Result<Vec<Vec<f32>>> {
        if texts.len() <= chunk_size {
            return self.embed(texts).await;
        }
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(chunk_size) {
            let vecs = self.embed(chunk.to_vec()).await?;
            all.extend(vecs);
        }
        Ok(all)
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut vecs = self.embed(vec![text.to_string()]).await?;
        vecs.pop().ok_or_else(|| anyhow::anyhow!("empty embed response"))
    }

    /// Embed an image by URL or base64 data-URI.
    /// The plugin generates a caption via Vision LLM, then embeds it in the same
    /// vector space as text nodes — enabling cross-modal graph search.
    pub async fn embed_image(&self, image_url: &str) -> Result<Vec<f32>> {
        let resp: EmbedImageResponse = self
            .post(
                &self.embed_image_url,
                self.embed_image_socket.as_deref(),
                self.embed_image_token.as_deref(),
                "/embed/image",
                &EmbedImageRequest { image_url: image_url.to_string() },
                self.embed_timeout,
            )
            .await?;
        Ok(resp.vector)
    }

    /// Store an image on the image plugin and return the stable URL.
    ///
    /// `data` is either an `http(s)://` URL or a `data:<mime>;base64,<payload>` URI.
    /// The plugin writes it content-addressed to local disk and returns the served URL.
    /// Only call this when `ImageConfig.auto_store` is true.
    pub async fn store_image(&self, data: &str) -> Result<String> {
        let req = if data.starts_with("data:") {
            StoreImageRequest { url: None, data: Some(data.to_string()) }
        } else {
            StoreImageRequest { url: Some(data.to_string()), data: None }
        };
        let resp: StoreImageResponse = self
            .post(
                &self.image_store_url,
                self.image_store_socket.as_deref(),
                self.image_store_token.as_deref(),
                "/store",
                &req,
                self.embed_timeout,
            )
            .await?;
        Ok(resp.url)
    }

    pub async fn extract(&self, text: &str, hints: Option<LlmHints>) -> Result<ExtractResponse> {
        self.post(
            &self.extract_url,
            self.extract_socket.as_deref(),
            self.extract_token.as_deref(),
            "/extract",
            &ExtractRequest { text: text.to_string(), hints: hints.unwrap_or_default() },
            self.extract_timeout,
        )
        .await
    }

    pub async fn generate(
        &self,
        context: &[NodeInfo],
        edges:   &[(u32, u32, f32, String)],
        query:   &str,
        hints:   Option<LlmHints>,
    ) -> Result<String> {
        let resp: GenerateResponse = self
            .post(
                &self.generate_url,
                self.generate_socket.as_deref(),
                self.generate_token.as_deref(),
                "/generate",
                &GenerateRequest {
                    context:   build_context(context),
                    relations: build_edges(context, edges),
                    query:     query.to_string(),
                    hints:     hints.unwrap_or_default(),
                },
                self.generate_timeout,
            )
            .await?;
        Ok(resp.answer)
    }

    /// Multi-hop reasoning step.
    ///
    /// Returns `ReasonResponse::done = true` with the final answer, or
    /// `done = false` with `follow_ups` entity names for the next hop.
    pub async fn reason(
        &self,
        context:        &[NodeInfo],
        edges:          &[(u32, u32, f32, String)],
        query:          &str,
        iteration:      u32,
        max_iterations: u32,
        hints:          Option<LlmHints>,
    ) -> Result<ReasonResponse> {
        // /reason lives on the generate endpoint server
        self.post(
            &self.generate_url,
            self.generate_socket.as_deref(),
            self.generate_token.as_deref(),
            "/reason",
            &ReasonRequest {
                context:        build_context(context),
                relations:      build_edges(context, edges),
                query:          query.to_string(),
                iteration,
                max_iterations,
                hints:          hints.unwrap_or_default(),
            },
            self.generate_timeout,
        )
        .await
    }

    pub async fn health(&self) -> bool {
        // Unix socket transport: the socket file existing is sufficient proof
        // that the server is up — no HTTP round-trip needed.
        if let Some(sock) = &self.embed_text_socket {
            return sock.exists();
        }
        // HTTP transport: hit the /health endpoint.
        self.client
            .get(format!("{}/health", self.embed_text_url))
            .timeout(self.health_timeout)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

// ─── reqwest extension helper ──────────────────────────────────────────────

trait BearerAuthOpt {
    fn bearer_auth_opt(self, token: Option<&str>) -> Self;
}

impl BearerAuthOpt for reqwest::RequestBuilder {
    fn bearer_auth_opt(self, token: Option<&str>) -> Self {
        match token {
            Some(t) => self.bearer_auth(t),
            None    => self,
        }
    }
}
