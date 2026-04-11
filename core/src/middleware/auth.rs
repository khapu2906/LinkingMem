/// Auth + rate limiting middleware.
///
/// Auth:   Bearer token in Authorization header, or X-API-Key header.
///         Single master key stored in env var MASTER_KEY.
///         If MASTER_KEY is unset → auth disabled (dev mode).
///
/// Rate limiting:
///         Token bucket per API key: N requests per window (default 60 req/min).
///         Implemented with a DashMap of (key → bucket state).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

// ── token bucket ─────────────────────────────────────────────────────────────

struct Bucket {
    tokens: f64,
    last_refill: Instant,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

impl Bucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
            capacity,
            refill_rate,
        }
    }

    /// Returns true if a token was consumed (request allowed).
    fn consume(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

// ── rate limiter ─────────────────────────────────────────────────────────────

pub struct RateLimiter {
    buckets: Mutex<HashMap<String, Bucket>>,
    capacity: f64,
    refill_rate: f64,
}

impl RateLimiter {
    /// capacity   = max burst (e.g. 20)
    /// per_minute = sustained rate (e.g. 60 req/min → 1.0 tokens/sec)
    pub fn new(capacity: u32, per_minute: u32) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            capacity: capacity as f64,
            refill_rate: per_minute as f64 / 60.0,
        }
    }

    pub fn check(&self, key: &str) -> bool {
        let mut buckets = match self.buckets.lock() { Ok(g) => g, Err(e) => e.into_inner() };
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| Bucket::new(self.capacity, self.refill_rate));
        bucket.consume()
    }

    /// Prune stale buckets (keys not seen for > 10 min) to avoid unbounded growth
    pub fn prune(&self) {
        let cutoff = Instant::now() - Duration::from_secs(600);
        let mut buckets = match self.buckets.lock() { Ok(g) => g, Err(e) => e.into_inner() };
        buckets.retain(|_, b| b.last_refill > cutoff);
    }
}

// ── multi-key auth ────────────────────────────────────────────────────────────

/// Supports a list of valid API keys (comma-separated in API_KEYS env var).
/// If the list is empty, auth is disabled (dev mode — all requests allowed).
pub struct ApiKeys {
    keys: Vec<String>,
}

impl ApiKeys {
    pub fn new(keys: Vec<String>) -> Self {
        Self { keys }
    }

    /// Load from `API_KEYS` env var (comma-separated).
    /// Falls back to `MASTER_KEY` for backward compatibility.
    /// If neither is set → auth disabled (dev mode).
    pub fn from_env() -> Self {
        let raw = std::env::var("API_KEYS")
            .or_else(|_| std::env::var("MASTER_KEY"))
            .unwrap_or_default();
        let keys: Vec<String> = raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if keys.is_empty() {
            tracing::warn!("API_KEYS not set — auth disabled (dev mode)");
        } else {
            tracing::info!("auth enabled: {} key(s) loaded", keys.len());
        }
        Self { keys }
    }

    /// Returns true if the token matches any registered key (constant-time per key).
    pub fn is_valid(&self, token: &str) -> bool {
        if self.keys.is_empty() { return true; } // dev mode — allow all
        self.keys.iter().any(|k| {
            k.len() == token.len()
                && k.bytes()
                    .zip(token.bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    == 0
        })
    }

    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty()
    }
}

// ── master key ───────────────────────────────────────────────────────────────

pub struct MasterKey {
    key: Option<String>,
}

impl MasterKey {
    /// Load from MASTER_KEY env var.
    /// If unset or empty → auth disabled (dev mode, warns on startup).
    pub fn from_env() -> Self {
        match std::env::var("MASTER_KEY") {
            Ok(val) if !val.trim().is_empty() => {
                tracing::info!("auth enabled: master key loaded");
                Self { key: Some(val.trim().to_string()) }
            }
            _ => {
                tracing::warn!("MASTER_KEY not set — auth disabled (dev mode)");
                Self { key: None }
            }
        }
    }

    pub fn is_valid(&self, token: &str) -> bool {
        match &self.key {
            None => true, // dev mode — allow all
            Some(k) => {
                // constant-time comparison to resist timing attacks
                k.len() == token.len()
                    && k.bytes()
                        .zip(token.bytes())
                        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                        == 0
            }
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.key.is_some()
    }
}

// ── extract API key from request headers ─────────────────────────────────────

pub fn extract_key(headers: &HeaderMap) -> Option<String> {
    // 1. Authorization: Bearer <key>
    if let Some(v) = headers.get("authorization") {
        if let Ok(s) = v.to_str() {
            if let Some(key) = s.strip_prefix("Bearer ") {
                return Some(key.trim().to_string());
            }
        }
    }
    // 2. X-API-Key: <key>
    if let Some(v) = headers.get("x-api-key") {
        if let Ok(s) = v.to_str() {
            return Some(s.trim().to_string());
        }
    }
    None
}

// ── axum middleware ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AuthState {
    pub master_key: std::sync::Arc<MasterKey>,
    pub limiter: std::sync::Arc<RateLimiter>,
}

impl AuthState {
    pub fn from_env() -> Self {
        let per_minute: u32 = std::env::var("RATE_LIMIT_PER_MINUTE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        let burst: u32 = std::env::var("RATE_LIMIT_BURST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        tracing::info!("rate limit: {per_minute} req/min, burst={burst}");

        Self {
            master_key: std::sync::Arc::new(MasterKey::from_env()),
            limiter: std::sync::Arc::new(RateLimiter::new(burst, per_minute)),
        }
    }
}

pub async fn auth_middleware(
    axum::extract::State(auth): axum::extract::State<AuthState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let key = extract_key(req.headers()).unwrap_or_default();

    // ── 1. authenticate ───────────────────────────────────────────────────
    if !auth.master_key.is_valid(&key) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "invalid or missing API key",
                "hint":  "set Authorization: Bearer <key> or X-API-Key: <key>"
            })),
        )
            .into_response();
    }

    // ── 2. rate limit ─────────────────────────────────────────────────────
    let bucket_key = if key.is_empty() { "anonymous".to_string() } else { key };
    if !auth.limiter.check(&bucket_key) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "rate limit exceeded",
                "hint":  "slow down — check RATE_LIMIT_PER_MINUTE env var"
            })),
        )
            .into_response();
    }

    next.run(req).await
}

// ── public endpoint rate limiter ──────────────────────────────────────────────

/// Lightweight rate limiter for public (unauthenticated) endpoints.
/// Buckets are keyed by client IP extracted from X-Forwarded-For or the
/// connection's remote addr. Default: 120 req/min per IP.
#[derive(Clone)]
pub struct PublicRateLimitState {
    pub limiter: std::sync::Arc<RateLimiter>,
}

impl PublicRateLimitState {
    pub fn from_env() -> Self {
        let per_minute: u32 = std::env::var("RATE_LIMIT_PUBLIC_PER_MINUTE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        tracing::info!("public rate limit: {per_minute} req/min per IP");
        Self {
            limiter: std::sync::Arc::new(RateLimiter::new(per_minute, per_minute)),
        }
    }
}

pub async fn public_rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<PublicRateLimitState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // prefer X-Forwarded-For (behind reverse proxy), fall back to socket addr
    let ip = req.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| addr.ip().to_string());

    if !state.limiter.check(&ip) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "rate limit exceeded",
                "hint":  "slow down — check RATE_LIMIT_PUBLIC_PER_MINUTE env var"
            })),
        )
            .into_response();
    }

    next.run(req).await
}
