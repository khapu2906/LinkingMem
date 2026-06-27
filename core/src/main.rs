use axum::{middleware, routing::{get, post}, Router};
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use ai_graph_engine::{
    api::handlers::{
        admin::{force_merge, get_node, graph_stats, health, metrics_endpoint, search_nodes, search_edges},
        export::{handle_export_graph, handle_export_graph_post, handle_import_json, handle_import_multipart},
        ingest::{ingest_json, ingest_text},
        query::{handle_query, handle_query_image, handle_query_text, handle_query_vector, handle_query_node, handle_query_multihop},
    },
    app_state::{boot, run_merge, AppState},
    config::AppConfig,
    middleware::auth::{auth_middleware, AuthState, PublicRateLimitState, public_rate_limit_middleware},
};

// ── router ────────────────────────────────────────────────────────────────────

fn build_cors(origins: &[String]) -> CorsLayer {
    if origins.is_empty() {
        tracing::warn!("CORS_ALLOWED_ORIGINS not set — using permissive CORS (dev mode only)");
        return CorsLayer::permissive();
    }
    use tower_http::cors::AllowOrigin;
    use axum::http::HeaderValue;
    let allowed: Vec<HeaderValue> = origins.iter()
        .filter_map(|o| o.parse().ok())
        .collect();
    tracing::info!("CORS restricted to {} origin(s)", allowed.len());
    CorsLayer::new().allow_origin(AllowOrigin::list(allowed))
}

fn build_router(state: AppState, auth: AuthState, public_rl: PublicRateLimitState, cors: CorsLayer) -> Router {
    let protected = Router::new()
        .route("/query",        post(handle_query))
        .route("/query/text",   post(handle_query_text))
        .route("/query/vector", post(handle_query_vector))
        .route("/query/node",      post(handle_query_node))
        .route("/query/image",     post(handle_query_image))
        .route("/query/multihop",  post(handle_query_multihop))
        .route("/ingest/text",  post(ingest_text))
        .route("/ingest/json", post(ingest_json))
        .route("/export/graph", get(handle_export_graph).post(handle_export_graph_post))
        .route("/import/graph", post(handle_import_json))
        .route("/import/graph/upload", post(handle_import_multipart))
        .route("/delta/merge", post(force_merge))
        .layer(middleware::from_fn_with_state(auth, auth_middleware))
        .with_state(state.clone());

    let public = Router::new()
        .route("/health",       get(health))
        .route("/metrics",      get(metrics_endpoint))
        .route("/graph/stats",  get(graph_stats))
        .route("/nodes",        get(search_nodes))
        .route("/nodes/{id}",   get(get_node))
        .route("/edges",        get(search_edges))
        .layer(middleware::from_fn_with_state(public_rl, public_rate_limit_middleware))
        .with_state(state);

    Router::new()
        .merge(protected)
        .merge(public)
        .layer(cors)
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = AppConfig::load()?;

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(&cfg.server.log_level))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!(
        "config loaded — bind={} data={} plugin={}",
        cfg.server.bind_addr,
        cfg.data.dir.display(),
        cfg.plugins.embed_text.url(),
    );

    let state      = boot(&cfg).await?;
    let auth_state = AuthState::from_env();
    let public_rl  = PublicRateLimitState::from_env();
    let cors       = build_cors(&cfg.auth.cors_origins);
    let bind_addr  = cfg.server.bind_addr.clone();

    // background pruner for rate limiter buckets (every 5 min)
    {
        let limiter = auth_state.limiter.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                limiter.prune();
            }
        });
    }

    let app = build_router(state.clone(), auth_state, public_rl, cors);

    tracing::info!("engine listening on {bind_addr}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Flush any pending delta entries to disk before exit so no writes are lost.
    let pending = state.delta.size();
    if pending > 0 {
        tracing::info!("flushing {pending} pending delta entries before exit…");
        run_merge(state).await;
    }

    tracing::info!("engine shut down cleanly");
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal;
    let ctrl_c = async { signal::ctrl_c().await.expect("failed to install Ctrl-C handler") };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
