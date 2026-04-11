# syntax=docker/dockerfile:1
# ══════════════════════════════════════════════════════════════════════════════
# AI Graph Engine — Unified Dockerfile
#
# Two build targets:
#
#   engine  — Rust engine only. Bring your own plugin via HTTP or unix socket.
#             Small image (~50 MB). Ideal for production where the plugin runs
#             separately or you supply a custom embedding/LLM backend.
#
#   full    — Rust engine + Python text plugin in one container.
#             Plugin speaks to engine via unix socket (no TCP overhead).
#             Convenient for local dev or single-host deployments.
#
# Build:
#   docker build --target engine -t ai-graph-engine:engine .
#   docker build --target full   -t ai-graph-engine:full   .
#   docker build --target full   --build-arg EMBED_MODEL=BAAI/bge-m3 \
#                                -t ai-graph-engine:full-bge .
#
# Run (all env vars passed at runtime — no secrets baked in):
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/data \
#     --env-file .env \
#     ai-graph-engine:engine
#
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/data \
#     --env-file .env \
#     ai-graph-engine:full
# ══════════════════════════════════════════════════════════════════════════════


# ── Stage 1: Rust builder ─────────────────────────────────────────────────────
FROM rust:1.85-slim-bookworm AS rust-builder

WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency compilation separately so source changes don't re-compile deps.
COPY core/Cargo.toml core/Cargo.lock core/
RUN mkdir -p core/src/bin \
    && echo 'fn main(){}' > core/src/main.rs \
    && echo 'fn main(){}' > core/src/bin/ingest.rs \
    && cargo build --manifest-path core/Cargo.toml --release --bin server --bin ingest 2>/dev/null || true \
    && rm -rf core/src

# Build the real source.
COPY core/src     core/src
COPY core/benches core/benches
RUN cargo build --manifest-path core/Cargo.toml --release --bin server --bin ingest


# ── Stage 2: Python deps + model pre-download ─────────────────────────────────
# Uses the official uv image which ships Python 3.11 + uv.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS python-builder

WORKDIR /app

# Install Python deps from lock file (no source yet — maximise layer cache).
COPY plugins/text/pyproject.toml plugins/text/uv.lock* plugins/text/
RUN cd plugins/text && uv sync --frozen --no-dev --no-install-project

# Copy plugin source.
COPY plugins/text plugins/text

# Pre-download the embedding model so runtime startup is instant.
# Override at build time: --build-arg EMBED_MODEL=BAAI/bge-m3
ARG EMBED_MODEL=all-MiniLM-L6-v2
RUN EMBED_MODEL=${EMBED_MODEL} \
    uv --project plugins/text run python -c \
    "import os; from sentence_transformers import SentenceTransformer; \
     SentenceTransformer(os.environ['EMBED_MODEL'])"


# ══════════════════════════════════════════════════════════════════════════════
# Target: engine
# Minimal Rust-only image. Plugin is NOT bundled.
# Point PLUGIN_URL (HTTP) or mount a plugins.toml (unix socket) at runtime.
# ══════════════════════════════════════════════════════════════════════════════
FROM debian:bookworm-slim AS engine

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /src/core/target/release/server  /usr/local/bin/server
COPY --from=rust-builder /src/core/target/release/ingest  /usr/local/bin/ingest

# Default plugins.toml (HTTP, localhost:8001).
# Override by mounting your own: -v ./my-plugins.toml:/app/plugins.toml:ro
COPY plugins.toml /app/plugins.toml

WORKDIR /app
VOLUME  ["/data"]
EXPOSE  8000

ENV DATA_DIR=/data \
    BIND_ADDR=0.0.0.0:8000 \
    PLUGIN_CONFIG_FILE=/app/plugins.toml \
    RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/server"]


# ══════════════════════════════════════════════════════════════════════════════
# Target: full
# Single-container: Rust engine + Python text plugin over unix socket.
# No external plugin server needed. All env vars still passed at runtime.
# ══════════════════════════════════════════════════════════════════════════════
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS full

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Rust binaries.
COPY --from=rust-builder /src/core/target/release/server  /usr/local/bin/server
COPY --from=rust-builder /src/core/target/release/ingest  /usr/local/bin/ingest

# Python plugin with pre-installed deps and pre-downloaded model.
COPY --from=python-builder /app/plugins /app/plugins

# plugins.toml wired for unix socket.
COPY docker/plugins-unix.toml /app/plugins.toml

# Startup script: launches plugin then engine.
COPY docker/entrypoint-full.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app
VOLUME  ["/data"]
EXPOSE  8000

ENV DATA_DIR=/data \
    BIND_ADDR=0.0.0.0:8000 \
    PLUGIN_CONFIG_FILE=/app/plugins.toml \
    PLUGIN_SOCKET=/tmp/ai-graph-plugin.sock \
    RUST_LOG=info

ENTRYPOINT ["/entrypoint.sh"]
