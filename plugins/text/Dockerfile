# syntax=docker/dockerfile:1
# AI Graph Engine — Text Plugin (standalone HTTP image)
# Used by docker-compose HTTP mode (two-container deployment).

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install deps from lock file — source not copied yet so this layer is cached.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy plugin source as the `text` package so `text.main:app` is importable.
COPY . text/

# Pre-download embedding model so startup is instant at runtime.
ARG EMBED_MODEL=all-MiniLM-L6-v2
RUN EMBED_MODEL=${EMBED_MODEL} uv run python -c \
    "import os; from sentence_transformers import SentenceTransformer; \
     SentenceTransformer(os.environ['EMBED_MODEL'])"

EXPOSE 8001

ENV EMBED_MODEL=all-MiniLM-L6-v2 \
    EMBED_WARMUP=1

HEALTHCHECK --interval=5s --timeout=3s --retries=12 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

ENTRYPOINT ["uv", "run", "uvicorn", "text.main:app", \
            "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
