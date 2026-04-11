"""Embedding — lazy model loading and /embed endpoint.

Built-in models (any sentence-transformers model name is accepted):
  all-MiniLM-L6-v2       384-dim  English        fast, 22 MB (default)
  BAAI/bge-small-en-v1.5 384-dim  English        high quality, small
  thenlper/gte-small      384-dim  English        high quality, small
  BAAI/bge-m3            1024-dim  multilingual   best quality, large

Tuning env vars:
  EMBED_MODEL             model name (default: all-MiniLM-L6-v2)
  EMBED_BACKEND           inference backend: "torch" (default) | "onnx"
                          ONNX requires: pip install optimum[onnxruntime]
                          First load exports the model to ONNX (~10s); subsequent
                          loads reuse the cached file for fast startup.
  EMBED_BATCH_SIZE        sentences per encode batch (default: 32)
  EMBED_MAX_CONCURRENCY   max simultaneous encode calls (default: 4)
  EMBED_WARMUP            set to "0" to skip warmup on startup (default: enabled)
"""

import asyncio
import logging
import os
import threading

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .schemas import EmbedRequest, EmbedResponse

log = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Model registry — add entries here when bundling new models.
# ---------------------------------------------------------------------------
EMBED_MODELS: dict[str, dict] = {
    "all-MiniLM-L6-v2":       {"dim": 384,  "lang": "en",           "note": "default, fast, 22 MB"},
    "BAAI/bge-small-en-v1.5": {"dim": 384,  "lang": "en",           "note": "high quality, small"},
    "thenlper/gte-small":      {"dim": 384,  "lang": "en",           "note": "high quality, small"},
    "BAAI/bge-m3":             {"dim": 1024, "lang": "multilingual", "note": "best quality, large"},
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_EMBED_MODEL       = os.getenv("EMBED_MODEL",           "all-MiniLM-L6-v2")
_EMBED_BACKEND     = os.getenv("EMBED_BACKEND",         "torch")   # "torch" | "onnx"
_BATCH_SIZE        = int(os.getenv("EMBED_BATCH_SIZE",  "32"))
_MAX_CONCURRENCY   = int(os.getenv("EMBED_MAX_CONCURRENCY", "4"))

# Semaphore prevents CPU spike when N requests arrive simultaneously.
# Encode is CPU-bound; saturating all cores degrades latency for everyone.
_encode_sem = asyncio.Semaphore(_MAX_CONCURRENCY)

# ---------------------------------------------------------------------------
# Lazy model loading (thread-safe double-checked locking)
# ---------------------------------------------------------------------------
_embedder      = None
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                kwargs: dict = {}
                if _EMBED_BACKEND == "onnx":
                    # Requires: pip install optimum[onnxruntime]
                    # First run exports the model to ONNX and caches it locally.
                    kwargs["backend"] = "onnx"
                    log.info("loading embedding model (ONNX): %s", _EMBED_MODEL)
                else:
                    log.info("loading embedding model (torch): %s", _EMBED_MODEL)
                _embedder = SentenceTransformer(_EMBED_MODEL, **kwargs)
    return _embedder


def warmup() -> None:
    """Force model load and run one dummy encode to JIT-compile kernels.

    Call this on startup so the first real request doesn't pay the load penalty.
    Controlled by EMBED_WARMUP env var (set to "0" to skip).
    """
    if os.getenv("EMBED_WARMUP", "1") == "0":
        return
    log.info("warming up embedding model…")
    get_embedder().encode(["warmup"], normalize_embeddings=True, batch_size=1)
    log.info("embedding model ready")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/embed/text", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest, _: None = Depends(verify_auth)):
    if not req.texts:
        raise HTTPException(400, "texts array is empty")

    try:
        async with _encode_sem:
            # encode is CPU-bound — offload to thread pool so the event loop
            # stays free to accept other requests during inference.
            vecs = await run_in_threadpool(
                get_embedder().encode,
                req.texts,
                normalize_embeddings=True,
                batch_size=_BATCH_SIZE,
                show_progress_bar=False,
            )
    except Exception as e:
        log.error("embed error: %s", e)
        raise HTTPException(500, f"embed failed: {e}")

    dim = len(vecs[0]) if len(vecs) > 0 else 0
    return EmbedResponse(vectors=vecs.tolist(), dim=dim, model=_EMBED_MODEL)
