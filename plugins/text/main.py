"""Text plugin server — entry point.

Handles embedding, entity extraction, and answer generation for the AI Graph Engine.

Run:
    uv run uvicorn text.main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .auth import PLUGIN_AUTH_TOKEN
from .embed import _EMBED_MODEL, EMBED_MODELS, warmup
from .embed import router as embed_router
from .embed_image import _IMAGE_CAPTION_MODEL
from .embed_image import router as embed_image_router
from .extract import router as extract_router
from .generate import router as generate_router
from .llm import default_model
from .reason import router as reason_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # startup — warm up the embedding model so the first request is fast
    warmup()
    yield
    # shutdown (nothing to clean up currently)


app = FastAPI(title="AI Graph Engine — Text Plugin", lifespan=lifespan)

app.include_router(embed_router)
app.include_router(embed_image_router)
app.include_router(extract_router)
app.include_router(generate_router)
app.include_router(reason_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    meta = EMBED_MODELS.get(_EMBED_MODEL, {})
    return {
        "name": "text-plugin",
        "version": "1.0.0",
        "capabilities": ["embed/text", "embed/image", "extract", "generate", "reason"],
        "image_caption_model": _IMAGE_CAPTION_MODEL,
        "embed_model": _EMBED_MODEL,
        "embed_dim": meta.get("dim", "?"),
        "embed_lang": meta.get("lang", "?"),
        "extract_model": os.getenv("EXTRACT_MODEL", default_model("extract")),
        "generate_model": os.getenv("GENERATE_MODEL", default_model("generate")),
        "auth_enabled": bool(PLUGIN_AUTH_TOKEN),
        "available_embed_models": EMBED_MODELS,
    }
