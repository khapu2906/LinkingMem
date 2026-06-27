"""Image plugin server — entry point.

Handles image storage (local disk) and image embedding for the AI Graph Engine.

Run:
    uv run uvicorn image.main:app --host 0.0.0.0 --port 8002 --reload

Endpoints:
  POST /store            save image to local disk, return stable URL
  GET  /images/{file}    serve stored image
  POST /embed            caption via Vision LLM + embed caption text
  GET  /health
  GET  /info

Config env vars:
  IMAGE_LOCAL_DIR        where to store files  (default: ./data/images)
  IMAGE_SERVE_BASE_URL   base URL for /images/ (default: http://localhost:8002)
  IMAGE_CAPTION_MODEL    vision LLM            (default: google/gemini-2.5-flash-lite)
  EMBED_MODEL            embedding model       (default: all-MiniLM-L6-v2)
  PLUGIN_AUTH_TOKEN      optional Bearer auth
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .auth import PLUGIN_AUTH_TOKEN
from .embed import _EMBED_MODEL, _IMAGE_CAPTION_MODEL, warmup
from .embed import router as embed_router
from .store import _LOCAL_DIR, _SERVE_BASE
from .store import router as store_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup()
    yield


app = FastAPI(title="AI Graph Engine — Image Plugin", lifespan=lifespan)

app.include_router(store_router)
app.include_router(embed_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "name": "image-plugin",
        "version": "0.1.0",
        "capabilities": ["store", "embed"],
        "embed_model":          _EMBED_MODEL,
        "image_caption_model":  _IMAGE_CAPTION_MODEL,
        "storage_backend":      "local",
        "local_dir":            str(_LOCAL_DIR),
        "serve_base_url":       _SERVE_BASE,
        "auth_enabled":         bool(PLUGIN_AUTH_TOKEN),
    }
