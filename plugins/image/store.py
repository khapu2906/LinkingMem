"""Image storage — POST /store + GET /images/{filename}.

Saves images to local disk, content-addressed by SHA-256.
Returns a stable HTTP URL served by this plugin.

Config env vars:
  IMAGE_LOCAL_DIR      directory to store image files (default: ./data/images)
  IMAGE_SERVE_BASE_URL base URL prefix for returned URLs
                       (default: http://localhost:8002)
"""

import base64
import hashlib
import logging
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .schemas import StoreRequest, StoreResponse

log = logging.getLogger(__name__)

router = APIRouter()

_LOCAL_DIR = Path(os.getenv("IMAGE_LOCAL_DIR", "./data/images"))
_SERVE_BASE = os.getenv("IMAGE_SERVE_BASE_URL", "http://localhost:8002").rstrip("/")
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard limit

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png":  ".png",
    "image/gif":  ".gif",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
}


def _ensure_dir() -> None:
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def _ext_from_mime(mime: str) -> str:
    return _MIME_TO_EXT.get(mime.lower().split(";")[0].strip(), ".bin")


def _resolve_input(req: StoreRequest) -> tuple[bytes, str]:
    """Return (raw_bytes, content_type) from either URL or data-URI."""
    if req.data:
        # data:image/jpeg;base64,<data>
        if not req.data.startswith("data:"):
            raise ValueError("data field must be a base64 data-URI (data:<mime>;base64,...)")
        header, _, payload = req.data.partition(",")
        mime = header.split(";")[0].removeprefix("data:").strip()
        raw = base64.b64decode(payload)
        return raw, mime or "image/jpeg"

    if req.url:
        resp = httpx.get(req.url, follow_redirects=True, timeout=20)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        return resp.content, mime

    raise ValueError("request must include 'url' or 'data'")


def _save(raw: bytes, content_type: str) -> tuple[Path, str]:
    """Write bytes to content-addressed path. Returns (path, filename)."""
    digest = hashlib.sha256(raw).hexdigest()
    ext    = _ext_from_mime(content_type)
    fname  = f"{digest}{ext}"
    path   = _LOCAL_DIR / fname
    if not path.exists():
        path.write_bytes(raw)
        log.info("stored image: %s (%d bytes)", fname, len(raw))
    else:
        log.debug("image already exists: %s", fname)
    return path, fname


@router.post("/store", response_model=StoreResponse)
async def store_image(req: StoreRequest, _: None = Depends(verify_auth)):
    if not req.url and not req.data:
        raise HTTPException(400, "request must include 'url' or 'data'")

    _ensure_dir()

    try:
        raw, content_type = await run_in_threadpool(_resolve_input, req)
    except Exception as e:
        log.error("failed to fetch/decode image: %s", e)
        raise HTTPException(422, f"could not load image: {e}")

    if len(raw) > _MAX_BYTES:
        raise HTTPException(413, f"image too large: {len(raw)} bytes (max {_MAX_BYTES})")

    try:
        path, filename = await run_in_threadpool(_save, raw, content_type)
    except Exception as e:
        log.error("failed to save image: %s", e)
        raise HTTPException(500, f"storage error: {e}")

    return StoreResponse(
        url=f"{_SERVE_BASE}/images/{filename}",
        filename=filename,
        size_bytes=len(raw),
        content_type=content_type,
    )


@router.get("/images/{filename}")
async def serve_image(filename: str):
    # Sanitize: no path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")

    path = _LOCAL_DIR / filename
    if not path.exists():
        raise HTTPException(404, "image not found")

    ext = path.suffix.lower()
    mime_map = {v: k for k, v in _MIME_TO_EXT.items()}
    media_type = mime_map.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type)
