"""Image embedding — POST /embed.

Pipeline:
  1. Load image from URL (http/https) or base64 data-URI
  2. Caption via Vision LLM
  3. Embed caption using sentence-transformers (same model as text plugin)
  4. Return vector in the shared embedding space

Config env vars:
  IMAGE_CAPTION_MODEL   vision-capable LLM (default: google/gemini-2.5-flash-lite)
  EMBED_MODEL           sentence-transformers model (default: all-MiniLM-L6-v2)
  EMBED_BACKEND         "torch" | "onnx"  (default: torch)
  EMBED_BATCH_SIZE      sentences per batch (default: 32)
  LLM_PROVIDER          fallback provider when model has no prefix (default: google)
"""

import asyncio
import base64
import logging
import os
import threading

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .schemas import EmbedRequest, EmbedResponse

log = logging.getLogger(__name__)
router = APIRouter()

# ── embedding model (lazy, thread-safe) ──────────────────────────────────────

_EMBED_MODEL   = os.getenv("EMBED_MODEL",         "all-MiniLM-L6-v2")
_EMBED_BACKEND = os.getenv("EMBED_BACKEND",        "torch")
_BATCH_SIZE    = int(os.getenv("EMBED_BATCH_SIZE", "32"))
_MAX_CONCURRENCY = int(os.getenv("EMBED_MAX_CONCURRENCY", "4"))

_encode_sem    = asyncio.Semaphore(_MAX_CONCURRENCY)
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
                    kwargs["backend"] = "onnx"
                _embedder = SentenceTransformer(_EMBED_MODEL, **kwargs)
                log.info("image plugin: embedding model loaded (%s)", _EMBED_MODEL)
    return _embedder


def warmup() -> None:
    if os.getenv("EMBED_WARMUP", "1") == "0":
        return
    log.info("image plugin: warming up embedding model…")
    get_embedder().encode(["warmup"], normalize_embeddings=True, batch_size=1)
    log.info("image plugin: embedding model ready")


# ── vision LLM captioning ─────────────────────────────────────────────────────

_IMAGE_CAPTION_MODEL: str = os.getenv(
    "IMAGE_CAPTION_MODEL",
    os.getenv("GENERATE_MODEL", "google/gemini-2.5-flash-lite"),
)

_CAPTION_SYSTEM = (
    "You are a precise image captioning assistant. "
    "Describe the image content in a single dense sentence focusing on entities, "
    "relationships, and key attributes. Be specific and factual. "
    "Output only the caption — no preamble, no explanation."
)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _load_image(image_url: str) -> tuple[bytes, str]:
    """Return (raw_bytes, mime_type). Handles http(s) and data-URIs."""
    if image_url.startswith("data:"):
        header, _, data = image_url.partition(",")
        mime = header.split(";")[0].removeprefix("data:").strip()
        return base64.b64decode(data), mime or "image/jpeg"
    resp = httpx.get(image_url, follow_redirects=True, timeout=20)
    resp.raise_for_status()
    mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    raw  = resp.content
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {len(raw)} bytes")
    return raw, mime


def _parse_model(model_str: str) -> tuple[str, str]:
    _ALIASES = {"gemini": "google"}
    _SUPPORTED = {"google", "openai", "anthropic"}
    _LEGACY = os.getenv("LLM_PROVIDER", "google").lower()
    if "/" in model_str:
        prefix, name = model_str.split("/", 1)
        provider = _ALIASES.get(prefix, prefix)
    else:
        provider = _ALIASES.get(_LEGACY, _LEGACY)
        name = model_str
    if provider not in _SUPPORTED:
        raise ValueError(f"Unknown provider '{provider}'")
    return provider, name


_client_cache: dict = {}
_client_lock = threading.Lock()


def _get_client(provider: str):
    if provider not in _client_cache:
        with _client_lock:
            if provider not in _client_cache:
                if provider == "openai":
                    from openai import OpenAI
                    _client_cache[provider] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                elif provider == "anthropic":
                    import anthropic
                    _client_cache[provider] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                else:
                    from google import genai
                    _client_cache[provider] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client_cache[provider]


def _caption(image_url: str) -> str:
    provider, model_name = _parse_model(_IMAGE_CAPTION_MODEL)
    client = _get_client(provider)
    log.info("captioning via %s/%s", provider, model_name)

    if provider == "openai":
        if image_url.startswith("http"):
            content = [{"type": "image_url", "image_url": {"url": image_url}}]
        else:
            raw, mime = _load_image(image_url)
            b64 = base64.b64encode(raw).decode()
            content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
        content.append({"type": "text", "text": "Describe this image in one dense sentence."})
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": _CAPTION_SYSTEM},
                      {"role": "user", "content": content}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()

    if provider == "anthropic":
        raw, mime = _load_image(image_url)
        b64 = base64.b64encode(raw).decode()
        resp = client.messages.create(
            model=model_name, max_tokens=200, system=_CAPTION_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": "Describe this image in one dense sentence."},
            ]}],
        )
        return resp.content[0].text.strip()

    # google / gemini
    from google.genai import types
    if image_url.startswith("http"):
        image_part = types.Part.from_uri(file_uri=image_url, mime_type="image/jpeg")
    else:
        raw, mime = _load_image(image_url)
        image_part = types.Part.from_bytes(data=raw, mime_type=mime)
    resp = client.models.generate_content(
        model=model_name,
        contents=[image_part, "Describe this image in one dense sentence."],
        config=types.GenerateContentConfig(
            system_instruction=_CAPTION_SYSTEM, max_output_tokens=200,
        ),
    )
    return resp.text.strip()


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.post("/embed", response_model=EmbedResponse)
async def embed_image(req: EmbedRequest, _: None = Depends(verify_auth)):
    if not req.image_url:
        raise HTTPException(400, "image_url is required")

    try:
        caption = await run_in_threadpool(_caption, req.image_url)
    except Exception as e:
        log.error("caption error: %s", e)
        raise HTTPException(500, f"captioning failed: {e}")

    log.info("caption: %.120s", caption)

    try:
        async with _encode_sem:
            vecs = await run_in_threadpool(
                get_embedder().encode,
                [caption],
                normalize_embeddings=True,
                batch_size=_BATCH_SIZE,
                show_progress_bar=False,
            )
    except Exception as e:
        log.error("embed error: %s", e)
        raise HTTPException(500, f"embed failed: {e}")

    vector = vecs[0].tolist()
    return EmbedResponse(vector=vector, dim=len(vector), caption=caption, model=_IMAGE_CAPTION_MODEL)
