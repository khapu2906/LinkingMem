"""Image embedding — POST /embed/image.

Pipeline:
  1. Receive image_url (http/https or base64 data-URI)
  2. Call Vision LLM to generate a descriptive caption
  3. Embed the caption with the same text model as /embed/text
  4. Return the vector in the shared 768-dim (or model-dim) space

This keeps image nodes in the SAME vector space as text nodes so cross-modal
similarity search works without any extra infrastructure.

Config env vars:
  IMAGE_CAPTION_MODEL   Vision LLM for captioning (default: same as GENERATE_MODEL
                        or google/gemini-2.5-flash-lite).
                        Must be a vision-capable model.
                        Examples: openai/gpt-4o, google/gemini-2.5-flash-lite,
                                  anthropic/claude-haiku-4-5-20251001
"""

import base64
import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .embed import _EMBED_MODEL, _BATCH_SIZE, _encode_sem, get_embedder
from .llm import default_model, llm_generate
from .schemas import EmbedImageRequest, EmbedImageResponse

log = logging.getLogger(__name__)

router = APIRouter()

_IMAGE_CAPTION_MODEL: str = os.getenv(
    "IMAGE_CAPTION_MODEL",
    os.getenv("GENERATE_MODEL", default_model("generate")),
)

_CAPTION_SYSTEM = (
    "You are a precise image captioning assistant. "
    "Describe the image content in a single dense sentence focusing on entities, "
    "relationships, and key attributes. Be specific and factual. "
    "No preamble, no trailing punctuation variants — just the caption."
)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB guard


def _load_image_bytes(image_url: str) -> tuple[bytes, str]:
    """Return (raw_bytes, mime_type). Handles both http(s) and data URIs."""
    if image_url.startswith("data:"):
        # data:image/jpeg;base64,<data>
        header, _, data = image_url.partition(",")
        mime = header.split(";")[0].removeprefix("data:")
        return base64.b64decode(data), mime

    resp = httpx.get(image_url, follow_redirects=True, timeout=15)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    raw = resp.content
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {len(raw)} bytes (max {_MAX_IMAGE_BYTES})")
    return raw, content_type


def _caption_openai(image_url: str, model_name: str, client) -> str:
    from openai import OpenAI
    # OpenAI vision: pass URL directly if http, else convert to data URI
    if image_url.startswith("http"):
        content = [{"type": "image_url", "image_url": {"url": image_url}}]
    else:
        raw, mime = _load_image_bytes(image_url)
        b64 = base64.b64encode(raw).decode()
        content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
    content.append({"type": "text", "text": "Describe this image in one dense sentence."})
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _CAPTION_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def _caption_anthropic(image_url: str, model_name: str, client) -> str:
    raw, mime = _load_image_bytes(image_url)
    b64 = base64.b64encode(raw).decode()
    resp = client.messages.create(
        model=model_name,
        max_tokens=200,
        system=_CAPTION_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": "Describe this image in one dense sentence."},
            ],
        }],
    )
    return resp.content[0].text.strip()


def _caption_google(image_url: str, model_name: str, client) -> str:
    from google.genai import types
    if image_url.startswith("http"):
        image_part = types.Part.from_uri(file_uri=image_url, mime_type="image/jpeg")
    else:
        raw, mime = _load_image_bytes(image_url)
        image_part = types.Part.from_bytes(data=raw, mime_type=mime)
    resp = client.models.generate_content(
        model=model_name,
        contents=[image_part, "Describe this image in one dense sentence."],
        config=types.GenerateContentConfig(
            system_instruction=_CAPTION_SYSTEM,
            max_output_tokens=200,
        ),
    )
    return resp.text.strip()


def caption_image(image_url: str) -> str:
    """Generate a text caption for the image using the configured Vision LLM."""
    from .llm import parse_model, get_client
    provider, model_name = parse_model(_IMAGE_CAPTION_MODEL)
    client = get_client(provider)
    log.info("captioning image via %s/%s", provider, model_name)
    if provider == "openai":
        return _caption_openai(image_url, model_name, client)
    if provider == "anthropic":
        return _caption_anthropic(image_url, model_name, client)
    return _caption_google(image_url, model_name, client)


@router.post("/embed/image", response_model=EmbedImageResponse)
async def embed_image(req: EmbedImageRequest, _: None = Depends(verify_auth)):
    if not req.image_url:
        raise HTTPException(400, "image_url is required")

    try:
        caption = await run_in_threadpool(caption_image, req.image_url)
    except Exception as e:
        log.error("image caption error: %s", e)
        raise HTTPException(500, f"captioning failed: {e}")

    log.info("image caption: %s", caption[:120])

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
        log.error("embed error after caption: %s", e)
        raise HTTPException(500, f"embed failed: {e}")

    vector = vecs[0].tolist()
    return EmbedImageResponse(
        vector=vector,
        dim=len(vector),
        caption=caption,
        model=_IMAGE_CAPTION_MODEL,
    )
