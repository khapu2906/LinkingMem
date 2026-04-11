"""LLM provider factory — models specified as "provider/model-name".

Supported formats:
  openai/gpt-4o-mini
  openai/gpt-4o
  anthropic/claude-haiku-4-5-20251001
  anthropic/claude-sonnet-4-6
  google/gemini-2.5-flash-lite
  google/gemini-2.0-flash

Provider aliases (all resolve to the canonical name):
  gemini/...  →  google/...

Backward-compatible: if no "/" prefix is given, LLM_PROVIDER env var is used
as the provider (defaults to "google").

Examples (.env):
  EXTRACT_MODEL=openai/gpt-4o-mini
  GENERATE_MODEL=google/gemini-2.5-flash-lite
  # mix providers freely — each gets its own client
"""

import asyncio
import os
import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

# ── concurrency limiter ───────────────────────────────────────────────────────

LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
llm_sem = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

# ── provider config ───────────────────────────────────────────────────────────

# Supported providers — only this list is validated.
# Model names are passed through as-is to the SDK (no whitelist).
SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"google", "openai", "anthropic"})

# Aliases → canonical provider name
_ALIASES: dict[str, str] = {
    "gemini": "google",
}

# Default models (only used when no model is configured)
_DEFAULTS: dict[str, str] = {
    "google":    "google/gemini-2.5-flash-lite",
    "openai":    "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
}

# Backward-compat: LLM_PROVIDER without prefix
_LEGACY_PROVIDER: str = os.getenv("LLM_PROVIDER", "google").lower()

# Per-provider client cache
_clients: dict[str, Any] = {}
_clients_lock = threading.Lock()


# ── model parsing ─────────────────────────────────────────────────────────────

def parse_model(model_str: str) -> tuple[str, str]:
    """Parse a model string into (provider, model_name).

    "openai/gpt-4o-mini"          →  ("openai", "gpt-4o-mini")
    "google/gemini-2.5-flash-lite" →  ("google", "gemini-2.5-flash-lite")
    "gemini/gemini-2.0-flash"      →  ("google", "gemini-2.0-flash")
    "gpt-4o-mini"                  →  (LLM_PROVIDER, "gpt-4o-mini")  # legacy

    Only the provider is validated. Model name is passed through as-is to the SDK.
    Raises ValueError for unknown providers.
    """
    if "/" in model_str:
        prefix, name = model_str.split("/", 1)
        provider = _ALIASES.get(prefix, prefix)
    else:
        # No prefix — fall back to LLM_PROVIDER env var
        provider = _ALIASES.get(_LEGACY_PROVIDER, _LEGACY_PROVIDER)
        name = model_str

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}. "
            f"Format: 'provider/model-name'  e.g. 'openai/gpt-4o-mini'"
        )
    return provider, name


def default_model(purpose: str) -> str:
    """Return the default model string (with provider prefix) for a purpose.

    purpose: "extract" | "generate"

    Returns e.g. "google/gemini-2.5-flash-lite"
    """
    provider = _ALIASES.get(_LEGACY_PROVIDER, _LEGACY_PROVIDER)
    return _DEFAULTS.get(provider, _DEFAULTS["google"])


# ── client factory ────────────────────────────────────────────────────────────

def get_client(provider: str) -> Any:
    """Return a lazily-initialised SDK client for the given provider."""
    if provider not in _clients:
        with _clients_lock:
            if provider not in _clients:
                _clients[provider] = _build_client(provider)
    return _clients[provider]


def _build_client(provider: str) -> Any:
    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        log.info("LLM client: openai (base_url=%s)", base_url or "default")
        return OpenAI(api_key=api_key, base_url=base_url)

    if provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        log.info("LLM client: anthropic")
        return anthropic.Anthropic(api_key=api_key)

    # google / gemini (default)
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    log.info("LLM client: google/gemini")
    return genai.Client(api_key=api_key)


# ── generation ────────────────────────────────────────────────────────────────

def llm_generate(
    prompt: str,
    model: str,
    system: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Generate text with the provider inferred from the model string.

    model: "openai/gpt-4o-mini", "google/gemini-2.5-flash-lite", etc.
    """
    provider, model_name = parse_model(model)
    client = get_client(provider)

    if provider == "openai":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()

    if provider == "anthropic":
        kwargs: dict = {
            "model":      model_name,
            "max_tokens": max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text.strip()

    # google / gemini
    from google.genai import types
    cfg_kwargs: dict = {"max_output_tokens": max_tokens}
    if system:
        cfg_kwargs["system_instruction"] = system
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    return resp.text.strip()
