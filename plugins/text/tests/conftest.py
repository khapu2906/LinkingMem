"""Shared fixtures — mock heavy deps before any plugin module is imported."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ── stub heavy packages so tests don't need torch / sentence-transformers / genai ──

def _make_module(name: str) -> ModuleType:
    m = ModuleType(name)
    sys.modules[name] = m
    return m


# sentence-transformers
st = _make_module("sentence_transformers")
st.SentenceTransformer = MagicMock  # type: ignore

# torch (imported transitively)
_make_module("torch")

# google.genai
google = _make_module("google")
google_genai = _make_module("google.genai")
google.genai = google_genai  # type: ignore

# openai / anthropic (optional)
_make_module("openai")
_make_module("anthropic")

# numpy — return a real zero array so embed tests work
import numpy as np  # noqa: E402  (numpy is a light dep — always available)

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402


# ── LLM stub ──────────────────────────────────────────────────────────────────

def _fake_llm(text: str, model: str, system: str | None, max_tokens: int) -> str:
    """Return canned responses used by all LLM-dependent tests."""
    if system and "extractor" in system.lower():
        return (
            '{"entities": [{"id": "e1", "name": "Alice", "type": "Person",'
            ' "full_context": "Alice, a person", "props": {}}], "relations": []}'
        )
    if system and "reasoning" in system.lower():
        return '{"answer": "reason answer", "follow_ups": [], "done": true}'
    return "generate answer"


@pytest.fixture()
def fake_llm_patch(monkeypatch):
    """Patch llm_generate in every module that imports it."""
    import text.llm as llm_mod
    monkeypatch.setattr(llm_mod, "llm_generate", _fake_llm)


@pytest.fixture()
def fake_embedder_patch(monkeypatch):
    """Patch the SentenceTransformer instance so tests skip model loading."""
    import text.embed as embed_mod

    class _FakeEmbedder:
        def encode(self, texts, **_kw):
            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(embed_mod, "_embedder", _FakeEmbedder(), raising=False)
    monkeypatch.setattr(embed_mod, "_DIM", 4, raising=False)


@pytest.fixture()
async def client(fake_embedder_patch, fake_llm_patch):  # noqa: F811
    """Full async HTTP test client against the FastAPI app."""
    from text.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
