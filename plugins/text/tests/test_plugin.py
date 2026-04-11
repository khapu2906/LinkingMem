"""Text plugin tests — pytest.

Run from plugins/text/:
    uv run pytest tests/ -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from text.main import app

client = TestClient(app)

# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

# ── /info ─────────────────────────────────────────────────────────────────────

def test_info_returns_expected_fields():
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    for field in ("name", "version", "capabilities", "embed_model", "llm_provider"):
        assert field in body

# ── /embed ────────────────────────────────────────────────────────────────────

@patch("text.embed.get_embedder")
def test_embed_returns_vectors(mock_get):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_get.return_value = mock_model

    r = client.post("/embed", json={"texts": ["hello", "world"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["vectors"]) == 2
    assert body["dim"] == 3
    assert "model" in body


@patch("text.embed.get_embedder")
def test_embed_encode_kwargs(mock_get):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 4))
    mock_get.return_value = mock_model

    client.post("/embed", json={"texts": ["test"]})
    kwargs = mock_model.encode.call_args[1]
    assert kwargs.get("normalize_embeddings") is True
    assert "batch_size" in kwargs
    assert kwargs.get("show_progress_bar") is False


def test_embed_empty_texts_400():
    r = client.post("/embed", json={"texts": []})
    assert r.status_code == 400


def test_embed_missing_field_422():
    r = client.post("/embed", json={"wrong": []})
    assert r.status_code == 422

# ── /extract ──────────────────────────────────────────────────────────────────

_GOOD_JSON = """{
  "entities": [{"id": "e1", "name": "Alice", "type": "Person", "props": {}}],
  "relations": [{"from": "e1", "to": "e2", "type": "knows", "weight": 0.9}]
}"""


@patch("text.extract.llm_generate", return_value=_GOOD_JSON)
def test_extract_returns_entities(mock_gen):
    r = client.post("/extract", json={"text": "Alice knows Bob."})
    assert r.status_code == 200
    body = r.json()
    assert body["entities"][0]["name"] == "Alice"
    # relation referencing unknown id "e2" should be dropped
    assert body["relations"] == []


@patch("text.extract.llm_generate", return_value="```json\n" + _GOOD_JSON + "\n```")
def test_extract_strips_markdown_fences(mock_gen):
    r = client.post("/extract", json={"text": "test"})
    assert r.status_code == 200
    assert "entities" in r.json()


@patch("text.extract.llm_generate", return_value="NOT JSON")
def test_extract_bad_json_500(mock_gen):
    r = client.post("/extract", json={"text": "test"})
    assert r.status_code == 500


def test_extract_missing_field_422():
    r = client.post("/extract", json={})
    assert r.status_code == 422

# ── /generate ─────────────────────────────────────────────────────────────────

@patch("text.generate.llm_generate", return_value="The answer is 42.")
def test_generate_returns_answer(mock_gen):
    payload = {
        "context": [{"id": 0, "name": "Alice", "node_type": "Person", "props": {"role": "CEO"}}],
        "query": "Who is CEO?",
    }
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    assert r.json()["answer"] == "The answer is 42."


def test_generate_empty_context_fallback():
    r = client.post("/generate", json={"context": [], "query": "anything"})
    assert r.status_code == 200
    assert "No relevant context" in r.json()["answer"]


@patch("text.generate.llm_generate", return_value="ok")
def test_generate_prompt_contains_node_and_query(mock_gen):
    payload = {
        "context": [{"id": 0, "name": "GraphEngine", "node_type": "Product", "props": {}}],
        "query": "What is it?",
    }
    client.post("/generate", json=payload)

    prompt = mock_gen.call_args[1]["prompt"] if mock_gen.call_args[1] else mock_gen.call_args[0][0]
    assert "GraphEngine" in prompt
    assert "What is it?" in prompt


def test_generate_missing_query_422():
    r = client.post("/generate", json={"context": []})
    assert r.status_code == 422

# ── auth ──────────────────────────────────────────────────────────────────────

@patch("text.auth.PLUGIN_AUTH_TOKEN", "test-secret")
@patch("text.embed.get_embedder")
def test_embed_requires_auth_when_token_set(mock_get):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1]])
    mock_get.return_value = mock_model

    # no token → 401
    r = client.post("/embed", json={"texts": ["hi"]})
    assert r.status_code == 401

    # correct token → 200
    r = client.post("/embed", json={"texts": ["hi"]},
                    headers={"Authorization": "Bearer test-secret"})
    assert r.status_code == 200
