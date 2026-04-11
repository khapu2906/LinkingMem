"""Tests for /embed endpoint."""

import pytest


@pytest.mark.asyncio
class TestEmbedEndpoint:
    async def test_embed_single_text(self, client, fake_embedder_patch):
        resp = await client.post("/embed", json={"texts": ["hello world"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "vectors" in data
        assert len(data["vectors"]) == 1
        assert data["dim"] == 4

    async def test_embed_multiple_texts(self, client, fake_embedder_patch):
        resp = await client.post("/embed", json={"texts": ["foo", "bar", "baz"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["vectors"]) == 3
        assert all(len(v) == 4 for v in data["vectors"])

    async def test_embed_empty_list_returns_400_or_empty(self, client, fake_embedder_patch):
        resp = await client.post("/embed", json={"texts": []})
        # Either 400 (validation) or 200 with empty vectors — both are acceptable
        assert resp.status_code in (200, 400, 422)

    async def test_embed_response_has_model_field(self, client, fake_embedder_patch):
        resp = await client.post("/embed", json={"texts": ["test"]})
        assert resp.status_code == 200
        assert "model" in resp.json()
