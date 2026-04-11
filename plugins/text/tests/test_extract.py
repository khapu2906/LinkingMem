"""Tests for /extract endpoint and _build_system_prompt helper."""

import pytest
from text.schemas import LlmHints


class TestBuildSystemPrompt:
    """Unit-test the prompt builder without touching LLM or network."""

    def setup_method(self):
        import text.extract as mod
        self.build = mod._build_system_prompt

    def test_returns_default_when_no_hints(self):
        from text.extract import _SYSTEM_PROMPT
        h = LlmHints()
        result = self.build(h)
        assert result == _SYSTEM_PROMPT

    def test_system_prompt_override_replaces_default(self):
        h = LlmHints(system_prompt="MY CUSTOM PROMPT")
        result = self.build(h)
        assert result == "MY CUSTOM PROMPT"

    def test_rules_are_appended(self):
        h = LlmHints(rules=["Focus on people only", "Ignore dates"])
        result = self.build(h)
        assert "Focus on people only" in result
        assert "Ignore dates" in result
        assert "Additional rules:" in result

    def test_system_prompt_overrides_rules(self):
        """system_prompt takes full precedence — rules are ignored."""
        h = LlmHints(system_prompt="Override", rules=["Should be ignored"])
        result = self.build(h)
        assert result == "Override"
        assert "Should be ignored" not in result


@pytest.mark.asyncio
class TestExtractEndpoint:
    async def test_basic_extraction(self, client):
        resp = await client.post("/extract", json={"text": "Alice works at Acme."})
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "relations" in data

    async def test_extract_with_rules(self, client):
        resp = await client.post(
            "/extract",
            json={
                "text": "Bob founded TechCorp in 2010.",
                "hints": {"rules": ["Focus on organisations only"]},
            },
        )
        assert resp.status_code == 200

    async def test_extract_with_system_prompt_override(self, client):
        resp = await client.post(
            "/extract",
            json={
                "text": "Carol leads the AI team.",
                "hints": {"system_prompt": "Return {\"entities\":[], \"relations\":[]}"},
            },
        )
        # Stub LLM ignores system_prompt content; just verify 200
        assert resp.status_code == 200
