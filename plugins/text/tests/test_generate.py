"""Tests for /generate and /reason endpoints + prompt builder helpers."""

import pytest
from text.schemas import ContextNode, ContextEdge, LlmHints, GenerateRequest, ReasonRequest


def _node(name="Alice", score=0.9):
    return ContextNode(name=name, node_type="Person", score=score, full_context=f"{name}, a person")


def _edge(frm="Alice", to="Acme"):
    return ContextEdge(from_node=frm, to_node=to, weight=0.8, edge_type="works_at")


# ── _build_prompt unit tests ──────────────────────────────────────────────────

class TestBuildPrompt:
    def setup_method(self):
        import text.generate as mod
        self.build = mod._build_prompt

    def test_nodes_appear_in_prompt(self):
        req = GenerateRequest(context=[_node("Alice"), _node("Bob")], query="Who?")
        p = self.build(req)
        assert "Alice" in p
        assert "Bob" in p

    def test_relations_section_present_when_edges_given(self):
        req = GenerateRequest(
            context=[_node(), _node("Acme")],
            relations=[_edge()],
            query="Who?",
        )
        p = self.build(req)
        assert "Relationships:" in p
        assert "works_at" in p

    def test_no_relations_section_when_empty(self):
        req = GenerateRequest(context=[_node()], query="Anything?")
        p = self.build(req)
        assert "Relationships:" not in p

    def test_extend_context_appended(self):
        req = GenerateRequest(
            context=[_node()],
            query="Q?",
            hints=LlmHints(extend_context=["Alice was born in 1990"]),
        )
        p = self.build(req)
        assert "Alice was born in 1990" in p
        assert "Additional context:" in p

    def test_rules_appended_to_instructions(self):
        req = GenerateRequest(
            context=[_node()],
            query="Q?",
            hints=LlmHints(rules=["Answer in French"]),
        )
        p = self.build(req)
        assert "Answer in French" in p

    def test_system_prompt_override(self):
        req = GenerateRequest(
            context=[_node()],
            query="Q?",
            hints=LlmHints(system_prompt="You are a pirate."),
        )
        p = self.build(req)
        assert "You are a pirate." in p
        assert "knowledge graph" not in p.lower().split("you are a pirate.")[0]

    def test_nodes_sorted_by_score_descending(self):
        low  = _node("LowScore", score=0.1)
        high = _node("HighScore", score=0.95)
        req  = GenerateRequest(context=[low, high], query="?")
        p    = self.build(req)
        assert p.index("HighScore") < p.index("LowScore")


# ── /generate endpoint ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGenerateEndpoint:
    async def test_basic_generate(self, client):
        resp = await client.post(
            "/generate",
            json={"context": [_node().model_dump()], "query": "Who is Alice?"},
        )
        assert resp.status_code == 200
        assert "answer" in resp.json()

    async def test_empty_context_returns_no_context_message(self, client):
        resp = await client.post("/generate", json={"context": [], "query": "Q?"})
        assert resp.status_code == 200
        assert "No relevant context" in resp.json()["answer"]

    async def test_generate_with_hints(self, client):
        resp = await client.post(
            "/generate",
            json={
                "context":   [_node().model_dump()],
                "query":     "Q?",
                "hints":     {"rules": ["Answer in bullet points"], "extend_context": ["Extra fact"]},
            },
        )
        assert resp.status_code == 200


# ── /reason endpoint ─────────────────────────────────────────────────────────

class TestBuildReasonHelpers:
    def setup_method(self):
        import text.reason as mod
        self.build_prompt   = mod._build_reason_prompt
        self.effective_sys  = mod._effective_system

    def test_effective_system_default(self):
        from text.reason import _REASON_SYSTEM
        req = ReasonRequest(context=[_node()], query="Q?")
        assert self.effective_sys(req) == _REASON_SYSTEM

    def test_effective_system_override(self):
        req = ReasonRequest(
            context=[_node()], query="Q?",
            hints=LlmHints(system_prompt="Custom reason system"),
        )
        assert self.effective_sys(req) == "Custom reason system"

    def test_effective_system_rules_appended(self):
        req = ReasonRequest(
            context=[_node()], query="Q?",
            hints=LlmHints(rules=["Only request follow_ups if critical"]),
        )
        result = self.effective_sys(req)
        assert "Only request follow_ups if critical" in result

    def test_extend_context_in_prompt(self):
        req = ReasonRequest(
            context=[_node()], query="Q?", iteration=0, max_iterations=2,
            hints=LlmHints(extend_context=["Alice joined in 2020"]),
        )
        p = self.build_prompt(req)
        assert "Alice joined in 2020" in p
        assert "Additional context:" in p

    def test_last_iteration_note_present(self):
        req = ReasonRequest(context=[_node()], query="Q?", iteration=1, max_iterations=2)
        p = self.build_prompt(req)
        assert "LAST iteration" in p

    def test_first_iteration_no_last_note(self):
        req = ReasonRequest(context=[_node()], query="Q?", iteration=0, max_iterations=2)
        p = self.build_prompt(req)
        assert "LAST iteration" not in p


@pytest.mark.asyncio
class TestReasonEndpoint:
    async def test_basic_reason(self, client):
        resp = await client.post(
            "/reason",
            json={"context": [_node().model_dump()], "query": "Why?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "done" in data
        assert "follow_ups" in data

    async def test_empty_context_returns_done(self, client):
        resp = await client.post("/reason", json={"context": [], "query": "Why?"})
        assert resp.status_code == 200
        assert resp.json()["done"] is True

    async def test_reason_with_hints(self, client):
        resp = await client.post(
            "/reason",
            json={
                "context": [_node().model_dump()],
                "query":   "Why?",
                "hints":   {"rules": ["Never ask for follow-ups on the last iteration"]},
            },
        )
        assert resp.status_code == 200
