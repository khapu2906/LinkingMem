"""Unit tests for Pydantic schemas — no heavy deps needed."""

from text.schemas import (
    LlmHints,
    ExtractRequest,
    GenerateRequest,
    ReasonRequest,
    ContextNode,
)


class TestLlmHints:
    def test_default_is_empty(self):
        h = LlmHints()
        assert h.system_prompt is None
        assert h.rules == []
        assert h.extend_context == []

    def test_partial_construction(self):
        h = LlmHints(rules=["Answer in French"])
        assert h.rules == ["Answer in French"]
        assert h.system_prompt is None
        assert h.extend_context == []

    def test_full_construction(self):
        h = LlmHints(
            system_prompt="Custom system",
            rules=["Rule 1", "Rule 2"],
            extend_context=["Extra fact"],
        )
        assert h.system_prompt == "Custom system"
        assert len(h.rules) == 2
        assert len(h.extend_context) == 1


class TestExtractRequest:
    def test_default_hints(self):
        req = ExtractRequest(text="hello")
        assert isinstance(req.hints, LlmHints)
        assert req.hints.rules == []

    def test_custom_hints(self):
        req = ExtractRequest(text="hello", hints=LlmHints(rules=["Focus on people"]))
        assert req.hints.rules == ["Focus on people"]


class TestGenerateRequest:
    def _node(self):
        return ContextNode(name="Alice", node_type="Person")

    def test_default_hints(self):
        req = GenerateRequest(context=[self._node()], query="Who?")
        assert isinstance(req.hints, LlmHints)
        assert req.hints.extend_context == []

    def test_extend_context_passed_through(self):
        req = GenerateRequest(
            context=[self._node()],
            query="Who?",
            hints=LlmHints(extend_context=["Alice is 30 years old"]),
        )
        assert req.hints.extend_context == ["Alice is 30 years old"]


class TestReasonRequest:
    def _node(self):
        return ContextNode(name="Bob", node_type="Person")

    def test_default_fields(self):
        req = ReasonRequest(context=[self._node()], query="Why?")
        assert req.iteration == 0
        assert req.max_iterations == 2
        assert isinstance(req.hints, LlmHints)

    def test_hints_forwarded(self):
        req = ReasonRequest(
            context=[self._node()],
            query="Why?",
            hints=LlmHints(system_prompt="Custom reason system"),
        )
        assert req.hints.system_prompt == "Custom reason system"
