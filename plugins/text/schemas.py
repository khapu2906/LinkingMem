"""Pydantic request/response models shared across all endpoints."""

from pydantic import BaseModel


# ── LLM customization hints ───────────────────────────────────────────────────

class LlmHints(BaseModel):
    """Per-request LLM behaviour overrides forwarded from the Rust API layer."""

    # Fully replace the operation's default system prompt.
    system_prompt: str | None = None

    # Extra rules appended after the default rules block (extract) or
    # instructions section (generate/reason).
    rules: list[str] = []

    # Extra free-text snippets injected into the prompt context
    # (generate/reason only — ignored by extract).
    extend_context: list[str] = []


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int
    model: str


class ExtractRequest(BaseModel):
    text: str
    hints: LlmHints = LlmHints()


class ExtractResponse(BaseModel):
    entities: list[dict]
    relations: list[dict]


class ContextNode(BaseModel):
    id:           int   = 0
    name:         str
    node_type:    str
    props:        dict  = {}
    score:        float = 0.0
    full_context: str   = ""


class ContextEdge(BaseModel):
    from_node: str
    to_node:   str
    weight:    float = 1.0
    edge_type: str   = ""


class GenerateRequest(BaseModel):
    context:   list[ContextNode]
    relations: list[ContextEdge] = []
    query:     str
    hints:     LlmHints = LlmHints()


class GenerateResponse(BaseModel):
    answer: str


# ── multi-hop reasoning ───────────────────────────────────────────────────────

class ReasonRequest(BaseModel):
    context:        list[ContextNode]
    relations:      list[ContextEdge] = []
    query:          str
    iteration:      int = 0   # current hop (0-based)
    max_iterations: int = 2   # total hops allowed
    hints:          LlmHints = LlmHints()


class ReasonResponse(BaseModel):
    answer:     str            # final answer (done=True) or partial reasoning
    follow_ups: list[str] = [] # entity names Rust should look up next
    done:       bool = True    # True = final answer, False = need more hops
