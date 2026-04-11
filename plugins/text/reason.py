"""Multi-hop reasoning — /reason endpoint.

The Rust engine calls this iteratively:
  iteration 0: initial context
  iteration 1+: enriched context from follow-up BFS expansions

The LLM returns one of two shapes:
  done=True  → final answer; Rust stops iterating
  done=False → follow_ups list of entity names; Rust expands them and calls again
"""

import json
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .llm import llm_generate, default_model, llm_sem
from .schemas import ReasonRequest, ReasonResponse

log = logging.getLogger(__name__)

router = APIRouter()

_REASON_SYSTEM = """You are a knowledge graph reasoning assistant.
You will receive a knowledge graph subgraph and a question.
Your task is to decide whether the context is sufficient to answer the question,
or whether you need to explore more of the graph first.

Respond with ONLY valid JSON in one of two forms:

Form 1 — you can answer now (done=true):
{
  "answer": "<your final answer here>",
  "follow_ups": [],
  "done": true
}

Form 2 — you need more context (done=false):
{
  "answer": "<brief explanation of what is missing>",
  "follow_ups": ["EntityName1", "EntityName2"],
  "done": false
}

Rules:
- follow_ups must be exact entity names you want to explore (max 5).
- Only request follow_ups when the missing information would materially change the answer.
- If you are on the last allowed iteration, always set done=true and give your best answer.
- No markdown, no explanation — only the JSON object."""


def _effective_system(req: ReasonRequest) -> str:
    if req.hints.system_prompt:
        return req.hints.system_prompt
    if not req.hints.rules:
        return _REASON_SYSTEM
    extra = "\n".join(f"- {r}" for r in req.hints.rules)
    return f"{_REASON_SYSTEM}\n\nAdditional rules:\n{extra}"


def _build_reason_prompt(req: ReasonRequest) -> str:
    sorted_nodes = sorted(req.context, key=lambda n: n.score, reverse=True)

    entity_lines = []
    for n in sorted_nodes:
        props_str = (" | " + ", ".join(f"{k}: {v}" for k, v in n.props.items() if v)) if n.props else ""
        ctx_str   = f" — {n.full_context}" if n.full_context else ""
        entity_lines.append(f"- [{n.node_type}] {n.name} (relevance: {n.score:.2f}){props_str}{ctx_str}")

    relation_lines = []
    for r in req.relations:
        label = f" [{r.edge_type}]" if r.edge_type else ""
        relation_lines.append(f"- {r.from_node}{label}→ {r.to_node} (strength: {r.weight:.2f})")

    relations_section = ("\nRelationships:\n" + "\n".join(relation_lines)) if relation_lines else ""

    extra_section = ""
    if req.hints.extend_context:
        extra_lines = "\n".join(f"- {c}" for c in req.hints.extend_context)
        extra_section = f"\nAdditional context:\n{extra_lines}"

    last_iter_note = (
        "\n⚠️  This is the LAST iteration — you MUST set done=true and give your best answer."
        if req.iteration >= req.max_iterations - 1 else ""
    )

    return (
        f"Reasoning iteration {req.iteration + 1} of {req.max_iterations}.{last_iter_note}\n\n"
        f"Entities (sorted by relevance):\n{chr(10).join(entity_lines)}{relations_section}{extra_section}\n\n"
        f"Question: {req.query}"
    )


@router.post("/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest, _: None = Depends(verify_auth)):
    if not req.context:
        return ReasonResponse(
            answer="No relevant context found in the knowledge graph.",
            follow_ups=[],
            done=True,
        )

    model       = os.getenv("GENERATE_MODEL", default_model("generate"))
    prompt      = _build_reason_prompt(req)
    system_text = _effective_system(req)

    try:
        async with llm_sem:
            raw = await run_in_threadpool(
                llm_generate, prompt, model, system_text, 1000,
            )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error("reason LLM call failed: %s", e)
        raise HTTPException(502, f"LLM reason call failed: {e}")

    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        log.error("reason JSON parse error: %s\nraw: %s", e, raw[:200])
        # Graceful degradation: treat LLM text as final answer
        return ReasonResponse(answer=raw.strip(), follow_ups=[], done=True)

    return ReasonResponse(
        answer     = str(data.get("answer", "")),
        follow_ups = [str(f) for f in data.get("follow_ups", []) if f][:5],
        done       = bool(data.get("done", True)),
    )
