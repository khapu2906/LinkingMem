"""Answer generation — /generate endpoint."""

import os
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .llm import llm_generate, default_model, llm_sem
from .schemas import GenerateRequest, GenerateResponse

log = logging.getLogger(__name__)

router = APIRouter()


def _build_prompt(req: GenerateRequest) -> str:
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

    # extra context snippets injected by the caller
    extra_section = ""
    if req.hints.extend_context:
        extra_lines = "\n".join(f"- {c}" for c in req.hints.extend_context)
        extra_section = f"\nAdditional context:\n{extra_lines}"

    instructions = (
        "- Answer based strictly on the entities and relationships above.\n"
        "- Cite entity names directly (e.g. \"Alice works at Acme Corp\").\n"
        "- If the context is insufficient, clearly state what is missing.\n"
        "- Be concise — one to three sentences is ideal."
    )
    if req.hints.rules:
        instructions += "\n" + "\n".join(f"- {r}" for r in req.hints.rules)

    system = req.hints.system_prompt or "You are a knowledge graph question-answering system."

    return (
        f"{system}\n\n"
        "Below is the relevant subgraph retrieved for this query.\n\n"
        f"Entities (sorted by relevance):\n{chr(10).join(entity_lines)}{relations_section}{extra_section}\n\n"
        f"Question: {req.query}\n\n"
        f"Instructions:\n{instructions}"
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, _: None = Depends(verify_auth)):
    if not req.context:
        return GenerateResponse(answer="No relevant context found in the knowledge graph.")

    model  = os.getenv("GENERATE_MODEL", default_model("generate"))
    prompt = _build_prompt(req)

    try:
        async with llm_sem:
            answer = await run_in_threadpool(
                llm_generate, prompt, model, None, 1000,
            )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error("generate LLM call failed: %s", e)
        raise HTTPException(502, f"LLM generate call failed: {e}")

    if not answer:
        raise HTTPException(500, "LLM returned empty answer")

    return GenerateResponse(answer=answer)
