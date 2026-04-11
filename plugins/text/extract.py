"""Entity extraction — /extract endpoint."""

import os
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .auth import verify_auth
from .llm import llm_generate, default_model, llm_sem
from .schemas import ExtractRequest, ExtractResponse

log = logging.getLogger(__name__)

router = APIRouter()

_SYSTEM_PROMPT = """You are a knowledge graph extractor.
Given text, extract entities and relations.
Return ONLY valid JSON with this exact schema:
{
  "entities": [
    {
      "id": "e1",
      "name": "Alice",
      "type": "Person|Company|Concept|Event|Location|...",
      "full_context": "Alice (Person): senior ML researcher at Google Brain, specialises in transformer architectures and NLP",
      "props": {"role": "researcher", "organization": "Google Brain"}
    }
  ],
  "relations": [
    {
      "from": "e1",
      "to": "e2",
      "type": "works_at",
      "weight": 0.9,
      "full_context": "Alice works at Google Brain as a senior ML researcher since 2018, focusing on large language models"
    }
  ]
}
Rules:
- "from" and "to" must reference entity ids defined in "entities"
- weight must be between 0.0 and 1.0
- full_context for entities: rich one-sentence description combining name, type, role, and key attributes (used for semantic search)
- full_context for relations: rich one-sentence description of the relationship including context and specifics (used for semantic search)
- props should capture meaningful structured attributes (role, location, date, etc.)
- Return empty arrays if no entities or relations are found
No explanation, no markdown, just JSON."""


def _build_system_prompt(hints) -> str:
    """Build the effective system prompt, applying any caller hints."""
    if hints.system_prompt:
        return hints.system_prompt
    if not hints.rules:
        return _SYSTEM_PROMPT
    extra = "\n".join(f"- {r}" for r in hints.rules)
    return f"{_SYSTEM_PROMPT}\n\nAdditional rules:\n{extra}"


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest, _: None = Depends(verify_auth)):
    model       = os.getenv("EXTRACT_MODEL", default_model("extract"))
    system_text = _build_system_prompt(req.hints)

    try:
        async with llm_sem:
            raw = await run_in_threadpool(
                llm_generate,
                req.text, model, system_text, 2000,
            )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error("extract LLM call failed: %s", e)
        raise HTTPException(502, f"LLM extract call failed: {e}")

    # strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("extract JSON parse error: %s\nraw: %s", e, raw[:200])
        raise HTTPException(500, f"LLM returned invalid JSON: {e}")

    entities  = data.get("entities",  [])
    relations = data.get("relations", [])

    # drop relations that reference unknown entity ids
    valid_ids = {e["id"] for e in entities if "id" in e}
    relations = [r for r in relations if r.get("from") in valid_ids and r.get("to") in valid_ids]

    return ExtractResponse(entities=entities, relations=relations)
