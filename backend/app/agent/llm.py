"""Intent classification via OpenRouter.

One small chat completion that maps a free-text question onto the intent
taxonomy. OpenRouter fronts both DeepSeek and Gemini, so `openrouter_model`
switches provider without touching this code.

The classifier is best-effort by design: no key, a timeout, a rate limit or an
unparseable reply all fall back to the keyword heuristic rather than failing the
request. Which path ran is recorded in the trace, so a silent downgrade is still
a visible one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.models.schemas import Intent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You classify questions asked about satellite and aerial imagery.

Reply with JSON only, no prose, no code fence:
{"intent": "<label>", "confidence": <0-1>, "reason": "<max 12 words>"}

Labels:
- single_image_vqa: a question about the content of one image (counts, presence,
  attributes, description) that does not need the object's location marked.
- single_image_grounding: asks where something is in one image, or to locate,
  point out, mark, outline or box an object.
- change_vqa: compares two captures of the same place over time - what changed,
  what was added or removed, growth, before/after.
- optical_sar_fusion: involves SAR or radar data, or asks to see through cloud,
  haze or darkness by combining an optical image with a radar one.

Pick the single best label. If the question is ambiguous, prefer single_image_vqa."""

_GROUNDING_WORDS = (
    "where", "locate", "location", "mark", "outline", "box", "bounding",
    "point out", "highlight", "show me the", "pinpoint", "segment",
)
_CHANGE_WORDS = (
    "change", "changed", "before", "after", "temporal", "difference", "since",
    "grew", "growth", "expanded", "new construction", "added", "removed",
    "between the two", "bi-temporal",
)
_FUSION_WORDS = (
    "sar", "radar", "backscatter", "fusion", "fuse", "cloud", "cloudy",
    "overcast", "all-weather", "through the cloud", "polarisation", "polarization",
)


@dataclass
class IntentDecision:
    intent: Intent
    confidence: float
    reason: str
    source: str  # "llm" | "heuristic" | "forced"
    model: str | None = None


def classify_heuristic(query: str) -> IntentDecision:
    """Keyword fallback. Ordered most specific first."""
    lowered = query.lower()

    if any(word in lowered for word in _FUSION_WORDS):
        return IntentDecision(Intent.OPTICAL_SAR_FUSION, 0.5, "radar or cloud keyword", "heuristic")
    if any(word in lowered for word in _CHANGE_WORDS):
        return IntentDecision(Intent.CHANGE_VQA, 0.5, "temporal comparison keyword", "heuristic")
    if any(word in lowered for word in _GROUNDING_WORDS):
        return IntentDecision(
            Intent.SINGLE_IMAGE_GROUNDING, 0.5, "localisation keyword", "heuristic"
        )
    return IntentDecision(Intent.SINGLE_IMAGE_VQA, 0.4, "no specialist keyword", "heuristic")


def _extract_json(text: str) -> dict | None:
    """Tolerate a code fence or stray prose around the object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.S)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def classify_intent(query: str) -> IntentDecision:
    """Ask the LLM; fall back to keywords on any failure."""
    settings = get_settings()

    if not settings.openrouter_api_key:
        decision = classify_heuristic(query)
        decision.reason = f"{decision.reason} (no OPENROUTER_API_KEY set)"
        return decision

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if settings.openrouter_referer:
        headers["HTTP-Referer"] = settings.openrouter_referer
        headers["X-Title"] = settings.app_name

    body = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=settings.openrouter_timeout_s) as client:
            response = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - any failure downgrades, never raises
        logger.warning("intent LLM unavailable (%s); using heuristic", exc)
        decision = classify_heuristic(query)
        decision.reason = f"{decision.reason} (LLM unreachable)"
        return decision

    parsed = _extract_json(content or "")
    if parsed is None:
        logger.warning("intent LLM returned unparseable content: %r", content)
        decision = classify_heuristic(query)
        decision.reason = f"{decision.reason} (LLM reply unparseable)"
        return decision

    try:
        intent = Intent(str(parsed.get("intent", "")).strip())
    except ValueError:
        logger.warning("intent LLM returned unknown label: %r", parsed.get("intent"))
        decision = classify_heuristic(query)
        decision.reason = f"{decision.reason} (LLM label unknown)"
        return decision

    raw_confidence = parsed.get("confidence", 0.7)
    try:
        confidence = min(max(float(raw_confidence), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.7

    return IntentDecision(
        intent=intent,
        confidence=confidence,
        reason=str(parsed.get("reason", ""))[:120] or "classified by LLM",
        source="llm",
        model=settings.openrouter_model,
    )
