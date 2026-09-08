"""intent_classifier — decide what the question is actually asking for."""

from __future__ import annotations

import time
from typing import Any

from app.agent.llm import IntentDecision, classify_intent
from app.agent.state import GraphState, step
from app.models.schemas import MODE_TO_INTENTS, TraceStepStatus, UploadMode


def reconcile_with_upload(
    decision: IntentDecision, mode: UploadMode | None
) -> tuple[IntentDecision, str | None]:
    """Correct a label the bound imagery rules out.

    Classification reads the words; the upload says what is actually on the
    table. Two of the three modes admit exactly one intent — a bi-temporal pair
    can only be compared over time, an optical+SAR pair can only be fused — so
    when the classifier picks something else for one of those, it is wrong about
    the request rather than the request being wrong about the imagery. Switching
    beats letting the validator reject a run nobody could have phrased better.

    Single-image mode admits two intents, so it never triggers this; a change or
    fusion question asked of one image is a real mismatch, and the validator
    still rejects it with an explanation.

    Returns the decision to use and, when it changed, a note for the trace.
    """
    if mode is None:
        return decision, None

    candidates = MODE_TO_INTENTS.get(mode, ())
    if len(candidates) != 1 or decision.intent is candidates[0]:
        return decision, None

    was = decision.intent
    corrected = IntentDecision(
        intent=candidates[0],
        confidence=decision.confidence,
        reason=f"{decision.reason}; corrected from {was.value} by the {mode.value} upload",
        source=decision.source,
        model=decision.model,
    )
    return corrected, f"{was.value} -> {candidates[0].value} (bound imagery is {mode.value})"


async def intent_classifier(state: GraphState) -> dict[str, Any]:
    if state.get("status") and state["status"].value != "ok":
        return {}  # intake already rejected the request

    forced = state.get("forced_intent")
    if forced is not None:
        return {
            "intent": forced,
            "intent_confidence": 1.0,
            "intent_source": "forced",
            "intent_reason": "supplied by the caller",
            "steps": [
                step("Classify intent", TraceStepStatus.COMPLETE, f"forced to {forced.value}")
            ],
        }

    started = time.perf_counter()
    decision = await classify_intent(state["query"])
    decision, correction = reconcile_with_upload(decision, state.get("upload_mode"))
    duration_ms = int((time.perf_counter() - started) * 1000)

    via = decision.model if decision.source == "llm" else decision.source
    detail = f"{decision.intent.value} via {via} — {decision.reason}"
    if correction:
        detail = f"{detail} [{correction}]"

    return {
        "intent": decision.intent,
        "intent_confidence": decision.confidence,
        "intent_source": decision.source,
        "intent_reason": decision.reason,
        "intent_model": decision.model,
        "steps": [
            step("Classify intent", TraceStepStatus.COMPLETE, detail, duration_ms)
        ],
    }
