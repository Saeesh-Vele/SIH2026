"""intent_classifier — decide what the question is actually asking for."""

from __future__ import annotations

import time
from typing import Any

from app.agent.llm import classify_intent
from app.agent.state import GraphState, step
from app.models.schemas import TraceStepStatus


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
    duration_ms = int((time.perf_counter() - started) * 1000)

    via = decision.model if decision.source == "llm" else decision.source
    return {
        "intent": decision.intent,
        "intent_confidence": decision.confidence,
        "intent_source": decision.source,
        "intent_reason": decision.reason,
        "intent_model": decision.model,
        "steps": [
            step(
                "Classify intent",
                TraceStepStatus.COMPLETE,
                f"{decision.intent.value} via {via} — {decision.reason}",
                duration_ms,
            )
        ],
    }
