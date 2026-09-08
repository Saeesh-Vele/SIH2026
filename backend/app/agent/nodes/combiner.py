"""output_combiner — settle the final answer shape, whichever path got here."""

from __future__ import annotations

from typing import Any

import time

from app.agent.state import GraphState, elapsed_ms, step
from app.models.schemas import QueryStatus, TraceStepStatus

#: What to say when a run ends with nothing to report. The node-level `error`
#: carries the specifics; this is the sentence that goes in the answer slot.
_FALLBACK_ANSWER: dict[QueryStatus, str] = {
    QueryStatus.REJECTED: "The bound imagery does not match what this question needs.",
    QueryStatus.UNAVAILABLE: "The model for this task is not available on this machine.",
    QueryStatus.FAILED: "The run did not complete.",
}


async def output_combiner(state: GraphState) -> dict[str, Any]:
    started = time.perf_counter()
    status = state.get("status", QueryStatus.OK)
    answer = state.get("answer", "")

    if status != QueryStatus.OK and not answer:
        answer = _FALLBACK_ANSWER.get(status, "No answer was produced.")

    evidence = state.get("evidence", [])
    # Overlays are the drawable subset of the evidence: anything the canvas
    # knows how to render. Non-geometric evidence stays out of it.
    overlays = [item for item in evidence if item.get("kind") in {"box", "mask"}]

    models_used = [m for m in (state.get("model_used"),) if m]
    if state.get("intent_model"):
        models_used.insert(0, f"{state['intent_model']} (intent)")

    return {
        "status": status,
        "answer": answer,
        "evidence": evidence,
        "confidence": state.get("confidence", 0.0) if status == QueryStatus.OK else 0.0,
        "metrics": state.get("metrics", []),
        "model_used": state.get("model_used", ""),
        "steps": [
            # Always COMPLETE: this node's job is to report the outcome, and
            # the node that actually failed is already marked above it.
            step(
                "Combine output",
                TraceStepStatus.COMPLETE,
                f"status {status.value}"
                + (f", {len(overlays)} overlay(s)" if overlays else "")
                + (f" — {'; '.join(models_used)}" if models_used else ""),
                elapsed_ms(started),
            )
        ],
    }
