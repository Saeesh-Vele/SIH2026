"""change_node — describe and locate what differs between two captures.

Contract identical to `vqa_grounding_node`: ``{answer, evidence, confidence,
model_used}``. What differs is the shape of the evidence — masks over the
changed regions rather than boxes around named objects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.agent.state import (
    GraphState,
    asset_by_role,
    degraded_detail,
    elapsed_ms,
    honesty_fields,
    step,
)
from app.core.model_registry import get_registry
from app.models.geochat import ModelUnavailable
from app.models.schemas import QueryStatus, TraceStepStatus

logger = logging.getLogger(__name__)

STEP_LABEL = "Run change detection"


async def change_node(state: GraphState) -> dict[str, Any]:
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return {}

    started = time.perf_counter()
    engine = get_registry().get_model("change_detection")

    t0 = asset_by_role(state, "t0")
    t1 = asset_by_role(state, "t1")
    if t0 is None or t1 is None:
        # The validator rejects this shape before routing, so reaching here
        # means the state was built by hand or the manifest changed underneath.
        return {
            "status": QueryStatus.FAILED,
            "error": "Change detection needs an earlier and a later capture.",
            "model_used": getattr(engine, "model_used", "change_detection"),
            "steps": [
                step(STEP_LABEL, TraceStepStatus.FAILED, "missing t0/t1 assets", elapsed_ms(started))
            ],
        }

    try:
        # Inference is synchronous and GPU-bound; keep the event loop free.
        result = await asyncio.to_thread(
            engine.infer,
            t0["stored_path"],
            t1["stored_path"],
            state["query"],
            max_new_tokens=state.get("parameters", {}).get("max_new_tokens"),
            temperature=state.get("parameters", {}).get("temperature"),
        )
        fields = honesty_fields(result)
    except ModelUnavailable as exc:
        logger.info("change detection unavailable: %s", exc)
        return {
            "status": QueryStatus.UNAVAILABLE,
            "error": str(exc),
            "model_used": getattr(engine, "model_used", "change_detection"),
            "steps": [step(STEP_LABEL, TraceStepStatus.FAILED, str(exc), elapsed_ms(started))],
        }
    except Exception as exc:  # noqa: BLE001 - one bad inference must not kill the graph
        logger.exception("change detection failed")
        return {
            "status": QueryStatus.FAILED,
            "error": f"Change detection failed: {exc}",
            "model_used": getattr(engine, "model_used", "change_detection"),
            "steps": [step(STEP_LABEL, TraceStepStatus.FAILED, str(exc)[:200], elapsed_ms(started))],
        }

    duration_ms = result.get("duration_ms") or elapsed_ms(started)
    evidence = result.get("evidence", [])
    changed = result.get("changed_fraction")

    metrics = [{"label": "changed regions", "value": str(len(evidence))}]
    if changed is not None:
        metrics.append({"label": "scene changed", "value": f"{changed * 100:.1f}%"})
    if result.get("grid"):
        metrics.append({"label": "diff grid", "value": str(result["grid"])})
    metrics.append({"label": "decode", "value": f"{duration_ms} ms"})

    return {
        "answer": result["answer"],
        "evidence": evidence,
        **fields,
        "model_used": result.get("model_used", "change_detection"),
        "metrics": metrics,
        "steps": [
            step(
                STEP_LABEL,
                TraceStepStatus.COMPLETE,
                degraded_detail(
                    f"{len(evidence)} changed region(s)"
                    + (f", {changed * 100:.1f}% of scene" if changed is not None else ""),
                    fields,
                ),
                duration_ms,
            )
        ],
    }
