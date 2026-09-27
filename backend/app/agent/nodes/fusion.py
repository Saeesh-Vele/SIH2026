"""fusion_node — answer from an optical scene and its SAR pair together.

Contract identical to `vqa_grounding_node`: ``{answer, evidence, confidence,
model_used}``. `evidence` is empty by design — fusion produces a described
scene, not geometry — so the metrics carry the fusion summary instead, and the
frontend renders the two inputs side by side rather than an overlay.
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

STEP_LABEL = "Run optical-SAR fusion"


async def fusion_node(state: GraphState) -> dict[str, Any]:
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return {}

    started = time.perf_counter()
    engine = get_registry().get_model("optical_sar_fusion")

    optical = asset_by_role(state, "optical")
    sar = asset_by_role(state, "sar")
    if optical is None or sar is None:
        # The validator rejects this shape before routing; reaching here means
        # the state was built by hand or the manifest changed underneath.
        return {
            "status": QueryStatus.FAILED,
            "error": "Optical-SAR fusion needs one optical image and one SAR image.",
            "model_used": getattr(engine, "model_used", "optical_sar_fusion"),
            "steps": [
                step(
                    STEP_LABEL,
                    TraceStepStatus.FAILED,
                    "missing optical/sar assets",
                    elapsed_ms(started),
                )
            ],
        }

    try:
        # Two encoder passes and a decode; both are GPU-bound and synchronous.
        result = await asyncio.to_thread(
            engine.infer,
            optical["stored_path"],
            sar["stored_path"],
            state["query"],
            max_new_tokens=state.get("parameters", {}).get("max_new_tokens"),
            temperature=state.get("parameters", {}).get("temperature"),
        )
        fields = honesty_fields(result)
    except ModelUnavailable as exc:
        logger.info("fusion unavailable: %s", exc)
        return {
            "status": QueryStatus.UNAVAILABLE,
            "error": str(exc),
            "model_used": getattr(engine, "model_used", "optical_sar_fusion"),
            "steps": [step(STEP_LABEL, TraceStepStatus.FAILED, str(exc), elapsed_ms(started))],
        }
    except Exception as exc:  # noqa: BLE001 - one bad inference must not kill the graph
        logger.exception("fusion inference failed")
        return {
            "status": QueryStatus.FAILED,
            "error": f"Fusion failed: {exc}",
            "model_used": getattr(engine, "model_used", "optical_sar_fusion"),
            "steps": [step(STEP_LABEL, TraceStepStatus.FAILED, str(exc)[:200], elapsed_ms(started))],
        }

    duration_ms = result.get("duration_ms") or elapsed_ms(started)
    stats = result.get("fusion_stats", {})
    description = result.get("fusion_description", "")

    # The summary is what the VQA model was told about the SAR half, so it is
    # shown rather than kept internal — otherwise half the reasoning is hidden.
    metrics = [{"label": "sensor summary", "value": description}] if description else []
    if "sar_share" in stats:
        metrics.append({"label": "SAR share", "value": f"{stats['sar_share']:.2f}"})
    if "agreement" in stats:
        metrics.append({"label": "sensor agreement", "value": f"{stats['agreement']:.2f}"})
    if result.get("fused_width"):
        metrics.append({"label": "fused width", "value": str(result["fused_width"])})
    metrics.append({"label": "decode", "value": f"{duration_ms} ms"})

    return {
        "answer": result["answer"],
        "evidence": result.get("evidence", []),
        **fields,
        "model_used": result.get("model_used", "optical_sar_fusion"),
        "metrics": metrics,
        "steps": [
            step(
                STEP_LABEL,
                TraceStepStatus.COMPLETE,
                degraded_detail(description or "fused optical and SAR embeddings", fields),
                duration_ms,
            )
        ],
    }
