"""vqa_grounding_node — the one specialist that runs a real checkpoint today."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.agent.state import GraphState, elapsed_ms, step
from app.core.model_registry import get_registry
from app.models.geochat import ModelUnavailable
from app.models.schemas import Intent, QueryStatus, TraceStepStatus

logger = logging.getLogger(__name__)

#: Appended for captioning intents. Same checkpoint as single-image VQA — the
#: two intents differ in the prompt, not the model. LLaVA-1.5 captions natively,
#: so this only widens the scope of the answer from one fact to the whole scene.
CAPTION_SUFFIX = (
    " Describe the whole scene: the main objects and how many, the land cover, "
    "and how they are laid out."
)


def _primary_asset(state: GraphState) -> dict[str, Any]:
    assets = state.get("assets", [])
    for role in ("primary", "optical", "t1"):
        for asset in assets:
            if asset.get("role") == role:
                return asset
    return assets[0]


async def vqa_grounding_node(state: GraphState) -> dict[str, Any]:
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return {}

    asset = _primary_asset(state)
    question = state["query"]
    if state.get("intent") is Intent.SINGLE_IMAGE_CAPTIONING:
        question = f"{question.rstrip()}{CAPTION_SUFFIX}"

    engine = get_registry().get_model("vqa_grounding")
    started = time.perf_counter()

    try:
        # Inference is synchronous and GPU-bound; keep the event loop free.
        result = await asyncio.to_thread(
            engine.infer,
            asset["stored_path"],
            question,
            max_new_tokens=state.get("parameters", {}).get("max_new_tokens"),
            temperature=state.get("parameters", {}).get("temperature"),
        )
    except ModelUnavailable as exc:
        logger.info("vqa model unavailable: %s", exc)
        return {
            "status": QueryStatus.UNAVAILABLE,
            "error": str(exc),
            "model_used": getattr(engine, "model_used", "vqa_grounding"),
            "steps": [step("Run VQA model", TraceStepStatus.FAILED, str(exc), elapsed_ms(started))],
        }
    except Exception as exc:  # noqa: BLE001 - one bad inference must not kill the graph
        logger.exception("vqa inference failed")
        return {
            "status": QueryStatus.FAILED,
            "error": f"Inference failed: {exc}",
            "model_used": getattr(engine, "model_used", "vqa_grounding"),
            "steps": [step("Run VQA model", TraceStepStatus.FAILED, str(exc)[:200], elapsed_ms(started))],
        }

    duration_ms = result.get("duration_ms") or int((time.perf_counter() - started) * 1000)
    evidence = result.get("evidence", [])
    metrics = [
        {"label": "tokens", "value": str(result.get("tokens", 0))},
        {"label": "grounded boxes", "value": str(len(evidence))},
        {"label": "decode", "value": f"{duration_ms} ms"},
    ]

    return {
        "answer": result["answer"],
        "evidence": evidence,
        "confidence": float(result.get("confidence", 0.0)),
        "model_used": result.get("model_used", "vqa_grounding"),
        "metrics": metrics,
        "steps": [
            step(
                "Run VQA model",
                TraceStepStatus.COMPLETE,
                f"{result.get('tokens', 0)} tokens, {len(evidence)} box(es)",
                duration_ms,
            )
        ],
    }


async def specialist_stub(state: GraphState) -> dict[str, Any]:
    """Placeholder for change_detection and optical_sar_fusion.

    Returns the registry's mock payload in the standard shape so the whole path
    — validation, tracing, rendering — is exercisable before those checkpoints
    exist. Replaced node-for-node in Phase 4.
    """
    started = time.perf_counter()
    task = state["task"]
    model = get_registry().get_model(task.value)
    payload = model.predict(query=state["query"], upload_id=state.get("upload_id"))

    return {
        "answer": payload.get("result", ""),
        "evidence": [],
        "confidence": float(payload.get("confidence", 0.0)),
        "model_used": f"{task.value} (mock)",
        "status": QueryStatus.UNAVAILABLE,
        "error": f"{task.value} has no checkpoint wired yet — this is a placeholder response.",
        "metrics": [],
        "steps": [
            step(
                f"Run {task.value}",
                TraceStepStatus.FAILED,
                "no specialist model registered; returned the registry mock",
                elapsed_ms(started),
            )
        ],
    }
