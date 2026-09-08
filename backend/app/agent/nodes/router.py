"""task_router — resolve the intent to a task and pull its model off the registry."""

from __future__ import annotations

import logging
from typing import Any

import time

from app.agent.state import GraphState, elapsed_ms, step
from app.core.model_registry import UnknownTaskError, get_registry
from app.models.schemas import INTENT_TO_TASK, QueryStatus, TaskType, TraceStepStatus

logger = logging.getLogger(__name__)

#: Which node runs each task. A task absent here falls through to
#: `specialist_stub`, which reports it as unwired rather than pretending.
TASK_NODES: dict[TaskType, str] = {
    TaskType.VQA_GROUNDING: "vqa_grounding_node",
    TaskType.CHANGE_DETECTION: "change_node",
    TaskType.OPTICAL_SAR_FUSION: "fusion_node",
}


def _config_summary(config: dict[str, Any]) -> str:
    """The one field of a task's config worth putting in the trace.

    Encoder first: fusion carries both an encoder and a VQA base_model, and the
    encoder is the half that distinguishes it.
    """
    for key in ("encoder", "base_model", "method"):
        if config.get(key):
            return str(config[key])
    return "no checkpoint configured"


async def task_router(state: GraphState) -> dict[str, Any]:
    started = time.perf_counter()
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return {}

    intent = state["intent"]
    task = INTENT_TO_TASK[intent]
    registry = get_registry()

    try:
        config = registry.config_for(task.value)
    except UnknownTaskError as exc:
        return {
            "task": task,
            "status": QueryStatus.FAILED,
            "error": f"No model is configured for {task.value}.",
            "steps": [step("Route task", TraceStepStatus.FAILED, str(exc), elapsed_ms(started))],
        }

    # Resolving the handle here means a bad registry entry surfaces as a routing
    # failure rather than an exception inside a specialist node.
    try:
        registry.get_model(task.value)
    except Exception as exc:  # noqa: BLE001 - registry errors are routing errors
        logger.warning("registry could not produce %s", task.value, exc_info=True)
        return {
            "task": task,
            "status": QueryStatus.FAILED,
            "error": f"Could not obtain a model for {task.value}: {exc}",
            "steps": [step("Route task", TraceStepStatus.FAILED, str(exc), elapsed_ms(started))],
        }

    return {
        "task": task,
        "model_config_used": config,
        "steps": [
            step(
                "Route task",
                TraceStepStatus.COMPLETE,
                f"{intent.value} -> {task.value} ({_config_summary(config)})",
                elapsed_ms(started),
            )
        ],
    }


def route_to_specialist(state: GraphState) -> str:
    """Conditional edge out of task_router.

    Anything that already failed skips inference. Every task in TASK_NODES has a
    real node; the stub stays as the landing place for one added to the enum
    before its node exists.
    """
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return "output_combiner"
    return TASK_NODES.get(state.get("task"), "specialist_stub")


def route_after_validation(state: GraphState) -> str:
    """Conditional edge out of input_validator — a rejection skips routing."""
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return "output_combiner"
    return "task_router"
