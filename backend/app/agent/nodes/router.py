"""task_router — resolve the intent to a task and pull its model off the registry."""

from __future__ import annotations

import logging
from typing import Any

import time

from app.agent.state import GraphState, elapsed_ms, step
from app.core.model_registry import UnknownTaskError, get_registry
from app.models.schemas import INTENT_TO_TASK, QueryStatus, TaskType, TraceStepStatus

logger = logging.getLogger(__name__)


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
                f"{intent.value} -> {task.value} ({config.get('base_model') or config.get('method') or config.get('encoder')})",
                elapsed_ms(started),
            )
        ],
    }


def route_to_specialist(state: GraphState) -> str:
    """Conditional edge out of task_router.

    Anything that already failed skips inference. Only vqa_grounding has a real
    node today; the rest land on the stub until their Phase 4 implementations
    arrive.
    """
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return "output_combiner"
    if state.get("task") is TaskType.VQA_GROUNDING:
        return "vqa_grounding_node"
    return "specialist_stub"


def route_after_validation(state: GraphState) -> str:
    """Conditional edge out of input_validator — a rejection skips routing."""
    status = state.get("status")
    if status is not None and status != QueryStatus.OK:
        return "output_combiner"
    return "task_router"
