"""Shared state for the controller graph.

Every node takes the whole state and returns only the keys it changed;
LangGraph merges the result. `steps` is the exception — it is annotated as an
accumulator so each node appends its own trace entry without having to know
what ran before it.
"""

from __future__ import annotations

import operator
import time
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from app.models.schemas import (
    Intent,
    QueryStatus,
    TaskType,
    TraceStep,
    TraceStepStatus,
    UploadMode,
)


class GraphState(TypedDict, total=False):
    # -- request ------------------------------------------------------------
    query: str
    upload_id: str | None
    forced_intent: Intent | None
    parameters: dict[str, Any]
    query_id: str

    # -- intake -------------------------------------------------------------
    assets: list[dict[str, Any]]
    upload_mode: UploadMode | None
    benchmark_mode: bool

    # -- intent_classifier --------------------------------------------------
    intent: Intent | None
    intent_confidence: float
    intent_source: str
    intent_reason: str
    intent_model: str | None

    # -- task_router --------------------------------------------------------
    task: TaskType | None
    model_config_used: dict[str, Any]

    # -- specialist nodes ---------------------------------------------------
    answer: str
    evidence: list[dict[str, Any]]
    #: None exactly when `degraded` — a CPU-fallback answer has no measured
    #: confidence. See `honesty_fields`.
    confidence: float | None
    model_used: str
    #: True when the answer came from the opt-in CPU fallback, not the
    #: configured checkpoint. `degraded_reason` says why the checkpoint did not run.
    degraded: bool
    degraded_reason: str | None
    metrics: list[dict[str, str]]

    # -- outcome ------------------------------------------------------------
    status: QueryStatus
    error: str | None
    trace_id: str

    # -- trace --------------------------------------------------------------
    steps: Annotated[list[dict[str, Any]], operator.add]


def asset_by_role(state: "GraphState", *roles: str) -> dict[str, Any] | None:
    """The first bound asset carrying any of `roles`, in the order given.

    The validator has already checked the roles a task needs, so a None here
    means something changed underneath the graph rather than a bad request.
    """
    for role in roles:
        for asset in state.get("assets", []):
            if asset.get("role") == role:
                return asset
    return None


def elapsed_ms(started: float) -> int:
    """Milliseconds since a `time.perf_counter()` mark."""
    return int((time.perf_counter() - started) * 1000)


def step(
    label: str,
    status: TraceStepStatus = TraceStepStatus.COMPLETE,
    detail: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """One trace entry, serialised the way the API and the UI expect it."""
    return TraceStep(
        label=label,
        status=status,
        detail=detail,
        duration_ms=duration_ms,
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")


def honesty_fields(result: dict[str, Any]) -> dict[str, Any]:
    """`confidence`, `degraded` and `degraded_reason` from an engine result.

    The two always travel together: a degraded (CPU-fallback) result carries
    confidence None, and a real-model result carries a float. Anything else is
    an engine bug, raised here so the node reports a failed run rather than
    passing a made-up or missing number downstream.
    """
    degraded = bool(result.get("degraded", False))
    confidence = result.get("confidence")
    if degraded:
        if confidence is not None:
            raise ValueError(f"degraded result carries confidence {confidence!r}; expected None")
        return {
            "confidence": None,
            "degraded": True,
            "degraded_reason": result.get("degraded_reason") or "cpu_fallback in use",
        }
    if confidence is None:
        raise ValueError("non-degraded result has no confidence; real-model paths must measure one")
    return {"confidence": float(confidence), "degraded": False, "degraded_reason": None}


def degraded_detail(detail: str, fields: dict[str, Any]) -> str:
    """Prefix a trace step detail so a fallback run cannot be misread."""
    if not fields["degraded"]:
        return detail
    return f"DEGRADED — CPU fallback ({fields['degraded_reason']}): {detail}"
