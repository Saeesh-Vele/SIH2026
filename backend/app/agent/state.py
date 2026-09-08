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
    confidence: float
    model_used: str
    metrics: list[dict[str, str]]

    # -- outcome ------------------------------------------------------------
    status: QueryStatus
    error: str | None
    trace_id: str

    # -- trace --------------------------------------------------------------
    steps: Annotated[list[dict[str, Any]], operator.add]


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
