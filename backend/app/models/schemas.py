"""Pydantic schemas mirroring the MongoDB document shapes."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskType(str, Enum):
    """Task names — kept in lockstep with the keys in model_config.yaml."""

    VQA_GROUNDING = "vqa_grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class Intent(str, Enum):
    """What the user is asking for, as decided by the intent classifier.

    Finer-grained than TaskType: two intents can share one model. The mapping
    lives in INTENT_TO_TASK.
    """

    SINGLE_IMAGE_VQA = "single_image_vqa"
    SINGLE_IMAGE_CAPTIONING = "single_image_captioning"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class UploadMode(str, Enum):
    SINGLE = "single"
    CROSS_MODAL = "cross_modal"      # optical + SAR
    BI_TEMPORAL = "bi_temporal"      # t0 + t1


#: Which model a given intent needs. Both single-image intents — a pointed
#: question and a whole-scene caption — run on the same VQA checkpoint; they
#: differ in the prompt, not the model.
INTENT_TO_TASK: dict[Intent, TaskType] = {
    Intent.SINGLE_IMAGE_VQA: TaskType.VQA_GROUNDING,
    Intent.SINGLE_IMAGE_CAPTIONING: TaskType.VQA_GROUNDING,
    Intent.CHANGE_VQA: TaskType.CHANGE_DETECTION,
    Intent.OPTICAL_SAR_FUSION: TaskType.OPTICAL_SAR_FUSION,
}

#: The upload modes each intent can work from.
INTENT_TO_MODES: dict[Intent, tuple["UploadMode", ...]] = {
    Intent.SINGLE_IMAGE_VQA: (UploadMode.SINGLE,),
    Intent.SINGLE_IMAGE_CAPTIONING: (UploadMode.SINGLE,),
    Intent.CHANGE_VQA: (UploadMode.BI_TEMPORAL,),
    Intent.OPTICAL_SAR_FUSION: (UploadMode.CROSS_MODAL,),
}


#: Which intents a given upload mode can serve — INTENT_TO_MODES, inverted.
#: A mode with exactly one intent determines the task on its own, which is what
#: lets the classifier correct an obviously wrong label. Derived rather than
#: written out so the two cannot fall out of step.
MODE_TO_INTENTS: dict[UploadMode, tuple[Intent, ...]] = {
    mode: tuple(i for i, modes in INTENT_TO_MODES.items() if mode in modes)
    for mode in UploadMode
}


class QueryStatus(str, Enum):
    """How a graph run ended.

    The graph completing is not the same as the question being answered: a
    rejected input or an unloadable checkpoint is a successful run with nothing
    to say, and the trace shows exactly which node stopped it.
    """

    OK = "ok"
    REJECTED = "rejected"        # input_validator refused the assets
    UNAVAILABLE = "unavailable"  # the specialist model could not be loaded
    FAILED = "failed"            # a node raised


# --------------------------------------------------------------------------
# query_history
# --------------------------------------------------------------------------
class QueryHistoryBase(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Raw user query text")
    task_type: TaskType | None = Field(None, description="Task the router selected, if known")
    intent: Intent | None = Field(None, description="Intent the classifier settled on")


class QueryHistoryCreate(QueryHistoryBase):
    asset_ids: list[str] = Field(default_factory=list)


class QueryHistoryDoc(QueryHistoryBase):
    """Document stored in the ``query_history`` collection."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="_id")
    asset_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)


def _check_degraded_confidence(model: Any) -> Any:
    """`confidence` is None exactly when `degraded` is true.

    A CPU-fallback answer has no measured confidence; a real-model answer (or a
    run with no answer, reported as 0.0) always has a number.
    """
    if model.degraded != (model.confidence is None):
        raise ValueError(
            f"confidence={model.confidence!r} with degraded={model.degraded}: "
            "confidence must be None exactly when degraded is true"
        )
    return model


# --------------------------------------------------------------------------
# execution_traces
# --------------------------------------------------------------------------
class TraceStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class TraceStep(BaseModel):
    """One agent step, streamed to the execution-trace panel."""

    label: str
    status: TraceStepStatus = TraceStepStatus.PENDING
    detail: str | None = None
    duration_ms: int | None = Field(None, ge=0)
    timestamp: datetime = Field(default_factory=_utcnow)


class ExecutionTraceCreate(BaseModel):
    query_id: str | None = None
    task_selected: TaskType
    intent: Intent | None = None
    status: QueryStatus = QueryStatus.OK
    error: str | None = None
    models_used: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: None only for a degraded run — see `degraded`.
    confidence: float | None = Field(0.0, ge=0.0, le=1.0)
    #: The answer came from the opt-in CPU fallback, not the configured model.
    degraded: bool = False
    degraded_reason: str | None = None
    steps: list[TraceStep] = Field(default_factory=list)

    _degraded_confidence = model_validator(mode="after")(_check_degraded_confidence)


class ExecutionTraceDoc(ExecutionTraceCreate):
    """Document stored in the ``execution_traces`` collection."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="_id")
    timestamp: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------
# uploads
# --------------------------------------------------------------------------
class UploadedAsset(BaseModel):
    asset_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int
    role: Literal["primary", "optical", "sar", "t0", "t1"] = "primary"
    stored_path: str


class UploadResponse(BaseModel):
    upload_id: str
    mode: UploadMode
    benchmark_mode: bool = False
    assets: list[UploadedAsset]
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------
# query request / response
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    upload_id: str | None = None
    intent: Intent | None = Field(
        None, description="Force an intent; omitted means the classifier decides"
    )
    task_type: TaskType | None = Field(
        None, description="Force a task; omitted means the router decides"
    )
    parameters: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query_id: str
    trace_id: str
    status: QueryStatus = QueryStatus.OK
    intent: Intent | None = None
    task_selected: TaskType
    answer: str
    #: None only for a degraded run — see `degraded`.
    confidence: float | None = Field(0.0, ge=0.0, le=1.0)
    #: The answer came from the opt-in CPU fallback (model_config.yaml
    #: `vqa_grounding.cpu_fallback`), not the configured model.
    degraded: bool = False
    #: Why the configured model did not run, when `degraded`.
    degraded_reason: str | None = None
    #: Geometry the canvas draws, normalised to 0-1 of the scene extent.
    overlays: list[dict[str, Any]] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    metrics: list[dict[str, str]] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    #: Set when status is not "ok" — why the run produced no answer.
    error: str | None = None

    _degraded_confidence = model_validator(mode="after")(_check_degraded_confidence)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    mongo: bool
    tasks: list[str]
