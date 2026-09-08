"""trace_logger — persist the run to MongoDB.

Best effort by design. Losing the audit record must not lose the answer, so a
write failure is logged and marked in the trace rather than raised.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import time

from app.agent.state import GraphState, elapsed_ms, step
from app.db import mongo
from app.models.schemas import (
    INTENT_TO_TASK,
    ExecutionTraceDoc,
    QueryHistoryDoc,
    QueryStatus,
    TaskType,
    TraceStepStatus,
)

logger = logging.getLogger(__name__)


async def trace_logger(state: GraphState) -> dict[str, Any]:
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    intent = state.get("intent")
    task = (
        state.get("task")
        or (INTENT_TO_TASK.get(intent) if intent else None)
        or TaskType.VQA_GROUNDING
    )
    status = state.get("status", QueryStatus.OK)

    def failed(detail: str) -> dict[str, Any]:
        return {
            "trace_id": trace_id,
            "steps": [step("Log trace", TraceStepStatus.FAILED, detail, elapsed_ms(started))],
        }

    if not mongo.available():
        return failed("MongoDB unreachable — this run was not persisted")

    history = QueryHistoryDoc(
        _id=state["query_id"],
        query=state["query"],
        task_type=task,
        intent=intent,
        asset_ids=[a.get("asset_id", "") for a in state.get("assets", [])],
        timestamp=now,
    )
    trace = ExecutionTraceDoc(
        _id=trace_id,
        query_id=state["query_id"],
        task_selected=task,
        intent=intent,
        status=status,
        error=state.get("error"),
        models_used=[m for m in (state.get("model_used"),) if m],
        parameters={
            **state.get("model_config_used", {}),
            **state.get("parameters", {}),
            "intent_source": state.get("intent_source"),
            "intent_confidence": state.get("intent_confidence"),
        },
        confidence=state.get("confidence", 0.0),
        steps=list(state.get("steps", [])),
        timestamp=now,
    )

    try:
        db = mongo.get_db()
        await db[mongo.QUERY_HISTORY].insert_one(history.model_dump(mode="json", by_alias=True))
        await db[mongo.EXECUTION_TRACES].insert_one(trace.model_dump(mode="json", by_alias=True))

        # The node cannot time its own write before making it, so its step is
        # appended afterwards. The stored trace then holds every step the
        # caller sees, with a real duration on this one.
        logged_step = step(
            "Log trace",
            TraceStepStatus.COMPLETE,
            f"execution_traces/{trace_id}",
            elapsed_ms(started),
        )
        await db[mongo.EXECUTION_TRACES].update_one(
            {"_id": trace_id}, {"$push": {"steps": logged_step}}
        )
    except Exception as exc:  # noqa: BLE001 - the answer outlives the audit record
        mongo.mark_unavailable()
        logger.warning("could not persist trace %s: %s", trace_id, exc)
        return failed("MongoDB write failed — this run was not persisted")

    return {"trace_id": trace_id, "steps": [logged_step]}
