"""Query endpoints.

Both run the same controller graph. `/query` waits for it and returns one JSON
object; `/query/stream` emits a server-sent event as each node lands, so the UI
can show progress while a checkpoint loads or a decode runs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.agent.graph import get_graph, initial_state, node_sequence
from app.core.auth import AuthUser, current_user
from app.db import mongo
from app.models.schemas import (
    INTENT_TO_TASK,
    ExecutionTraceDoc,
    QueryHistoryDoc,
    QueryRequest,
    QueryResponse,
    QueryStatus,
    TaskType,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def _to_response(state: dict[str, Any]) -> QueryResponse:
    evidence = state.get("evidence", [])
    intent = state.get("intent")
    # A run rejected before task_router still has a task the intent implies;
    # reporting the default would misattribute the rejection.
    task = state.get("task") or (INTENT_TO_TASK.get(intent) if intent else None)
    return QueryResponse(
        query_id=state["query_id"],
        trace_id=state.get("trace_id", ""),
        status=state.get("status", QueryStatus.OK),
        intent=intent,
        task_selected=task or TaskType.VQA_GROUNDING,
        answer=state.get("answer", ""),
        confidence=state.get("confidence", 0.0),
        degraded=bool(state.get("degraded", False)),
        degraded_reason=state.get("degraded_reason"),
        overlays=[item for item in evidence if item.get("kind") in {"box", "mask"}],
        models_used=[m for m in (state.get("model_used"),) if m],
        metrics=state.get("metrics", []),
        steps=state.get("steps", []),
        error=state.get("error"),
    )


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    payload: QueryRequest, user: AuthUser = Depends(current_user)
) -> QueryResponse:
    """Run the graph to completion and return the result."""
    state = initial_state(
        query=payload.query,
        upload_id=payload.upload_id,
        forced_intent=payload.intent,
        parameters=payload.parameters,
        uid=user.uid,
    )
    final = await get_graph().ainvoke(state)
    return _to_response(final)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query/stream")
async def stream_query(
    payload: QueryRequest, user: AuthUser = Depends(current_user)
) -> StreamingResponse:
    """Run the graph, emitting each node's trace steps as they complete.

    Events: `plan` once up front, `step` per trace entry, then `result` or
    `error`. Steps arrive in the order the graph produced them, so a client can
    append blindly without reconciling.
    """
    state = initial_state(
        query=payload.query,
        upload_id=payload.upload_id,
        forced_intent=payload.intent,
        parameters=payload.parameters,
        uid=user.uid,
    )

    async def events() -> AsyncIterator[str]:
        # With a forced intent the path is already known; otherwise this is the
        # single-image default, and the steps correct it as they land.
        planned = node_sequence(INTENT_TO_TASK.get(payload.intent) if payload.intent else None)
        yield _sse("plan", {"query_id": state["query_id"], "nodes": list(planned)})

        sent = 0
        merged: dict[str, Any] = dict(state)
        try:
            async for update in get_graph().astream(state, stream_mode="values"):
                merged = update
                steps = update.get("steps", [])
                for entry in steps[sent:]:
                    yield _sse("step", entry)
                sent = len(steps)

            yield _sse("result", _to_response(merged).model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 - the stream reports its own failure
            logger.exception("graph run failed")
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/query/history", response_model=list[QueryHistoryDoc])
async def query_history(
    limit: int = Query(50, ge=1, le=200), user: AuthUser = Depends(current_user)
) -> list[QueryHistoryDoc]:
    """The caller's own past queries, newest first. Records from before
    sign-in existed carry no uid and match nobody."""
    try:
        cursor = (
            mongo.get_db()[mongo.QUERY_HISTORY]
            .find({"uid": user.uid})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return [QueryHistoryDoc(**doc) async for doc in cursor]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"query history unavailable: {exc}"
        ) from exc


@router.get("/query/{query_id}/trace", response_model=ExecutionTraceDoc)
async def query_trace(query_id: str, user: AuthUser = Depends(current_user)) -> ExecutionTraceDoc:
    try:
        # Another user's trace is reported exactly like a missing one.
        doc = await mongo.get_db()[mongo.EXECUTION_TRACES].find_one(
            {"query_id": query_id, "uid": user.uid}
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"trace store unavailable: {exc}"
        ) from exc
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no trace for query {query_id}")
    return ExecutionTraceDoc(**doc)
