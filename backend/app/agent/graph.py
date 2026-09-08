"""The SatQuery controller graph.

    intake -> intent_classifier -> input_validator -> task_router
           -> vqa_grounding_node | specialist_stub
           -> output_combiner -> trace_logger

Two conditional edges short-circuit inference: a rejected input skips straight
to output_combiner, and task_router picks the specialist for the chosen task.
output_combiner and trace_logger always run, so every request — answered,
rejected or failed — leaves a complete trace behind.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.combiner import output_combiner
from app.agent.nodes.intake import intake
from app.agent.nodes.intent import intent_classifier
from app.agent.nodes.router import route_after_validation, route_to_specialist, task_router
from app.agent.nodes.tracelog import trace_logger
from app.agent.nodes.validator import input_validator
from app.agent.nodes.vqa import specialist_stub, vqa_grounding_node
from app.agent.state import GraphState
from app.models.schemas import Intent, QueryStatus

#: Node names, in the order they appear in a nominal run. Exposed so the API can
#: tell a client what is coming before anything has run.
NODE_SEQUENCE = (
    "intake",
    "intent_classifier",
    "input_validator",
    "task_router",
    "vqa_grounding_node",
    "output_combiner",
    "trace_logger",
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("intake", intake)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("input_validator", input_validator)
    graph.add_node("task_router", task_router)
    graph.add_node("vqa_grounding_node", vqa_grounding_node)
    graph.add_node("specialist_stub", specialist_stub)
    graph.add_node("output_combiner", output_combiner)
    graph.add_node("trace_logger", trace_logger)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "intent_classifier")
    graph.add_edge("intent_classifier", "input_validator")

    graph.add_conditional_edges(
        "input_validator",
        route_after_validation,
        {"task_router": "task_router", "output_combiner": "output_combiner"},
    )
    graph.add_conditional_edges(
        "task_router",
        route_to_specialist,
        {
            "vqa_grounding_node": "vqa_grounding_node",
            "specialist_stub": "specialist_stub",
            "output_combiner": "output_combiner",
        },
    )

    graph.add_edge("vqa_grounding_node", "output_combiner")
    graph.add_edge("specialist_stub", "output_combiner")
    graph.add_edge("output_combiner", "trace_logger")
    graph.add_edge("trace_logger", END)

    return graph.compile()


@lru_cache
def get_graph():
    """One compiled graph per process."""
    return build_graph()


def initial_state(
    query: str,
    upload_id: str | None = None,
    forced_intent: Intent | None = None,
    parameters: dict[str, Any] | None = None,
) -> GraphState:
    return {
        "query": query,
        "query_id": uuid.uuid4().hex,
        "upload_id": upload_id,
        "forced_intent": forced_intent,
        "parameters": parameters or {},
        "status": QueryStatus.OK,
        "error": None,
        "steps": [],
    }
