
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from generation.state import AgentState, create_initial_state
from generation.nodes import (
    academic_search_node,
    citation_validation_node,
    generate_node,
    local_retrieve_node,
    normalize_papers_node,
    paper_content_node,
    query_understanding_node,
    relevance_check_node,
    router_node,
)
from generation.routing import route_after_relevance


# ---------------------------------------------------------------------------
# LangGraph routing functions
# ---------------------------------------------------------------------------


def route_from_state(
    state: AgentState,
) -> Literal[
    "general_answer",
    "local_rag",
    "academic_research",
]:
    """Return the route selected by the query-understanding/router nodes."""

    route = state.get("route", "general_answer")

    if route not in {
        "general_answer",
        "local_rag",
        "academic_research",
    }:
        return "general_answer"

    return route


# ---------------------------------------------------------------------------
# Workflow construction
# ---------------------------------------------------------------------------


def create_workflow():
    """Create and compile the PaperPilot LangGraph workflow."""

    builder = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    builder.add_node(
        "query_understanding",
        query_understanding_node,
    )

    builder.add_node(
        "router",
        router_node,
    )

    builder.add_node(
        "local_retrieve",
        local_retrieve_node,
    )

    builder.add_node(
        "relevance_check",
        relevance_check_node,
    )

    builder.add_node(
        "academic_search",
        academic_search_node,
    )

    builder.add_node(
        "normalize_papers",
        normalize_papers_node,
    )

    builder.add_node(
        "paper_content",
        paper_content_node,
    )

    builder.add_node(
        "generate",
        generate_node,
    )

    builder.add_node(
        "citation_validation",
        citation_validation_node,
    )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    builder.add_edge(
        START,
        "query_understanding",
    )

    # ------------------------------------------------------------------
    # Query understanding → router
    # ------------------------------------------------------------------

    builder.add_edge(
        "query_understanding",
        "router",
    )

    # ------------------------------------------------------------------
    # Primary route selection
    #
    # general_answer:
    #     router → generate
    #
    # local_rag:
    #     router → local_retrieve
    #
    # academic_research:
    #     router → academic_search
    # ------------------------------------------------------------------

    builder.add_conditional_edges(
        "router",
        route_from_state,
        {
            "general_answer": "generate",
            "local_rag": "local_retrieve",
            "academic_research": "academic_search",
        },
    )

    # ------------------------------------------------------------------
    # Local RAG path
    # ------------------------------------------------------------------

    builder.add_edge(
        "local_retrieve",
        "relevance_check",
    )

    builder.add_conditional_edges(
        "relevance_check",
        route_after_relevance,
        {
            "generate": "generate",
            "academic_search": "academic_search",
        },
    )

    # ------------------------------------------------------------------
    # Academic research path
    # ------------------------------------------------------------------

    builder.add_edge(
        "academic_search",
        "normalize_papers",
    )

    builder.add_edge(
        "normalize_papers",
        "paper_content",
    )

    builder.add_edge(
        "paper_content",
        "generate",
    )

    # ------------------------------------------------------------------
    # Final answer and citation validation
    # ------------------------------------------------------------------

    builder.add_edge(
        "generate",
        "citation_validation",
    )

    builder.add_edge(
        "citation_validation",
        END,
    )

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    return builder.compile()


# ---------------------------------------------------------------------------
# Compiled production workflow
# ---------------------------------------------------------------------------

workflow = create_workflow()


# ---------------------------------------------------------------------------
# Standard non-streaming invocation
# ---------------------------------------------------------------------------


async def run_workflow(
    query: str,
    conversation_id: str = "",
    history: list[dict] | None = None,
    source_filter: str | None = None,
) -> AgentState:
    """Run the PaperPilot workflow for a single chat request.

    The conversation ID is supplied both as LangGraph's configurable
    thread identifier and as explicit LangSmith metadata.

    ``thread_id`` here identifies the conversation/thread. It does not
    itself provide persistent checkpoint storage because the current
    workflow is compiled without a persistent LangGraph checkpointer.
    """

    config = {
        "run_name": "PaperPilot Chat",
        "tags": [
            "paperpilot",
            "chat",
            "invoke",
        ],
        "metadata": {
            "thread_id": conversation_id,
            "conversation_id": conversation_id,
            "endpoint": "chat",
        },
        "configurable": {
            "thread_id": conversation_id,
        },
    }

    initial_state = create_initial_state(
        query=query,
        source_filter=source_filter,
        history=history,
        conversation_id=conversation_id,
    )

    return await workflow.ainvoke(
        initial_state,
        config=config,
    )