"""The active LangGraph orchestration for PaperPilot."""

from __future__ import annotations

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


def create_workflow():
    """Compile the production graph, including every supported research path."""
    builder = StateGraph(AgentState)
    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("router", router_node)
    builder.add_node("local_retrieve", local_retrieve_node)
    builder.add_node("relevance_check", relevance_check_node)
    builder.add_node("academic_search", academic_search_node)
    builder.add_node("normalize_papers", normalize_papers_node)
    builder.add_node("paper_content", paper_content_node)
    builder.add_node("generate", generate_node)
    builder.add_node("citation_validation", citation_validation_node)

    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "router")
    builder.add_conditional_edges(
        "router",
        lambda state: state.get("route", "general_answer"),
        {
            "general_answer": "generate",
            "local_rag": "local_retrieve",
            "academic_research": "academic_search",
        },
    )
    builder.add_edge("local_retrieve", "relevance_check")
    builder.add_conditional_edges(
        "relevance_check",
        route_after_relevance,
        {"generate": "generate", "academic_search": "academic_search"},
    )
    builder.add_edge("academic_search", "normalize_papers")
    builder.add_edge("normalize_papers", "paper_content")
    builder.add_edge("paper_content", "generate")
    builder.add_edge("generate", "citation_validation")
    builder.add_edge("citation_validation", END)
    return builder.compile()


workflow = create_workflow()


async def run_workflow(
    query: str,
    conversation_id: str = "",
    history: list[dict] | None = None,
    source_filter: str | None = None,
) -> AgentState:
    return await workflow.ainvoke(
        create_initial_state(query, source_filter, history, conversation_id)
    )
