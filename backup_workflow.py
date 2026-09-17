"""LangGraph workflow for the Phase 2 PaperPilot workflow.

Extended from Phase 1: when local RAG has insufficient evidence,
the graph routes to an academic research-paper search workflow
(arXiv + OpenAlex), with normalization, ranking, and optional PDF content.

Full flow:
  START
    │
    ▼
  query_understanding
    │
    ▼
  router
    │           ┌──────────────────────┐
    │           │  casual conversation│
    │           └───────→ chat ──────┘
    │           │              │
    │           │              ▼
    │           │  has local evidence?  │  (relevance_check)
    │           │              │  │
    │           │    sufficient ──────┘
    │           │              │
    │           │              ▼
    │           │   generate ──────────────┐
    │           │              │          │
    │           │              ▼          │
    │           │ citation_validation    │
    │           │              │          │
    │           │              ▼          │
    │           └─────────────────────────────────────┘
    │
    ▼
  local_rag
    │
    ▼
  local_retrieve
    │
    ▼
  relevance_check
    │           ┌──────────────────────┐
    │           │  sufficient evidence│
    │           └───────→ generate ────┘
    │           │              │
    │           │              ▼
    │           │  insufficient ──────┐
    │           │              │     │
    │           │              ▼     │
    │           │ academic_search ────┘
    │           │
    ▼
  academic_search
    │
    ▼
  normalize_papers
    │
    ▼
  paper_content   (optional PDF fetch for top-ranked papers)
    │
    ▼
  generate
    │
    ▼
 citation_validation
    │
    ▼
       END
"""

from generation.state import AgentState

def create_workflow():
    """Create and compile the LangGraph workflow for Phase 2."""
    from generation.nodes import (
        query_understanding_node,
        router_node,
        local_retrieve_node,
        relevance_check_node,
        academic_search_node,
        normalize_papers_node,
        paper_content_node,
        generate_node,
        citation_validation_node,
        error_handler_node,
    )

    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("router", router_node)
    builder.add_node("local_retrieve", local_retrieve_node)
    builder.add_node("relevance_check", relevance_check_node)
    builder.add_node("academic_search", academic_search_node)
    builder.add_node("normalize_papers", normalize_papers_node)
    builder.add_node("paper_content", paper_content_node)
    builder.add_node("generate", generate_node)
    builder.add_node("citation_validation", citation_validation_node)
    builder.add_node("error_handler", error_handler_node)

    # Entry point: query understanding then router
    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "router")

    # Router: decides chat vs local_rag vs academic_research
    builder.add_conditional_edges(
        "router",
        route_query,
        {
            "chat": "chat",
            "local_rag": "local_retrieve",
            "academic_research": "academic_search",
        },
    )

    # Chat path
    builder.add_edge("chat", "citation_validation")
    builder.add_edge("citation_validation", END)

    # Local RAG path
    builder.add_edge("local_retrieve", "relevance_check")

    # Relevance check: sufficient -> generate; insufficient -> academic_search
    builder.add_conditional_edges(
        "relevance_check",
        route_after_relevance,
        {
            "generate": "generate",
            "academic_search": "academic_search",
        },
    )

    # Academic search path
    builder.add_conditional_edges(
        "academic_search",
        route_after_academic_search,
        {"normalize_papers": "normalize_papers"},
    )

    builder.add_conditional_edges(
        "normalize_papers",
        route_after_ranking,
        {"paper_content": "paper_content"},
    )

    builder.add_conditional_edges(
        "paper_content",
        route_after_paper_content,
        {"generate": "generate"},
    )

    # Generation & citations
    builder.add_edge("generate", "citation_validation")
    builder.add_edge("citation_validation", END)

    # Use MemorySaver for short-term conversational memory
    memory = MemorySaver()

    return builder.compile(checkpointer=memory)


# Compile the workflow once at module import
workflow = create_workflow()


async def run_workflow(
    query: str,
    conversation_id: str,
    history: list | None = None,
) -> dict:
    """Run the Phase 2 LangGraph workflow with the given inputs.

    Returns the final state dict from the workflow.
    """
    from generation.state import create_initial_state

    initial_state = create_initial_state(
        query=query,
        source_filter=None,
        history=history,
    )

    result = await workflow.ainvoke(initial_state)
    return result


if __name__ == "__main__":
    # For testing
    import asyncio

    async def test():
        print("=== Testing Phase 2 workflow ===")
        result = await run_workflow("What is RAG?", "conv-001")
        print(f"Route: {result.get('route')}")
        print(f"Answer: {str(result.get('answer'))[:200]}...")
        print(f"Source type: {result.get('source_type')}")

        # Test research query with local knowledge
        print("\n=== Testing research query (local) ===")
        result = await run_workflow("What is RAG?")
        print(f"Route: {result.get('route')}")
        print(f"Retrieved docs: {len(result.get('reranked_docs', []))}")
        print(f"Relevant docs: {len(result.get('relevant_docs', []))}")
        print(f"Sufficient evidence: {result.get('has_sufficient_evidence')}")
        print(f"Answer: {str(result.get('answer'))[:200]}...")
        print(f"Citations: {result.get('citations')}")

        # Test research query triggering academic fallback
        print("\n=== Testing academic fallback ===")
        result = await run_workflow("Suggest recent papers about agentic AI")
        print(f"Route: {result.get('route')}")
        print(f"Academic papers found: {len(result.get('academic_papers', []))}")
        print(f"Ranked papers: {len(result.get('ranked_papers', []))}")
        print(f"Used academic: {result.get('academic_source_type')}")
        print(f"Answer: {str(result.get('answer'))[:200]}...")
        print(f"Citations: {result.get('citations')}")

    asyncio.run(test())


# When imported as a module, the `workflow` variable is the compiled graph.
# The `run_workflow` async function can be called from FastAPI endpoints or tests.
from generation.routing import route_query, route_after_relevance, route_after_academic_search, route_after_ranking, route_after_paper_content
