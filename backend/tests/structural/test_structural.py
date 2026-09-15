"""Deterministic structural tests for the active production LangGraph."""

from unittest.mock import AsyncMock, patch

import pytest

from generation.nodes import academic_search_node, normalize_papers_node
from generation.workflow import workflow
from retrieval.academic.base import AcademicPaper


def test_production_graph_contains_all_research_stages():
    graph = workflow.get_graph()
    assert {
        "query_understanding", "router", "local_retrieve", "relevance_check",
        "academic_search", "normalize_papers", "paper_content", "generate",
        "citation_validation",
    }.issubset(graph.nodes)


@pytest.mark.asyncio
async def test_academic_provider_failure_preserves_other_provider_results():
    paper = AcademicPaper(
        title="Recent RAG Evaluation", authors=["Author"], abstract="RAG evaluation benchmark",
        publication_date="2025-01-01", year=2025, provider="openalex",
        paper_url="https://openalex.org/W123",
    )
    with patch(
        "retrieval.academic.arxiv_provider.ArxivProvider.search",
        new=AsyncMock(side_effect=RuntimeError("arXiv unavailable")),
    ) as arxiv_search, patch(
        "retrieval.academic.openalex_provider.OpenAlexProvider.search",
        new=AsyncMock(return_value=[paper]),
    ) as openalex_search:
        state = {"query": "Find recent papers about RAG evaluation", "user_query": "Find recent papers about RAG evaluation"}
        searched = await academic_search_node(state)
        normalized = await normalize_papers_node({**state, **searched})

    assert arxiv_search.await_count == openalex_search.await_count == 1
    assert searched["provider_result_counts"] == {"arxiv": 0, "openalex": 1}
    assert normalized["final_docs"][0]["title"] == "Recent RAG Evaluation"
    assert normalized["final_docs"][0]["paper_url"] == "https://openalex.org/W123"
