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
    assert normalized["final_docs"][0]["url"] == "https://openalex.org/W123"
    assert normalized["final_docs"][0]["source"] == "Recent RAG Evaluation"
    assert "RAG evaluation benchmark" in normalized["final_docs"][0]["text"]


def test_source_card_prefers_pdf_url_and_abstract_text():
    from generation.sources import format_source_for_ui

    card = format_source_for_ui(
        {
            "kind": "academic",
            "title": "Attention Is All You Need",
            "abstract": "We propose the Transformer.",
            "pdf_url": "https://arxiv.org/pdf/1706.03762",
            "paper_url": "https://arxiv.org/abs/1706.03762",
        }
    )
    assert card["source"] == "Attention Is All You Need"
    assert card["url"] == "https://arxiv.org/pdf/1706.03762"
    assert card["text"] == "We propose the Transformer."


def test_library_source_card_keeps_filename_without_url():
    from generation.sources import format_source_for_ui

    card = format_source_for_ui(
        {
            "kind": "library",
            "source": "rag.pdf",
            "text": "Hybrid retrieval uses BM25.",
            "page": 2,
        }
    )
    assert card["source"] == "rag.pdf"
    assert card["title"] == "rag.pdf"
    assert "url" not in card
    assert card["page"] == 2


def test_generation_history_is_clipped_under_budget():
    from generation.generator import (
        MAX_MESSAGE_CHARS,
        MAX_PROMPT_CHARS,
        _convert_history_to_messages,
        _fit_messages,
        _prompt_chars,
        is_payload_too_large,
    )
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "A" * 5000},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "B" * 5000},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "C" * 5000},
    ]
    converted = _convert_history_to_messages(history)
    assert len(converted) == 4
    assert all(len(message.content) <= MAX_MESSAGE_CHARS for message in converted)

    fitted = _fit_messages(
        [
            SystemMessage(content="sys"),
            AIMessage(content="X" * MAX_PROMPT_CHARS),
            HumanMessage(content="follow up"),
        ]
    )
    assert _prompt_chars(fitted) <= MAX_PROMPT_CHARS
    assert is_payload_too_large(RuntimeError("Error code: 413 - Request too large"))

