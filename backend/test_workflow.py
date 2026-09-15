"""Unit tests for the active async LangGraph nodes and state contract."""

import pytest

from generation.nodes import citation_validation_node, query_understanding_node, relevance_check_node, router_node
from generation.routing import is_casual_query, normalize_query
from generation.state import create_initial_state


class TestRouting:
    def test_normalize_query(self):
        assert normalize_query("Hello!") == "hello"
        assert normalize_query("How's it going?") == "hows it going"

    def test_casual_detection(self):
        assert is_casual_query("Hi there!")
        assert not is_casual_query("Find recent papers about RAG")

    @pytest.mark.asyncio
    async def test_academic_intent_and_route(self):
        state = create_initial_state("Find recent papers about RAG evaluation")
        understood = await query_understanding_node(state)
        routed = await router_node({**state, **understood})
        assert understood["intent"] == "academic_research"
        assert routed["route"] == "academic_research"

    @pytest.mark.asyncio
    async def test_local_library_intent_and_route(self):
        state = create_initial_state("What do my uploaded PDFs say about RAG?")
        understood = await query_understanding_node(state)
        routed = await router_node({**state, **understood})
        assert understood["intent"] == "local_rag"
        assert routed["route"] == "local_rag"


class TestRelevanceCheck:
    @pytest.mark.asyncio
    async def test_relevance_uses_final_docs(self):
        state = create_initial_state("test query")
        state["final_docs"] = [{"text": "relevant evidence"}]
        assert (await relevance_check_node(state))["has_sufficient_evidence"] is True

    @pytest.mark.asyncio
    async def test_relevance_rejects_empty_docs(self):
        assert (await relevance_check_node(create_initial_state("test query")))["has_sufficient_evidence"] is False


class TestState:
    def test_initial_state_matches_active_graph_contract(self):
        history = [{"role": "user", "content": "previous"}]
        state = create_initial_state("test query", "paper.pdf", history, "conversation-1")
        assert state["query"] == state["user_query"] == "test query"
        assert state["source_filter"] == "paper.pdf"
        assert state["messages"] == history
        assert state["conversation_id"] == "conversation-1"
        assert state["final_docs"] == []
        assert state["response"] == ""


class TestCitationValidation:
    @pytest.mark.asyncio
    async def test_academic_citations_validate_against_final_docs(self):
        state = {
            "response": "Recommended reading [Paper 1].",
            "final_docs": [{"kind": "academic", "title": "A real paper"}],
            "citations": [],
        }
        assert (await citation_validation_node(state))["citations"] == [{
            "source": "Paper 1", "valid": True, "details": "Academic paper exists in context",
        }]

    @pytest.mark.asyncio
    async def test_out_of_range_citation_is_invalid(self):
        state = {
            "response": "Unsupported [Source 5].",
            "final_docs": [{"kind": "library", "text": "one"}],
            "citations": [],
        }
        assert (await citation_validation_node(state))["citations"][0]["valid"] is False
