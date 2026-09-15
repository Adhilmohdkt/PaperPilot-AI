"""Routing logic for the PaperPilot LangGraph workflow."""
import re
from typing import Literal

from .state import AgentState


CASUAL_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "how are you",
    "who are you",
    "what is your name",
    "what can you do",
    "help",
    "hi there",
}


def normalize_query(query: str) -> str:
    """Normalize query for casual matching."""
    return re.sub(r"[^a-z0-9\s]", "", query.lower()).strip()


def is_casual_query(query: str) -> bool:
    """Check if the query is casual conversation."""
    normalized = normalize_query(query)
    return normalized in CASUAL_PATTERNS


def route_query(state: AgentState) -> Literal["chat", "local_rag", "academic_search"]:
    """Route the query to chat, local RAG, or academic search.

    Decides based on whether the query appears to be a research question
    and whether local evidence exists (provided by caller).
    """
    query = state.get("query", "")
    is_casual = is_casual_query(query)

    # If casual, route to chat
    if is_casual:
        return "chat"

    # If we already have relevant local docs and they're sufficient, stay in local RAG
    # (This is handled by the relevance_check_node before we reach the router,
    #  but we keep a fallback here.)
    if state.get("has_sufficient_evidence", False):
        return "local_rag"

    # Otherwise route to academic search
    return "academic_search"


def route_after_relevance(state: AgentState) -> Literal["generate", "academic_search"]:
    """Route after the relevance check node.

    If sufficient local evidence -> generate
    Otherwise -> academic_search
    """
    if state.get("has_sufficient_evidence", False):
        return "generate"
    return "academic_search"


def route_after_academic_search(state: AgentState) -> Literal["normalize_papers"]:
    """After the academic search completes, always go to normalize papers."""
    return "normalize_papers"


def route_after_ranking(state: AgentState) -> Literal["paper_content"]:
    """After ranking/filtering, go to optional PDF content retrieval."""
    # If we have ranked papers with PDF URLs, go to paper_content
    # Otherwise still go (paper_content will handle the "no PDF" case)
    return "paper_content"


def route_after_paper_content(state: AgentState) -> Literal["generate"]:
    """After optional paper-content retrieval, always proceed to generate."""
    return "generate"