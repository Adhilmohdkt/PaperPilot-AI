"""State shared by the PaperPilot LangGraph workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """All fields exchanged by the graph nodes."""

    conversation_id: str
    query: str
    user_query: str
    source_filter: Optional[str]
    conversation_history: List[Dict[str, str]]
    intent: Literal["general_answer", "local_rag", "academic_research"]
    route: str
    final_docs: List[Dict[str, Any]]
    has_sufficient_evidence: bool
    academic_papers: List[Any]
    ranked_papers: List[Any]
    normalized_papers: List[Any]
    paper_content_texts: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    response: str
    error: str
    search_query: str
    query_topic: Optional[str]


def create_initial_state(
    query: str,
    source_filter: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_id: str = "",
) -> AgentState:
    """Create a complete, API-safe initial graph state."""
    return {
    "conversation_id": conversation_id,
    "query": query,
    "user_query": query,
    "search_query": "",
    "source_filter": source_filter,
    "conversation_history": history or [],
    "intent": "general_answer",
    "route": "general_answer",
    "final_docs": [],
    "academic_papers": [],
    "ranked_papers": [],
    "normalized_papers": [],
    "paper_content_texts": [],
    "citations": [],
    "response": "",
}