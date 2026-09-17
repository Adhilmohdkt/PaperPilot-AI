"""Routing functions for the PaperPilot LangGraph workflow.

LangGraph owns workflow orchestration and conditional routing.

The primary route is selected by the query-understanding node and stored
in AgentState["route"].

Additional conditional routing is used only where a workflow decision is
actually required, such as determining whether local RAG evidence is
sufficient.
"""

from __future__ import annotations

from typing import Literal

from .state import AgentState


# ---------------------------------------------------------------------------
# Local RAG relevance routing
# ---------------------------------------------------------------------------


def route_after_relevance(
    state: AgentState,
) -> Literal["generate", "academic_search"]:
    """Route after checking the quality of local RAG evidence.

    Sufficient local evidence:
        → generate

    Insufficient local evidence:
        → academic_search
    """

    if state.get("has_sufficient_evidence", False):
        return "generate"

    return "academic_search"