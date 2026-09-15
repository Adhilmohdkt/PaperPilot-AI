"""LangGraph nodes for the Phase 2 PaperPilot workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import asyncio

import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .state import AgentState
from .routing import route_after_relevance, route_after_academic_search, route_after_ranking, route_after_paper_content

# Groq LLM configuration - read API key from environment
# Uses the GROQ_API_KEY already configured in .env
_llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.2,
)


async def query_understanding_node(state: AgentState) -> Dict[str, Any]:
    """Understand the user query and extract structured intent.

    Determines the intent/route for the query using pattern-based classification
    for reliable routing.

    Sets:
        intent: one of "general_answer", "local_rag", "academic_research"
    """
    query = state.get("user_query", "")

    # Pattern-based classification for reliable routing
    normalized = re.sub(r"[^a-z0-9\s]", "", query.lower()).strip()

    # Academic research keywords
    academic_keys = {
        "find", "search", "papers", "article", "study", "research",
        "evaluation", "benchmark", "comparison", "survey",
    }
    # Local RAG keywords (references to the user's PDF library)
    local_keys = {
        "pdf", "library", "document", "my papers", "my library",
        "local", "uploaded",
    }
    # Casual/conversational keywords
    casual_keys = {
        "hello", "hi", "how are you", "good morning", "good afternoon",
        "good evening", "thank you", "thanks", "hey", "hey there",
        "bye", "goodbye", "chat", "conversation",
    }

    # Count matches in each category
    academic_score = sum(1 for kw in academic_keys if kw in normalized)
    local_score = sum(1 for kw in local_keys if kw in normalized)
    casual_score = sum(1 for kw in casual_keys if kw in normalized)

    # Determine intent based on highest score, with tiebreaking
    if casual_score > 0 and (academic_score == 0 and local_score == 0):
        intent = "general_answer"
    elif academic_score > local_score and academic_score > 0:
        intent = "academic_research"
    elif local_score > 0:
        intent = "local_rag"
    elif academic_score > 0:
        intent = "academic_research"
    else:
        intent = "general_answer"

    return {"intent": intent}


async def router_node(state: AgentState) -> Dict[str, Any]:
    """Route the query to the appropriate path based on understood intent.

    The routing is represented through LangGraph conditional edges.
    In Phase 2, three paths are implemented:
    - general_answer -> generate
    - local_rag -> local_retrieve -> relevance_check -> generate
    - academic_research -> academic_search -> normalize_papers -> paper_content -> generate
    """
    intent = state.get("intent", "general_answer")

    return {"route": intent}


async def local_retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve documents from the local Weaviate knowledge base.

    Uses the existing retriever function with hybrid search (BM25 + vector)
    and Cohere reranking.

    Sets:
        final_docs: retrieved document chunks with kind="library"
    """
    from retrieval.retriever import retrieve

    query = state.get("query", "")
    source_filter = state.get("source_filter")

    # Use the existing retriever with hybrid BM25+vector search + Cohere reranking
    docs = retrieve(query, source_filter=source_filter)

    # Normalize results to have consistent structure
    final_docs = [
        {
            "text": doc.get("text", ""),
            "source": doc.get("source", "Unknown"),
            "page": doc.get("page"),
            "kind": "library",
            "metadata": {
                "chunk_id": doc.get("chunk_id"),
                "content_hash": doc.get("content_hash"),
            },
        }
        for doc in docs
    ]

    return {"final_docs": final_docs}


async def relevance_check_node(state: AgentState) -> Dict[str, Any]:
    """Check if the retrieved local evidence is sufficient to answer the query.

    Sets:
        has_sufficient_evidence: bool
    """
    final_docs = state.get("final_docs", [])
    query = state.get("user_query", "")

    # Simple heuristic: sufficient evidence if we have at least the minimum relevant docs
    # and the docs contain some meaningful content related to the query
    min_relevant = state.get("min_relevant_docs", 1)

    if len(final_docs) >= min_relevant and any(
        doc.get("text", "").strip() for doc in final_docs
    ):
        has_sufficient = True
    else:
        has_sufficient = False

    return {
        "has_sufficient_evidence": has_sufficient,
    }


async def academic_search_node(state: AgentState) -> Dict[str, Any]:
    """Search for academic papers using arXiv and OpenAlex providers.

    Sets:
        academic_papers: list of raw AcademicPaper objects from providers
        ranked_papers: list of normalized/deduplicated papers with ranking
    """
    from retrieval.academic.arxiv_provider import ArxivProvider
    from retrieval.academic.openalex_provider import OpenAlexProvider
    from config import settings

    query = state.get("user_query") or state.get("query", "")
    max_results = settings.academic_search_max_results

    # Both providers are independently awaited. A timeout/failure in one must
    # not discard results returned by the other.
    results = await asyncio.gather(
        ArxivProvider().search(query, max_results=max_results),
        OpenAlexProvider().search(query, max_results=max_results),
        return_exceptions=True,
    )
    arxiv_papers = results[0] if isinstance(results[0], list) else []
    openalex_papers = results[1] if isinstance(results[1], list) else []
    all_papers = arxiv_papers + openalex_papers

    return {
        "academic_papers": all_papers,
        "provider_result_counts": {
            "arxiv": len(arxiv_papers),
            "openalex": len(openalex_papers),
        },
    }


async def normalize_papers_node(state: AgentState) -> Dict[str, Any]:
    """Normalize and deduplicate the ranked academic papers.

    Ensures papers have consistent metadata format and removes duplicates
    based on arXiv ID or OpenAlex DOI.

    Sets:
        normalized_papers: list of normalized AcademicPaper objects
    """
    from config import settings
    from retrieval.academic.ranker import normalize_and_rank

    query = state.get("user_query") or state.get("query", "")
    ranked_papers = normalize_and_rank(
        query,
        state.get("academic_papers", []),
        top_k=settings.academic_top_k,
    )
    # Generation and SSE only consume plain dictionaries. Preserve all useful
    # provider metadata rather than passing dataclass objects through state.
    normalized = [
        {
            "kind": "academic",
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "text": paper.abstract or "",
            "publication_date": paper.publication_date,
            "year": paper.year,
            "provider": paper.provider,
            "source": paper.paper_url,
            "paper_url": paper.paper_url,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "openalex_id": paper.openalex_id,
            "citation_count": paper.citation_count,
            "relevance_score": paper.relevance_score,
        }
        for paper in ranked_papers
    ]

    return {
        "ranked_papers": ranked_papers,
        "normalized_papers": normalized,
        "final_docs": normalized,
    }


async def paper_content_node(state: AgentState) -> Dict[str, Any]:
    """Optional PDF content retrieval for top-ranked academic papers.

    Fetches and processes PDF content for the top 1-2 ranked papers
    to provide additional context for generation. PDF content is temporary
    research context and is NOT permanently indexed into Weaviate.

    Sets:
        paper_content_texts: extracted text from top papers' PDFs
    """
    # Search-result metadata is sufficient for recommendations. PDF downloads
    # remain opt-in so a normal discovery query never downloads papers.
    return {"paper_content_texts": []}


async def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generate the final answer using the Groq LLM, grounded in retrieved context.

    Distinguishes between local library sources ([Source N]) and academic papers ([Paper N]).
    Implements citation validation as a deterministic post-processing step.

    Sets:
        answer: generated text
        final_docs: docs used as context
        citations: validated citation references
    """
    from generation.generator import generate_answer

    query = state.get("user_query") or state.get("query", "")
    final_docs = state.get("final_docs", [])
    citations = state.get("citations", [])
    intent = state.get("intent") or state.get("route", "general_answer")
    history = state.get("messages", [])

    # Generate answer using the existing generator pipeline
    try:
        result = await generate_answer(
            query=query,
            final_docs=final_docs,
            citations=citations,
            intent=intent,
            history=history,
        )
    except Exception as e:
        # Fallback error handling
        return {
            "answer": "I encountered an error while researching. Please try again or rephrase your question.",
            "final_docs": final_docs,
            "citations": [],
        }

    # Return response for compatibility with AgentState
    # The workflow stores the final answer in state["response"]
    response_text = result.get("answer", "")
    return {
        "response": response_text,
        "final_docs": final_docs,
        "citations": result.get("citations", []),
    }


async def citation_validation_node(state: AgentState) -> Dict[str, Any]:
    """Validate that citations/references in the generated answer correspond
    to available retrieved sources.

    Extracts [Source N] and [Paper N] references from the answer
    and verifies they exist in final_docs.

    Sets:
        citations: updated citation validity information
    """
    import re

    answer = state.get("response", "") or state.get("answer", "")
    final_docs = state.get("final_docs", [])
    existing_citations = state.get("citations", [])

    # Extract [Source N] references from the answer
    source_refs = re.findall(r"\[Source (\d+)\]", answer)

    # Extract [Paper N] references from the answer
    paper_refs = re.findall(r"\[Paper (\d+)\]", answer)

    # Validate [Source N] references - check against library docs
    source_entries = {}
    for i, doc in enumerate(final_docs):
        if doc.get("kind") == "library":
            source_entries[str(i + 1)] = doc

    validated_citations = list(existing_citations)

    # Validate and add [Source N] citations
    for ref_num in source_refs:
        if ref_num in source_entries:
            validated_citations.append(
                {
                    "source": f"Source {ref_num}",
                    "valid": True,
                    "details": "Library source exists in context",
                }
            )
        else:
            validated_citations.append(
                {
                    "source": f"Source {ref_num}",
                    "valid": False,
                    "details": "Library source not found in context",
                }
            )

    # Validate [Paper N] references - check against academic papers
    academic_entries = [d for d in final_docs if d.get("kind") == "academic"]

    for ref_num in paper_refs:
        idx = int(ref_num) - 1  # 0-based index
        if 0 <= idx < len(academic_entries):
            validated_citations.append(
                {
                    "source": f"Paper {ref_num}",
                    "valid": True,
                    "details": "Academic paper exists in context",
                }
            )
        else:
            validated_citations.append(
                {
                    "source": f"Paper {ref_num}",
                    "valid": False,
                    "details": "Academic paper not found in context",
                }
            )

    return {
        "citations": validated_citations,
    }


async def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """Error handler node for the LangGraph workflow.

    Captures and records errors that occur during workflow execution.
    Preserves the conversation state so it can be retried or inspected.

    Sets:
        error: error information
        error_occurred: bool flag
    """
    error = state.get("error")
    if error is None:
        answer = state.get("answer", "")
        if answer and "encountered an error" in answer:
            error = answer
        else:
            error = "Unknown error occurred during workflow execution"

    return {
        "error": error,
        "error_occurred": error is not None,
    }
