"""LangGraph nodes for the Phase 2 PaperPilot workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

import asyncio

import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


from .state import AgentState
from .routing import route_after_relevance

# Groq LLM configuration - read API key from environment
# Uses the GROQ_API_KEY already configured in .env
_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.2,
)
class QueryUnderstanding(BaseModel):
    """Structured interpretation of a PaperPilot user query."""

    intent: Literal[
        "general_answer",
        "local_rag",
        "academic_research",
    ]

    topic: Optional[str] = Field(
        default=None,
        description=(
            "The specific research topic being investigated. "
            "Use a precise technical description rather than generic keywords."
        ),
    )

    search_query: str = Field(
        description=(
            "A detailed but concise academic search query. "
            "Preserve the user's research intent and include important "
            "technical concepts, synonyms, methods, architectures, or "
            "research terminology when they are clearly implied by the query. "
            "Do not merely copy the user's wording or reduce the query to "
            "a few generic keywords. Do not invent a narrower research topic."
        ),
    )

    recency_requested: bool = Field(
        default=False,
        description=(
            "Whether the user explicitly requests recent, latest, new, "
            "or current research."
        ),
    )

async def query_understanding_node(state: AgentState) -> Dict[str, Any]:
    """Understand the query using LangChain structured output.

    The LLM is the primary semantic classifier.
    A deterministic fallback is used only if structured classification fails.
    """

    query = (state.get("user_query") or state.get("query") or "").strip()

    if not query:
        return {
            "intent": "general_answer",
            "route": "general_answer",
            "query_topic": None,
            "search_query": "",
            "recency_requested": False,
        }

    classifier = _llm.with_structured_output(QueryUnderstanding)

    prompt = f"""
You are the Query Understanding component of PaperPilot, an academic
research assistant.

Your task is to analyze the user's request and produce a structured
interpretation that will be used by the downstream LangGraph workflow.

You must distinguish between:
1. General questions
2. Questions about the user's local/uploaded documents
3. Requests for external academic research

Do not answer the user's question. Only classify and transform the request
for the downstream system.

==================================================
1. INTENT CLASSIFICATION
==================================================

Classify the user's intent into EXACTLY ONE of:

A. academic_research

Use academic_research when the user wants to:
- find academic papers
- search for research
- discover literature
- find studies
- find surveys
- find benchmarks
- find research papers
- retrieve papers
- recommend papers
- compare external academic papers
- investigate a research topic through academic literature
- find recent/latest/new research

Explicit research-discovery language such as:
"find papers", "show me papers", "give me papers", "find research",
"find studies", "search the literature", etc. MUST result in
academic_research.

Minor spelling mistakes must not affect this classification.

B. local_rag

Use local_rag ONLY when the user explicitly asks about information
contained in their uploaded/local documents, PDFs, papers, or library.

Examples of explicit local references:
- my uploaded paper
- my PDF
- my documents
- my library
- the paper I uploaded
- documents in my library

C. general_answer

Use general_answer for:
- explanations
- definitions
- conceptual questions
- how/why questions
- general technical questions
- casual conversation
- questions that do not request external academic research
- questions that do not explicitly refer to the user's local documents

IMPORTANT:
Do not classify a question as academic_research merely because it
contains a technical or scientific topic.

For example:
"What is BM25?" → general_answer

But:
"Find papers about BM25." → academic_research


==================================================
2. TOPIC EXTRACTION
==================================================

For academic_research, identify the main research topic.

The topic should be a concise semantic description of what the user
is researching.

It should capture the subject rather than merely copying keywords.

Examples of the type of interpretation expected:

"papers about memory in LLMs"
→ topic: "LLM memory"

"papers comparing sparse and dense retrieval"
→ topic: "sparse and dense retrieval comparison"

"research on hallucination detection in LLMs"
→ topic: "LLM hallucination detection"

Do not invent a narrower topic than the user requested.

For general_answer and local_rag, topic may be null when a research
topic is not needed.


==================================================
3. ACADEMIC SEARCH QUERY
==================================================

Only create a search_query when intent is academic_research.

For general_answer:
- search_query MUST be null.

For local_rag:
- search_query MUST be null unless an external academic search is
  explicitly requested.

    For academic_research:

    Construct a dedicated academic retrieval query.

    SPECIAL CASE — SPECIFIC PAPER REQUESTS

    If the user explicitly names or clearly identifies a specific academic
    paper, treat the request as a specific-paper lookup rather than a broad
    topic search.

    Examples:
    - "Find the paper Attention Is All You Need"
    - "Find Attention Is All You Need and explain the architecture"
    - "Find the paper BERT and explain its pre-training method"

    When a specific paper is identified:

    1. Preserve the paper title or identifying phrase in search_query.
    2. Do NOT append technical concepts from the user's explanation request.
    3. Do NOT transform the specific paper lookup into a broad topic query.
    4. The search_query should primarily identify the requested paper.
    5. The original user query will be used separately for PDF/content retrieval.

    Example:

    User:
    "Find the paper Attention Is All You Need and explain its Transformer
    encoder and decoder architecture and how multi-head attention is used."

    Good search_query:
    "Attention Is All You Need"

    Bad search_query:
    "Attention Is All You Need Transformer encoder decoder multi-head attention architecture"

    The paper-identification query and the content question are separate concerns.

    The search_query will be sent to academic search providers such as
    ArXiv and OpenAlex.


The search_query is NOT the user's original question.

The search_query is NOT the final answer.

The search_query is NOT simply a short list of generic keywords.

Construct it using the following reasoning process:

STEP 1 — Identify the central research subject.

Determine exactly what scientific, technical, or research topic the
user wants papers about.

STEP 2 — Identify the user's research objective.

Determine what the user wants to learn or investigate about that subject.

Possible objectives include:
- methods
- techniques
- architectures
- mechanisms
- evaluation
- benchmarking
- comparison
- detection
- mitigation
- implementation
- optimization
- efficiency
- performance
- reliability
- applications
- limitations
- surveys

Only include an objective when it is supported by the user's request.

STEP 3 — Extract important technical concepts.

Identify the technical concepts necessary to retrieve papers relevant
to the user's actual request.

STEP 4 — Add useful academic terminology.

Add closely related technical terminology or established synonyms when
they improve the probability of retrieving relevant academic literature.

Do not add concepts merely because they are common in the field.

Every important concept added should be directly stated or reasonably
implied by the user's request.

STEP 5 — Preserve important constraints.

Preserve constraints such as:
- comparison targets
- application domains
- specific methods
- architectures
- evaluation goals
- benchmarks
- mechanisms
- requested tasks
- explicit time requirements

Do not invent constraints.

STEP 6 — Correct obvious spelling mistakes.

Interpret obvious misspellings according to the surrounding context.

For example:
"retrival" → "retrieval"
"pappers" → "papers"
"persistance" → "persistence"

STEP 7 — Remove conversational wording.

Remove phrases such as:
- can you
- could you
- please
- show me
- find me
- give me
- I want
- I am looking for
- tell me
- can you find

STEP 8 — Produce the final retrieval query.

The resulting query should be:

- semantically precise
- technically meaningful
- focused on the user's research intent
- suitable for academic search engines
- concise enough to avoid unrelated results

Do NOT optimize for query length.

Use as many meaningful concepts as necessary to represent the user's
research intent accurately, but do not create a keyword dump.

Do NOT copy the original user sentence verbatim.

Do NOT add unrelated concepts.

Do NOT silently change the user's research question.


==================================================
4. ACRONYMS AND AMBIGUOUS TERMS
==================================================

Handle acronyms according to their context.

If the context clearly determines the meaning, use the appropriate
expanded terminology.

Example:

"papers about ANN retrieval"
→ interpret ANN as Approximate Nearest Neighbor.

A suitable search query could contain:
"approximate nearest neighbor retrieval vector search similarity search"

However, if an acronym has multiple plausible meanings and the user
provides insufficient context, DO NOT combine unrelated meanings into
one search query.

For example:

User:
"Find papers about ANN."

Do NOT produce a query containing both:
- artificial neural networks
- approximate nearest neighbors

because this can retrieve unrelated literature.

Instead, preserve the ambiguity:

topic:
"ANN"

search_query:
"ANN"

If the user's later context clarifies the meaning, use that context.


==================================================
5. RECENCY
==================================================

Set recency_requested to TRUE only when the user explicitly requests
recent, latest, newest, current, or otherwise time-constrained research.

Examples:

"Find recent papers about RAG."
→ recency_requested: true

"Find the latest research on LLM agents."
→ recency_requested: true

"Find papers about RAG."
→ recency_requested: false

Do not invent a year or date range.

Do not add a date to search_query unless the user explicitly provides
a time requirement.


==================================================
6. CONVERSATIONAL CONTEXT
==================================================

Use available conversation history when the current query is a
follow-up to an earlier research discussion.

Examples:
- "find more papers about this"
- "show me newer ones"
- "compare these papers"
- "find papers similar to them"

Use the previous context to resolve references when possible.

Do not invent missing context.

The current user query remains the primary source of intent.


==================================================
7. QUALITY REQUIREMENTS
==================================================

Before producing the structured output, internally verify:

1. Is the intent correct?
2. If academic_research, does the topic accurately represent the
   research subject?
3. Does the search_query represent the user's actual research intent?
4. Did the search_query preserve important technical concepts?
5. Did it preserve comparison/evaluation/methodology/application
   requirements when present?
6. Did it remove conversational wording?
7. Did it avoid unrelated concepts?
8. Did it avoid inventing information?
9. Did it handle ambiguous acronyms appropriately?
10. Is search_query null when academic retrieval is not required?

IMPORTANT:
Do not expose this reasoning or checklist in the response.

Return ONLY the structured QueryUnderstanding object.

==================================================

USER QUERY
==================================================

{query}
"""

    try:
        result = await classifier.ainvoke(prompt)

        return {
            "intent": result.intent,
            "route": result.intent,
            "query_topic": result.topic,
            "search_query": result.search_query,
            "recency_requested": result.recency_requested,
        }

    except Exception:
        # Deterministic fallback only if LLM structured classification fails.
        normalized = re.sub(r"[^a-z0-9\s]", " ", query.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()

        local_patterns = (
            "my pdf",
            "my document",
            "my documents",
            "my paper",
            "my papers",
            "my library",
            "uploaded",
            "local library",
        )

        academic_patterns = (
            "paper",
            "papers",
            "research",
            "study",
            "studies",
            "literature",
            "survey",
            "benchmark",
            "arxiv",
            "academic",
        )

        if any(pattern in normalized for pattern in local_patterns):
            intent = "local_rag"
        elif any(pattern in normalized for pattern in academic_patterns):
            intent = "academic_research"
        else:
            intent = "general_answer"

        return {
            "intent": intent,
            "route": intent,
            "query_topic": None,
            "search_query": query,
            "recency_requested": bool(
                re.search(r"\b(recent|latest|new|current)\b", normalized)
            ),
        }
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

    query = (
        state.get("search_query")
        or state.get("query_topic")
        or state.get("user_query")
        or state.get("query", "")
    )

    max_results = settings.academic_search_max_results

    # Both providers are independently awaited. A timeout/failure in one
    # must not discard results returned by the other.
    results = await asyncio.gather(
        ArxivProvider().search(
            query,
            max_results=max_results,
        ),
        OpenAlexProvider().search(
            query,
            max_results=max_results,
        ),
        return_exceptions=True,
    )

    arxiv_papers = (
        results[0]
        if isinstance(results[0], list)
        else []
    )

    openalex_papers = (
        results[1]
        if isinstance(results[1], list)
        else []
    )

    all_papers = arxiv_papers + openalex_papers

    return {
        "academic_papers": all_papers,
        "provider_result_counts": {
            "arxiv": len(arxiv_papers),
            "openalex": len(openalex_papers),
        },
    }


async def normalize_papers_node(state: AgentState) -> Dict[str, Any]:
    """Normalize, deduplicate, rank, and expose academic papers to generation."""

    from config import settings
    from retrieval.academic.ranker import normalize_and_rank

    query = (
        state.get("search_query")
        or state.get("query_topic")
        or state.get("user_query")
        or state.get("query", "")
    )

    ranked_papers = normalize_and_rank(
        query,
        state.get("academic_papers", []),
        top_k=settings.academic_top_k,
    )

    final_docs = []

    for paper in ranked_papers:
        final_docs.append(
            {
                "kind": "academic",
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "publication_date": paper.publication_date,
                "year": paper.year,
                "provider": paper.provider,
                "paper_url": paper.paper_url,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "openalex_id": paper.openalex_id,
                "citation_count": paper.citation_count,
                "relevance_score": paper.relevance_score,
            }
        )

    return {
        "ranked_papers": ranked_papers,
        "normalized_papers": ranked_papers,
        "final_docs": final_docs,
    }

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
    history = state.get("conversation_history", [])
    # Generate answer using the existing generator pipeline
    try:
        result = await generate_answer(
            query=query,
            final_docs=final_docs,
            citations=citations,
            intent=intent,
            history=history,
            paper_content_texts=state.get("paper_content_texts", []),
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
async def paper_content_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve temporary, question-relevant PDF content for academic papers.

    Public academic PDFs are downloaded only for the current research
    request. Their contents are processed through the temporary PDF
    retrieval pipeline and are never added to the permanent Weaviate
    library.

    The node uses the already normalized/ranked academic papers and
    retrieves content from up to the top two papers that expose a
    public PDF URL.
    """

    from retrieval.academic.pdf_fetch import fetch_paper_context

    query = (
        state.get("user_query")
        or state.get("query")
        or ""
    ).strip()

    ranked_papers = state.get("ranked_papers", [])

    if not ranked_papers:
        return {
            "paper_content_texts": []
        }

    paper_content_texts: List[Dict[str, Any]] = []

    # Only inspect the top two ranked papers.
    # This keeps PDF retrieval bounded and avoids unnecessary downloads.
    papers_with_pdf = [
        paper
        for paper in ranked_papers
        if getattr(paper, "pdf_url", None)
    ][:2]

    for paper in papers_with_pdf:
        try:
            context = await fetch_paper_context(
                paper=paper,
                query=query,
                max_chunks=8,
            )

            if not context:
                continue

            paper_content_texts.append(
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "publication_date": paper.publication_date,
                    "paper_url": paper.paper_url,
                    "pdf_url": paper.pdf_url,
                    "doi": paper.doi,
                    "arxiv_id": paper.arxiv_id,
                    "openalex_id": paper.openalex_id,
                    "content": context,
                }
            )

        except Exception as exc:
            # A failure for one paper must not stop the entire
            # academic research workflow.
            continue

    return {
        "paper_content_texts": paper_content_texts
    }