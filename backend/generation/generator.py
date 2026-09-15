"""Generation module for the PaperPilot Phase 2 workflow.

Provides answer generation using Groq through LangChain ChatGroq,
with citation-aware prompt engineering that distinguishes between
local library sources ([Source N]) and academic papers ([Paper N]).

Also provides a streaming generation function for the Phase 1 chat endpoint.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Iterable

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


async def generate_answer(
    query: str,
    final_docs: List[Dict[str, Any]],
    citations: Optional[List[Dict[str, Any]]] = None,
    intent: Optional[str] = None,
    history: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Generate the final answer using the Groq LLM, grounded in retrieved context.

    Distinguishes between local library sources ([Source N]) and academic papers ([Paper N]).
    For general_answer intent, provides direct generation without requiring retrieval evidence.

    Args:
        query: The user's original question.
        final_docs: Document chunks used as context, each with a "kind" field
            ("library" for local papers, "academic" for academic papers).
        citations: Citation validation results from citation_validation_node.
        intent: The determined intent/route (e.g., "general_answer", "local_rag", "academic_research").
        history: Prior conversation messages for multi-turn context.

    Returns:
        Dict with "answer" (str) and "citations" (list of citation dicts).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    if citations is None:
        citations = []

    model = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.2,
    )

    if intent == "general_answer":
        system_msg = (
            "You are PaperPilot, a knowledgeable and precise AI research assistant. "
            "Answer the user's question directly, clearly, and accurately."
        )
        messages: List[BaseMessage] = [SystemMessage(content=system_msg)]
        for msg in (history or []):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            elif isinstance(msg, BaseMessage):
                messages.append(msg)
        messages.append(HumanMessage(content=query))
        response = await model.ainvoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "answer": answer,
            "citations": [],
        }

    # Separate library and academic sources
    library_chunks = [d for d in final_docs if d.get("kind") == "library"]
    academic_entries = [d for d in final_docs if d.get("kind") == "academic"]

    # Build library context using [Source N] format
    library_parts: List[str] = []
    for i, chunk in enumerate(library_chunks, start=1):
        text = chunk.get("text", "")
        library_parts.append(f"[Source {i}]: {text}")

    # Build academic context using [Paper N] format
    academic_parts: List[str] = []
    for i, entry in enumerate(academic_entries, start=1):
        paper = entry if isinstance(entry, dict) else {}
        if not paper:
            continue
        parts: List[str] = [f"[Paper {i}]"]
        if paper.get("title"):
            parts.append(f"Title: {paper.get('title')}")
        if paper.get("authors"):
            authors = paper.get("authors", [])
            if authors:
                parts.append(f"Authors: {', '.join(authors)}")
        if paper.get("year"):
            parts.append(f"Year: {paper.get('year')}")
        if paper.get("abstract"):
            abstract = paper.get("abstract", "")
            parts.append(f"Abstract: {abstract[:400]}")
        if paper.get("doi"):
            parts.append(f"DOI: {paper.get('doi')}")
        if paper.get("paper_url"):
            parts.append(f"URL: {paper.get('paper_url')}")
        if paper.get("arxiv_id"):
            parts.append(f"arXiv: {paper.get('arxiv_id')}")
        academic_parts.append(" ".join(parts))

    library_context = "\n\n".join(library_parts) if library_parts else ""
    academic_context = "\n\n".join(academic_parts) if academic_parts else ""

    # Combine contexts
    if library_context and academic_context:
        full_context = f"{library_context}\n\n{academic_context}"
    elif library_context:
        full_context = library_context
    elif academic_context:
        full_context = academic_context
    else:
        full_context = "No retrieved evidence was found."

    # Prepare messages for the LLM
    system_msg = (
        "You are PaperPilot, a precise research assistant. "
        "Answer only from the retrieved evidence. Use inline citations "
        "in the form [Source N] for factual claims from the user's library, "
        "and [Paper N] for academic papers. If evidence is insufficient, "
        "say so clearly. Distinguish between [Source N] and [Paper N] "
        "formats. Do not fabricate page numbers or DOIs for academic papers."
    )

    user_msg_content = query

    # Build the message list
    messages: List[BaseMessage] = [SystemMessage(content=system_msg)]
    for msg in (history or []):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(msg, BaseMessage):
            messages.append(msg)

    # Add the user query
    messages.append(HumanMessage(content=user_msg_content))

    # Add context as a user message postscript
    full_user_message = f"{full_context}\n\nUser question: {user_msg_content}" if full_context else user_msg_content
    messages.append(HumanMessage(content=full_user_message))

    # Generate using Groq (async)
    response = await model.ainvoke(messages)

    answer = response.content if hasattr(response, "content") else str(response)

    # Deterministic citation validation (no LLM)
    import re

    validated_citations: List[Dict[str, Any]] = []

    # Extract [Source N] references from the answer
    source_refs = re.findall(r"\[Source (\d+)\]", answer)

    # Extract [Paper N] references from the answer
    paper_refs = re.findall(r"\[Paper (\d+)\]", answer)

    # Validate [Source N] references
    source_entries = {str(i + 1): doc for i, doc in enumerate(final_docs) if doc.get("kind") == "library"}
    for ref_num in source_refs:
        if ref_num in source_entries:
            validated_citations.append({
                "source": f"Source {ref_num}",
                "valid": True,
                "details": "Library source exists in context",
            })
        else:
            validated_citations.append({
                "source": f"Source {ref_num}",
                "valid": False,
                "details": "Library source not found in context",
            })

    # Validate [Paper N] references
    academic_entries = [d for d in final_docs if d.get("kind") == "academic"]
    for ref_num in paper_refs:
        idx = int(ref_num) - 1  # 0-based index
        if 0 <= idx < len(academic_entries):
            validated_citations.append({
                "source": f"Paper {ref_num}",
                "valid": True,
                "details": "Academic paper exists in context",
            })
        else:
            validated_citations.append({
                "source": f"Paper {ref_num}",
                "valid": False,
                "details": "Academic paper not found in context",
            })

    # Append any existing citations from the state
    for existing in citations:
        # Avoid duplicates
        if existing not in validated_citations:
            validated_citations.append(existing)

    return {
        "answer": answer,
        "citations": validated_citations,
    }


def generate_answer_stream(
    query: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> Iterable[str]:
    """Yield answer text while keeping citations tied to supplied retrieved evidence.

    Uses Gemini embeddings and is kept for Phase 1 chat/streaming compatibility.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are PaperPilot, a precise research assistant. Answer only from the retrieved evidence.
Use inline citations in the form [Source N] for factual claims. If evidence is insufficient, say so clearly.

Retrieved evidence:
{context}"""),
        ("human", "{query}"),
    ])
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-embedding-2-preview"),
        google_api_key=api_key,
        temperature=0.2,
        streaming=True,
    )
    chain = prompt | model | StrOutputParser()
    yield from chain.stream({"query": query, "context": _context(chunks)})