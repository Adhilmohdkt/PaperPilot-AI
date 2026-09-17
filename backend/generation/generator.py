"""Answer generation for the PaperPilot LangGraph workflow.

Uses Groq through LangChain ChatGroq for answer generation.
Gemini is reserved for embeddings elsewhere in the application.

The generator supports:
- General conversational answers
- Local library RAG answers
- Academic research answers
- Conversation history
- Citation-aware responses
- Deterministic citation validation
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


async def generate_answer(
    query: str,
    final_docs: List[Dict[str, Any]],
    citations: Optional[List[Dict[str, Any]]] = None,
    intent: Optional[str] = None,
    history: Optional[List[Any]] = None,
    paper_content_texts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate the final answer using Groq through LangChain.

    Args:
        query:
            The user's current question.

        final_docs:
            Retrieved context. Each document should contain a ``kind`` field:
            - ``library`` for local RAG sources
            - ``academic`` for academic papers

        citations:
            Existing citation validation results from the workflow.

        intent:
            Current workflow intent:
            - ``general_answer``
            - ``local_rag``
            - ``academic_research``

        history:
            Previous conversation turns represented as dictionaries or
            LangChain BaseMessage objects.

        paper_content_texts:
            Temporary question-relevant content retrieved from public
            academic PDFs. This content is used only for the current
            generation request and is never permanently indexed.

    Returns:
        A dictionary containing:
        - ``answer``: generated answer text
        - ``citations``: validated citation records
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured.")

    if citations is None:
        citations = []

    if paper_content_texts is None:
        paper_content_texts = []

    model = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.2,
    )

    # ------------------------------------------------------------------
    # General conversational path
    # ------------------------------------------------------------------

    if intent == "general_answer":
        system_message = SystemMessage(
            content=(
                "You are PaperPilot, a knowledgeable and precise AI research "
                "assistant. Answer the user's question directly, clearly, "
                "and accurately. Use the previous conversation when it is "
                "relevant to the current question."
            )
        )

        messages: List[BaseMessage] = [system_message]

        messages.extend(_convert_history_to_messages(history))

        # Add the current question exactly once.
        messages.append(HumanMessage(content=query))

        response = await model.ainvoke(messages)

        answer = _extract_response_text(response)

        return {
            "answer": answer,
            "citations": [],
        }

    # ------------------------------------------------------------------
    # Separate retrieved evidence
    # ------------------------------------------------------------------

    library_chunks = [
        document
        for document in final_docs
        if document.get("kind") == "library"
    ]

    academic_entries = [
        document
        for document in final_docs
        if document.get("kind") == "academic"
    ]

    # ------------------------------------------------------------------
    # Build local-library context
    # ------------------------------------------------------------------

    library_parts: List[str] = []

    for index, chunk in enumerate(library_chunks, start=1):
        text = chunk.get("text", "").strip()

        if not text:
            continue

        library_parts.append(
            f"[Source {index}]: {text}"
        )

    library_context = "\n\n".join(library_parts)

    # ------------------------------------------------------------------
    # Build academic-paper context
    # ------------------------------------------------------------------

    academic_parts: List[str] = []

    for index, paper in enumerate(academic_entries, start=1):
        if not isinstance(paper, dict):
            continue

        parts: List[str] = [f"[Paper {index}]"]

        title = paper.get("title")
        if title:
            parts.append(f"Title: {title}")

        authors = paper.get("authors")
        if authors:
            if isinstance(authors, list):
                parts.append(
                    f"Authors: {', '.join(str(author) for author in authors)}"
                )
            else:
                parts.append(f"Authors: {authors}")

        year = paper.get("year")
        if year:
            parts.append(f"Year: {year}")

        publication_date = paper.get("publication_date")
        if publication_date:
            parts.append(
                f"Publication date: {publication_date}"
            )

        abstract = paper.get("abstract")
        if abstract:
            # Keep academic metadata context compact.
            parts.append(
                f"Abstract: {str(abstract)[:400]}"
            )

        doi = paper.get("doi")
        if doi:
            parts.append(f"DOI: {doi}")

        paper_url = paper.get("paper_url")
        if paper_url:
            parts.append(f"URL: {paper_url}")

        pdf_url = paper.get("pdf_url")
        if pdf_url:
            parts.append(f"PDF: {pdf_url}")

        arxiv_id = paper.get("arxiv_id")
        if arxiv_id:
            parts.append(f"arXiv: {arxiv_id}")

        openalex_id = paper.get("openalex_id")
        if openalex_id:
            parts.append(f"OpenAlex: {openalex_id}")

        academic_parts.append("\n".join(parts))

    # ------------------------------------------------------------------
    # Add temporary PDF content to the corresponding academic paper.
    # ------------------------------------------------------------------

    for paper_content in paper_content_texts:
        if not isinstance(paper_content, dict):
            continue

        title = paper_content.get("title")
        content = paper_content.get("content")

        if not title or not content:
            continue

        for index, paper in enumerate(academic_entries):
            if not isinstance(paper, dict):
                continue

            if paper.get("title") != title:
                continue

            academic_parts[index] += (
                "\n\nRelevant PDF content:\n"
                f"{content}"
            )

            break

    academic_context = "\n\n".join(academic_parts)

    # ------------------------------------------------------------------
    # Combine retrieved context
    # ------------------------------------------------------------------

    if library_context and academic_context:
        full_context = (
            f"{library_context}\n\n{academic_context}"
        )
    elif library_context:
        full_context = library_context
    elif academic_context:
        full_context = academic_context
    else:
        full_context = "No retrieved evidence was found."

    # ------------------------------------------------------------------
    # RAG / academic research system prompt
    # ------------------------------------------------------------------

    system_message = SystemMessage(
        content=(
            "You are PaperPilot, a precise AI research assistant.\n\n"
            "Answer the user's question using the retrieved evidence "
            "provided in the conversation.\n\n"
            "Citation rules:\n"
            "- Use [Source N] for factual claims supported by the user's "
            "local library documents.\n"
            "- Use [Paper N] for factual claims supported by academic "
            "papers returned by the research providers.\n"
            "- Do not invent citations.\n"
            "- Do not fabricate page numbers, DOIs, URLs, authors, or "
            "publication details.\n"
            "- If the retrieved evidence is insufficient, say so clearly.\n"
            "- Do not present unsupported information as if it came from "
            "the retrieved evidence.\n"
            "- Distinguish clearly between local sources and academic papers."
        )
    )

    messages: List[BaseMessage] = [system_message]

    # Previous conversation is context, not the current user query.
    messages.extend(_convert_history_to_messages(history))

    # ------------------------------------------------------------------
    # Current question + retrieved evidence
    #
    # IMPORTANT:
    # The current question is included only once.
    # ------------------------------------------------------------------

    user_message = (
        "Retrieved evidence:\n\n"
        f"{full_context}\n\n"
        "Current user question:\n"
        f"{query}"
    )

    messages.append(
        HumanMessage(content=user_message)
    )

    # ------------------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------------------

    response = await model.ainvoke(messages)

    answer = _extract_response_text(response)

    # ------------------------------------------------------------------
    # Deterministic citation validation
    # ------------------------------------------------------------------

    validated_citations = validate_citations(
        answer=answer,
        final_docs=final_docs,
        existing_citations=citations,
    )

    return {
        "answer": answer,
        "citations": validated_citations,
    }


def _convert_history_to_messages(
    history: Optional[List[Any]],
) -> List[BaseMessage]:
    """Convert stored conversation history into LangChain messages.

    Supports both:
    - dictionaries: {"role": "...", "content": "..."}
    - existing LangChain BaseMessage objects
    """

    messages: List[BaseMessage] = []

    for message in history or []:

        if isinstance(message, BaseMessage):
            messages.append(message)
            continue

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")

        if not content:
            continue

        if role == "user":
            messages.append(
                HumanMessage(content=str(content))
            )

        elif role == "assistant":
            messages.append(
                AIMessage(content=str(content))
            )

    return messages


def _extract_response_text(response: Any) -> str:
    """Extract plain text from a LangChain model response."""

    content = getattr(response, "content", None)

    if content is None:
        return str(response)

    if isinstance(content, str):
        return content

    # Some LangChain model responses can contain structured content blocks.
    if isinstance(content, list):
        text_parts: List[str] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    text_parts.append(str(text))

        if text_parts:
            return "".join(text_parts)

    return str(content)


def validate_citations(
    answer: str,
    final_docs: List[Dict[str, Any]],
    existing_citations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Validate [Source N] and [Paper N] references deterministically."""

    validated_citations: List[Dict[str, Any]] = []

    existing_citations = existing_citations or []

    # --------------------------------------------------------------
    # Library sources
    # --------------------------------------------------------------

    source_entries = {
        str(index): document
        for index, document in enumerate(
            (
                document
                for document in final_docs
                if document.get("kind") == "library"
            ),
            start=1,
        )
    }

    source_refs = re.findall(
        r"\[Source\s+(\d+)\]",
        answer,
    )

    for reference_number in source_refs:

        if reference_number in source_entries:
            validated_citations.append(
                {
                    "source": f"Source {reference_number}",
                    "valid": True,
                    "details": "Library source exists in context",
                }
            )
        else:
            validated_citations.append(
                {
                    "source": f"Source {reference_number}",
                    "valid": False,
                    "details": "Library source not found in context",
                }
            )

    # --------------------------------------------------------------
    # Academic papers
    # --------------------------------------------------------------

    academic_entries = [
        document
        for document in final_docs
        if document.get("kind") == "academic"
    ]

    paper_refs = re.findall(
        r"\[Paper\s+(\d+)\]",
        answer,
    )

    for reference_number in paper_refs:

        index = int(reference_number) - 1

        if 0 <= index < len(academic_entries):
            validated_citations.append(
                {
                    "source": f"Paper {reference_number}",
                    "valid": True,
                    "details": "Academic paper exists in context",
                }
            )
        else:
            validated_citations.append(
                {
                    "source": f"Paper {reference_number}",
                    "valid": False,
                    "details": "Academic paper not found in context",
                }
            )

    # --------------------------------------------------------------
    # Preserve citations already produced by the workflow
    # --------------------------------------------------------------

    for existing in existing_citations:

        if existing not in validated_citations:
            validated_citations.append(existing)

    return validated_citations