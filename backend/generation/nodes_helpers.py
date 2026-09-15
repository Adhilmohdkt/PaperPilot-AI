"""Helper functions shared across the Phase 2 generation nodes."""
from __future__ import annotations

import os
from typing import Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage as HCHumanMessage

from config import settings
from generation.generator import _context, _history_messages


def _get_llm(streaming: bool = True):
    """Get configured Gemini LLM instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=settings.gemini_model,
        streaming=streaming,
        temperature=0.2,
    )


def _content_to_text(content: Any) -> str:
    """Normalize Gemini's string or structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content or "")


def _history_messages(history: Optional[List[Dict[str, str]]]) -> list[tuple[str, str]]:
    """Convert history dicts to LangChain message tuples."""
    if not history:
        return []
    return [
        ("ai" if item.get("role") == "assistant" else "human", item.get("content", ""))
        for item in history[-8:]
        if item.get("content")
    ]


def _validate_citation_structure(answer: str, docs: List[Any]) -> List[Dict[str, Any]]:
    """Validate that [Source N] or [Paper N] references in the answer map to actual docs.

    Keeps it deterministic: only checks index ranges, no LLM judge.
    """
    import re

    citations = []
    pattern = r"\[(?:Source|Paper)\s+(\d+)\]"
    matches = re.findall(pattern, answer)

    for match in matches:
        try:
            idx = int(match) - 1  # 0-based
            if 0 <= idx < len(docs):
                doc = docs[idx]
                kind = doc.get("kind", "library")
                label = "Source" if kind == "library" else "Paper"
                citations.append({
                    "source_index": idx + 1,
                    "label": label,
                    "source": doc.get("source", "Unknown"),
                    "page": doc.get("page"),
                    "kind": kind,
                    "validated": True,
                })
            else:
                citations.append({
                    "source_index": idx + 1,
                    "label": "Source" if idx < len(docs) else "Paper",
                    "validated": False,
                    "error": "Index out of range",
                })
        except ValueError:
            citations.append({
                "source_index": match,
                "validated": False,
                "error": "Invalid citation format",
            })

    # Ensure every doc used gets a citation entry if not already referenced
    cited_indices = {c["source_index"] for c in citations}
    for i, doc in enumerate(docs):
        if (i + 1) not in cited_indices:
            kind = doc.get("kind", "library")
            label = "Source" if kind == "library" else "Paper"
            citations.append({
                "source_index": i + 1,
                "label": label,
                "source": doc.get("source", "Unknown"),
                "page": doc.get("page"),
                "kind": kind,
                "validated": True,
                "referenced_in_answer": False,
            })

    return citations