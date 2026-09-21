"""Normalize retrieved documents into UI-safe source cards."""

from __future__ import annotations

from typing import Any


def _http_url(*candidates: Any) -> str | None:
    for value in candidates:
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def format_source_for_ui(document: dict[str, Any]) -> dict[str, Any]:
    """Return a source object the frontend can label, excerpt, and link."""

    kind = document.get("kind") or "library"
    title = (
        document.get("title")
        or document.get("filename")
        or document.get("source")
    )
    source_name = (
        document.get("source")
        or document.get("filename")
        or document.get("title")
        or "Unknown"
    )
    text = str(
        document.get("text")
        or document.get("abstract")
        or ""
    ).strip()
    url = _http_url(
        document.get("url"),
        document.get("pdf_url"),
        document.get("paper_url"),
        document.get("source"),
    )

    card = {
        "kind": kind,
        "title": title,
        "source": source_name,
        "text": text,
        "abstract": document.get("abstract"),
        "url": url,
        "pdf_url": document.get("pdf_url"),
        "paper_url": document.get("paper_url"),
        "page": document.get("page"),
        "authors": document.get("authors"),
        "year": document.get("year"),
        "doi": document.get("doi"),
    }
    return {key: value for key, value in card.items() if value not in (None, "")}


def format_sources_for_ui(documents: list[Any] | None) -> list[dict[str, Any]]:
    """Convert workflow final_docs into JSON-safe source cards."""

    sources: list[dict[str, Any]] = []
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        sources.append(format_source_for_ui(document))
    return sources
