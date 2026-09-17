"""Local hybrid retrieval and reranking for PaperPilot."""

from __future__ import annotations

import logging
import os
from typing import Any

import cohere

from config import settings
from db.weaviate_client import search
from embeddings.embedder import embed_query

logger = logging.getLogger(__name__)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def _create_cohere_client() -> cohere.Client | None:
    """Create the Cohere client when an API key is configured."""
    if not COHERE_API_KEY:
        logger.warning(
            "COHERE_API_KEY is not configured; "
            "local retrieval will use Weaviate results without reranking."
        )
        return None

    return cohere.Client(api_key=COHERE_API_KEY)


_cohere_client = _create_cohere_client()


def retrieve(
    query: str,
    source_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant local document chunks.

    Pipeline:
        Query
          ↓
        Gemini embedding
          ↓
        Weaviate hybrid search (BM25 + vector)
          ↓
        Candidate chunks
          ↓
        Cohere reranking
          ↓
        Top-k chunks

    Args:
        query: User's retrieval query.
        source_filter: Optional source/document filter.

    Returns:
        A list of retrieved document chunks containing their text,
        source, and available metadata.
    """
    if not query or not query.strip():
        return []

    # ------------------------------------------------------------------
    # 1. Generate query embedding
    # ------------------------------------------------------------------
    try:
        query_vector = embed_query(query)
    except Exception:
        logger.exception("Failed to generate query embedding.")
        return []

    # ------------------------------------------------------------------
    # 2. Hybrid retrieval from Weaviate
    # ------------------------------------------------------------------
    try:
        results = search(
            query_text=query,
            query_vector=query_vector,
            limit=settings.retrieval_k,
            source_filter=source_filter,
        )
    except Exception:
        logger.exception("Weaviate hybrid search failed.")
        return []

    if results is None:
        return []

    chunks: list[dict[str, Any]] = []

    for obj in getattr(results, "objects", []):
        properties = getattr(obj, "properties", {}) or {}

        text = properties.get("text", "")

        if not text or not str(text).strip():
            continue

        chunk: dict[str, Any] = {
            "text": str(text),
            "source": properties.get("source", "Unknown"),
        }

        # Preserve useful metadata when available.
        for field in (
            "page",
            "page_number",
            "chunk_id",
            "document_id",
            "filename",
        ):
            if field in properties:
                chunk[field] = properties[field]

        chunks.append(chunk)

    if not chunks:
        logger.info(
            "No local retrieval results found for query: %s",
            query,
        )
        return []

    # ------------------------------------------------------------------
    # 3. Cohere reranking
    # ------------------------------------------------------------------
    if _cohere_client is None:
        return chunks[: settings.rerank_top_n]

    try:
        documents = [chunk["text"] for chunk in chunks]

        response = _cohere_client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=min(
                settings.rerank_top_n,
                len(documents),
            ),
        )

        reranked_chunks: list[dict[str, Any]] = []

        for result in response.results:
            original_chunk = chunks[result.index].copy()

            # Preserve Cohere relevance score for downstream
            # relevance checks and evaluation.
            if hasattr(result, "relevance_score"):
                original_chunk["rerank_score"] = (
                    result.relevance_score
                )

            reranked_chunks.append(original_chunk)

        return reranked_chunks

    except Exception:
        logger.exception(
            "Cohere reranking failed; "
            "falling back to Weaviate hybrid results."
        )

        return chunks[: settings.rerank_top_n]