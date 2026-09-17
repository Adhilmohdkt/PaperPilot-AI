"""Gemini embeddings used for PaperPilot local retrieval.

The same embedding model must be used for both document indexing
and query embedding.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings


@lru_cache(maxsize=1)
def _embedder() -> GoogleGenerativeAIEmbeddings:
    """Create and cache the Gemini embedding model."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
    )


def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a text string."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    return _embedder().embed_query(text)


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a retrieval query."""
    return get_embedding(query)

def get_embedder() -> GoogleGenerativeAIEmbeddings:
    """Return the cached Gemini embedding model for LangChain retrieval."""
    return _embedder()