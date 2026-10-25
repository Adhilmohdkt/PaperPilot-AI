"""Resilient embeddings used for PaperPilot local retrieval and PDF extraction.

Supports Google Gemini embeddings with automatic local fallback to
HuggingFace sentence-transformers when Gemini is unavailable or rate-limited.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class ResilientEmbeddings(Embeddings):
    """Wrapper that tries Gemini embeddings with fallback to local HuggingFace."""

    def __init__(self):
        self._gemini = None
        self._local = None
        self._use_local_only = False

        from config import settings
        if settings.embedding_model in {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}:
            self._use_local_only = True
            return

        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._gemini = GoogleGenerativeAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=gemini_key,
                )
            except Exception as e:
                logger.warning(f"Could not initialize Gemini embeddings: {e}")
                self._use_local_only = True
        else:
            self._use_local_only = True

    def _get_local(self) -> Embeddings:
        if self._local is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._local = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception:
                from sentence_transformers import SentenceTransformer

                class _STWrapper(Embeddings):
                    def __init__(self):
                        self.model = SentenceTransformer("all-MiniLM-L6-v2")

                    def embed_documents(self, texts: list[str]) -> list[list[float]]:
                        return [self.model.encode(t).tolist() for t in texts]

                    def embed_query(self, text: str) -> list[float]:
                        return self.model.encode(text).tolist()

                self._local = _STWrapper()
        return self._local

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._use_local_only and self._gemini:
            try:
                return self._gemini.embed_documents(texts)
            except Exception as e:
                logger.warning(f"Gemini embed_documents failed ({e}); falling back to local embeddings.")
                self._use_local_only = True

        return self._get_local().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not self._use_local_only and self._gemini:
            try:
                return self._gemini.embed_query(text)
            except Exception as e:
                logger.warning(f"Gemini embed_query failed ({e}); falling back to local embeddings.")
                self._use_local_only = True

        return self._get_local().embed_query(text)


@lru_cache(maxsize=1)
def _embedder() -> Embeddings:
    """Create and cache the resilient embedding model."""
    return ResilientEmbeddings()


def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a text string."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    return _embedder().embed_query(text)


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a retrieval query."""
    return get_embedding(query)


def get_embedder() -> Embeddings:
    """Return the cached embedding model for LangChain retrieval."""
    return _embedder()