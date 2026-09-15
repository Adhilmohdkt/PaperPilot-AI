"""Gemini embeddings used for both indexing and hybrid-query vectors.

Uses GoogleGenerativeAIEmbeddings from langchain-google-genai with the
gemini-embedding-2-preview model, keyed by the GEMINI_API_KEY environment variable.
"""

import os
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings


@lru_cache(maxsize=1)
def _embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", api_key=os.getenv("GEMINI_API_KEY")
    )


def get_embedding(text: str) -> list:
    return _embedder().embed_query(text)


def embed_query(query: str) -> list:
    return get_embedding(query)