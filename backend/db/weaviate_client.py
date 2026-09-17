"""Weaviate client and hybrid-search utilities for PaperPilot."""

from __future__ import annotations

import os
from typing import Any

import weaviate
from weaviate.classes.query import Filter

from config import settings


def get_client() -> Any:
    host = os.getenv("WEAVIATE_HOST", "localhost")

    if host in {"localhost", "127.0.0.1"}:
        return weaviate.connect_to_local()

    return weaviate.connect_to_custom(
        http_host=host,
        http_port=8080,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
    )

def search(
    query_text: str,
    query_vector: list[float],
    limit: int = 15,
    source_filter: str | None = None,
):
    """Run hybrid BM25 + vector search against the local collection.

    Args:
        query_text: Text used for BM25/keyword retrieval.
        query_vector: Embedding vector for vector retrieval.
        limit: Maximum number of candidate chunks.
        source_filter: Optional source/document filter.

    Returns:
        Weaviate query response.
    """
    if not query_text or not query_text.strip():
        return None

    if not query_vector:
        return None

    if limit <= 0:
        return None

    client = get_client()

    try:
        collection = client.collections.get(settings.collection_name)

        filters = None

        if source_filter:
            filters = Filter.by_property("source").equal(source_filter)

        return collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=settings.hybrid_alpha,
            filters=filters,
            limit=limit,
        )

    finally:
        client.close()