"""Idempotent PDF indexing helpers backed by persistent Weaviate storage."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from weaviate.classes.query import Filter

from config import settings
from db.schema import ensure_schema
from db.weaviate_client import get_client
from embeddings.embedder import get_embedding
from ingestion.chunker import chunk_documents
from ingestion.loader import load_pdf


def document_hash(path: str | Path) -> str:
    """Return a stable content hash without loading the PDF into the LLM stack."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_indexed(content_hash: str) -> bool:
    client = get_client()
    ensure_schema(client)
    collection = client.collections.get(settings.collection_name)
    response = collection.query.fetch_objects(
        filters=Filter.by_property("content_hash").equal(content_hash),
        limit=1,
        return_properties=["content_hash"],
    )
    return bool(response.objects)


def _prepare_documents(path: str | Path, source: str, content_hash: str) -> list[Document]:
    pages = load_pdf(str(path))
    if len(pages) > settings.max_pdf_pages:
        raise ValueError(f"This PDF has {len(pages)} pages; the limit is {settings.max_pdf_pages}.")

    prepared = []
    for page in pages:
        text = page.page_content[: settings.max_chars_per_page]
        if text.strip():
            prepared.append(Document(
                page_content=text,
                metadata={
                    "source": source,
                    "content_hash": content_hash,
                    "page": int(page.metadata.get("page", 0)) + 1,
                },
            ))
    return prepared


def index_pdf(path: str | Path, source: str | None = None) -> dict:
    """Index a PDF once; unchanged content returns without embedding again."""
    path = Path(path)
    source = source or path.name
    content_hash = document_hash(path)
    if is_indexed(content_hash):
        return {"status": "already_indexed", "filename": source, "content_hash": content_hash, "chunks_inserted": 0}

    documents = _prepare_documents(path, source, content_hash)
    chunks = chunk_documents(documents)[: settings.max_chunks_per_document]
    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    client = get_client()
    ensure_schema(client)
    collection = client.collections.get(settings.collection_name)
    with collection.batch.dynamic() as batch:
        for index, chunk in enumerate(chunks):
            batch.add_object(
                properties={
                    "text": chunk.page_content,
                    "source": source,
                    "content_hash": content_hash,
                    "chunk_id": f"{content_hash}:{index}",
                    "page": int(chunk.metadata.get("page", 1)),
                },
                vector=get_embedding(chunk.page_content),
            )
    if collection.batch.failed_objects:
        raise RuntimeError(f"Weaviate rejected {len(collection.batch.failed_objects)} chunk(s).")
    return {"status": "success", "filename": source, "content_hash": content_hash, "chunks_inserted": len(chunks)}
