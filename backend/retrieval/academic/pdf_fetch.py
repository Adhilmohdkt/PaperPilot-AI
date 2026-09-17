"""Temporary PDF retrieval and question-aware text extraction for academic papers.

External academic PDFs are downloaded only for the current research
workflow. They are never added to the permanent local Weaviate library.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional

import aiohttp
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from embeddings.embedder import get_embedder

from .base import AcademicPaper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PDF_MAX = 2

PDF_DOWNLOAD_TIMEOUT = 30
PDF_MAX_SIZE = 50 * 1024 * 1024  # 50 MB

MAX_CHUNKS_PER_PAPER = 8


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------


async def _download_pdf(
    pdf_url: str,
    timeout: int = PDF_DOWNLOAD_TIMEOUT,
) -> Optional[bytes]:
    """Download a PDF asynchronously with timeout and size protection."""

    if not pdf_url:
        return None

    client_timeout = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(
            timeout=client_timeout
        ) as session:

            async with session.get(
                pdf_url,
                headers={
                    "User-Agent": "PaperPilot-AI/1.0",
                    "Accept": "application/pdf",
                },
            ) as response:

                if response.status != 200:
                    logger.warning(
                        "PDF download returned HTTP %s: %s",
                        response.status,
                        pdf_url,
                    )
                    return None

                content_length = response.content_length

                if (
                    content_length is not None
                    and content_length > PDF_MAX_SIZE
                ):
                    logger.warning(
                        "PDF exceeds size limit: %s bytes",
                        content_length,
                    )
                    return None

                pdf_bytes = await response.read()

                if len(pdf_bytes) > PDF_MAX_SIZE:
                    logger.warning(
                        "Downloaded PDF exceeds size limit: %s bytes",
                        len(pdf_bytes),
                    )
                    return None

                if not pdf_bytes:
                    logger.warning(
                        "Downloaded PDF is empty: %s",
                        pdf_url,
                    )
                    return None

                return pdf_bytes

    except asyncio.TimeoutError:
        logger.warning(
            "PDF download timed out: %s",
            pdf_url,
        )
        return None

    except aiohttp.ClientError as exc:
        logger.warning(
            "PDF download failed: %s — %s",
            pdf_url,
            exc,
        )
        return None

    except OSError as exc:
        logger.warning(
            "PDF download filesystem/network error: %s — %s",
            pdf_url,
            exc,
        )
        return None

    except Exception as exc:
        logger.warning(
            "Unexpected PDF download error: %s — %s",
            pdf_url,
            exc,
        )
        return None


async def download_pdf(
    pdf_url: str,
    timeout: int = PDF_DOWNLOAD_TIMEOUT,
) -> Optional[bytes]:
    """Download a public academic PDF for temporary research use."""

    return await _download_pdf(
        pdf_url,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Question-aware retrieval
# ---------------------------------------------------------------------------


def _retrieve_relevant_chunks(
    chunks: list[Document],
    query: str,
    max_chunks: int,
) -> list[Document]:
    """Retrieve chunks relevant to the current research question.

    Uses LangChain's in-memory vector store with the same Gemini embedding
    model used elsewhere in PaperPilot.

    The vector store exists only for this function call and is discarded
    afterwards. Nothing is persisted to Weaviate.
    """

    if not chunks:
        return []

    if not query.strip():
        return chunks[:max_chunks]

    try:
        vector_store = InMemoryVectorStore(
            embedding=get_embedder(),
        )

        vector_store.add_documents(chunks)

        retrieved = vector_store.similarity_search(
            query,
            k=min(max_chunks, len(chunks)),
        )

        return retrieved

    except Exception as exc:
        logger.warning(
            "Question-aware PDF retrieval failed: %s — %s",
            type(exc).__name__,
            exc,
        )

        # Safe fallback: preserve the existing extraction behavior
        # rather than failing the entire academic workflow.
        return chunks[:max_chunks]


# ---------------------------------------------------------------------------
# Fetch and extract paper text
# ---------------------------------------------------------------------------


async def fetch_paper_text(
    paper: AcademicPaper,
    query: str = "",
    max_chunks: int = MAX_CHUNKS_PER_PAPER,
) -> tuple[Optional[list[Document]], Optional[str]]:
    """Download a paper PDF and retrieve temporary relevant LangChain Documents.

    Args:
        paper:
            Academic paper containing a public PDF URL.

        query:
            The user's current research question. Used to retrieve the
            most relevant chunks from the paper.

        max_chunks:
            Maximum number of relevant chunks to return.

    Returns:
        (relevant_chunks, context_string)

    The downloaded PDF and temporary vector store are discarded after
    extraction/retrieval.
    """

    if not paper.pdf_url:
        return None, "No PDF URL available for this paper."

    pdf_bytes = await _download_pdf(paper.pdf_url)

    if pdf_bytes is None:
        return None, "Failed to download PDF."

    tmp_path: Optional[str] = None

    try:
        # --------------------------------------------------------------
        # Temporary file only.
        # This PDF is NOT added to the permanent document library.
        # --------------------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:

            temporary_file.write(pdf_bytes)
            tmp_path = temporary_file.name

        # --------------------------------------------------------------
        # Reuse the project's LangChain/PyMuPDF ingestion pipeline.
        # --------------------------------------------------------------

        raw_documents = load_pdf(tmp_path)

        if not raw_documents:
            return None, "PDF loaded but contained no text."

        chunks = chunk_documents(raw_documents)

        if not chunks:
            return None, "PDF was loaded but produced no chunks."

        # --------------------------------------------------------------
        # Question-aware retrieval.
        # --------------------------------------------------------------

        relevant_chunks = _retrieve_relevant_chunks(
            chunks=chunks,
            query=query,
            max_chunks=max_chunks,
        )

        if not relevant_chunks:
            return None, "No relevant PDF content was retrieved."

        # --------------------------------------------------------------
        # Build temporary research context.
        # --------------------------------------------------------------

        context_parts: list[str] = []

        for index, chunk in enumerate(
            relevant_chunks,
            start=1,
        ):
            page = chunk.metadata.get(
                "page",
                "unknown",
            )

            text = chunk.page_content.strip()

            if not text:
                continue

            context_parts.append(
                f"[Paper chunk {index}, page {page}] {text}"
            )

        context = "\n\n".join(context_parts)

        if not context:
            return None, "PDF contained no usable text."

        return relevant_chunks, context

    except Exception as exc:
        logger.warning(
            "PDF text extraction failed: %s — %s",
            type(exc).__name__,
            exc,
        )

        return None, f"PDF parsing error: {exc}"

    finally:
        # --------------------------------------------------------------
        # Always remove the temporary PDF.
        # --------------------------------------------------------------

        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


async def fetch_paper_context(
    paper: AcademicPaper,
    query: str = "",
    max_chunks: int = MAX_CHUNKS_PER_PAPER,
) -> Optional[str]:
    """Return temporary question-aware context from an academic PDF."""

    _, context = await fetch_paper_text(
        paper,
        query=query,
        max_chunks=max_chunks,
    )

    return context