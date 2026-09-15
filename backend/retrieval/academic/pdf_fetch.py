"""PDF retrieval and text extraction for externally retrieved academic papers.

Reuses the project's existing PyMuPDF/LangChain approach. External paper chunks
are kept separate from the permanent local document library.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp

from langchain_community.document_loaders import PyMuPDFLoader

from ...ingestion.loader import load_pdf
from ...ingestion.chunker import chunk_documents
from .base import AcademicPaper

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

# Default: attempt PDF fetch for this many top-ranked papers
DEFAULT_PDF_MAX = 2

# HTTP timeout for PDF download (seconds)
PDF_DOWNLOAD_TIMEOUT: int = 30

# Max PDF size to download (bytes) — prevent huge downloads
PDF_MAX_SIZE: int = 50 * 1024 * 1024  # 50 MB

# Chunk settings — reuse the project's existing values
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 150


# ── Helpers ───────────────────────────────────────────────────────────────

async def _download_pdf(pdf_url: str, timeout: int = PDF_DOWNLOAD_TIMEOUT) -> Optional[bytes]:
    """Download a PDF from a URL asynchronously.

    Returns the raw PDF bytes, or None on failure.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(pdf_url, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"PDF download returned status {resp.status}: {pdf_url}")
                    return None
                size = resp.content_length or 0
                if size > PDF_MAX_SIZE:
                    logger.warning(f"PDF exceeds size limit ({size} > {PDF_MAX_SIZE}): {pdf_url}")
                    return None
                bytes_data = await resp.read()
                return bytes_data
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
            logger.warning(f"PDF download failed ({type(e).__name__}): {pdf_url} — {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error downloading PDF: {type(e).__name__}: {pdf_url} — {e}")
            return None


# ── Core: fetch + extract ────────────────────────────────────────────────

async def fetch_paper_text(
    paper: AcademicPaper,
    max_chunks: int = 8,
) -> Tuple[Optional[list], Optional[str]]:
    """Try to retrieve and extract text from a paper's PDF.

    Returns a tuple of (chunks, error_message).
    - chunks: list of LangChain Document objects (or None if failed)
    - error_message: human-readable error (or None if success)
    """
    if not paper.pdf_url:
        return None, "No PDF URL available for this paper."

    pdf_bytes = await _download_pdf(paper.pdf_url)
    if pdf_bytes is None:
        return None, "Failed to download PDF."

    # Write to a temporary file — we do NOT persist to the permanent library
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # Reuse the project's existing PyMuPDFLoader
        # (this function's caller is responsible for cleanup)
        raw_docs = load_pdf(tmp_path)
        if not raw_docs:
            return None, "PDF loaded but contained no text."

        # Chunk using the project's existing splitter
        chunks = chunk_documents(raw_docs)
        # Limit chunks per paper
        chunks = chunks[:max_chunks]

        # Strip out the temporary file — do NOT add to permanent library
        # (os.unlink(tmp_path) would be called by the caller if desired)

        # Build a readable context string from chunks for generation
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            src = chunk.metadata.get("page", "unknown")
            ctx = f"[Paper chunk {i}, page {src}] {chunk.page_content[:500]}"
            context_parts.append(ctx)

        context_str = "\n\n".join(context_parts) if context_parts else None
        return chunks, context_str

    except Exception as e:
        logger.warning(f"PDF text extraction failed: {type(e).__name__}: {e}")
        return None, f"PDF parsing error: {e}"
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Convenience: extract just the context string (no chunk objects) ──────

async def fetch_paper_context(
    paper: AcademicPaper,
    max_chunks: int = 8,
) -> Optional[str]:
    """Download and extract a human-readable context string from a paper's PDF.

    Returns a string like:
        [Paper chunk 1, page 3] Introduction to RAG techniques...
        [Paper chunk 2, page 7] ...backpropagation details...
    """
    _, context_str = await fetch_paper_text(paper, max_chunks=max_chunks)
    return context_str