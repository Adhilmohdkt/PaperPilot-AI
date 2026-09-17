"""PDF loading utilities for PaperPilot."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(path: str) -> list[Document]:
    """Load a PDF into LangChain Documents while preserving page metadata."""
    loader = PyMuPDFLoader(path)
    return loader.load()