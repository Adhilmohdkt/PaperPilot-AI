"""Runtime configuration for the PaperPilot backend.

Only non-secret defaults live here. Deployments can override values
through environment variables. API keys are never given fallback values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Local RAG / Weaviate
    # ------------------------------------------------------------------
    collection_name: str = os.getenv(
        "WEAVIATE_COLLECTION",
        "PaperChunk",
    )

    hybrid_alpha: float = float(
        os.getenv("HYBRID_ALPHA", "0.5")
    )

    retrieval_k: int = int(
        os.getenv("RETRIEVAL_K", "15")
    )

    rerank_top_n: int = int(
        os.getenv("RERANK_TOP_N", "5")
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "gemini-embedding-2",
    )

    cohere_rerank_model: str = os.getenv(
        "COHERE_RERANK_MODEL",
        "rerank-english-v3.0",
    )

    # ------------------------------------------------------------------
    # Local document ingestion
    # ------------------------------------------------------------------
    max_library_documents: int = int(
        os.getenv("MAX_LIBRARY_DOCUMENTS", "10")
    )

    max_pdf_pages: int = int(
        os.getenv("MAX_PDF_PAGES", "30")
    )

    max_chunks_per_document: int = int(
        os.getenv("MAX_CHUNKS_PER_DOCUMENT", "80")
    )

    max_chars_per_page: int = int(
        os.getenv("MAX_CHARS_PER_PAGE", "12000")
    )

    # ------------------------------------------------------------------
    # Local RAG relevance
    # ------------------------------------------------------------------
    rag_relevance_threshold: float = float(
        os.getenv("RAG_RELEVANCE_THRESHOLD", "0.3")
    )

    min_relevant_docs: int = int(
        os.getenv("MIN_RELEVANT_DOCS", "2")
    )

        # ------------------------------------------------------------------
    # LangSmith
    # ------------------------------------------------------------------
    langsmith_tracing: bool = (
        os.getenv("LANGSMITH_TRACING", "false").lower()
        == "true"
    )

    langsmith_project: str = os.getenv(
        "LANGSMITH_PROJECT",
        "paperpilot",
    )

    langsmith_api_key: str | None = os.getenv(
        "LANGSMITH_API_KEY"
    )

    # ------------------------------------------------------------------
    # Academic research
    # ------------------------------------------------------------------
    academic_search_max_results: int = int(
        os.getenv("ACADEMIC_SEARCH_MAX_RESULTS", "10")
    )

    academic_top_k: int = int(
        os.getenv("ACADEMIC_TOP_K", "5")
    )

    academic_pdf_max: int = int(
        os.getenv("ACADEMIC_PDF_MAX", "2")
    )

    academic_pdf_timeout: int = int(
        os.getenv("ACADEMIC_PDF_TIMEOUT", "30")
    )

    academic_pdf_max_size: int = int(
        os.getenv(
            "ACADEMIC_PDF_MAX_SIZE",
            str(50 * 1024 * 1024),
        )
    )

    academic_api_timeout: int = int(
        os.getenv("ACADEMIC_API_TIMEOUT", "15")
    )


settings = Settings()