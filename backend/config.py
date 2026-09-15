"""Runtime configuration for the PaperPilot backend.

Only non-secret defaults live here. Deployments can override every value with
environment variables; API keys are never given fallback values.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    collection_name: str = os.getenv("WEAVIATE_COLLECTION", "PaperChunk")
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "15"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    cohere_rerank_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
    max_library_documents: int = int(os.getenv("MAX_LIBRARY_DOCUMENTS", "10"))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "30"))
    max_chunks_per_document: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "80"))
    max_chars_per_page: int = int(os.getenv("MAX_CHARS_PER_PAGE", "12000"))

    # Relevance threshold for local RAG (0.0-1.0)
    # Based on Cohere rerank scores: if top reranked docs have scores below this,
    # trigger web fallback. Default 0.3 is conservative.
    rag_relevance_threshold: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.3"))
    min_relevant_docs: int = int(os.getenv("MIN_RELEVANT_DOCS", "2"))

    # LangSmith observability
    langsmith_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_project: str = os.getenv("LANGCHAIN_PROJECT", "paperpilot")
    langsmith_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")

    # Academic search (Phase 2)
    # Max results requested per provider (arXiv, OpenAlex)
    academic_search_max_results: int = int(os.getenv("ACADEMIC_SEARCH_MAX_RESULTS", "10"))
    # Final number of papers kept after ranking/filtering
    academic_top_k: int = int(os.getenv("ACADEMIC_TOP_K", "5"))
    # How many top-ranked papers to attempt PDF retrieval for
    academic_pdf_max: int = int(os.getenv("ACADEMIC_PDF_MAX", "2"))
    # PDF download timeout in seconds
    academic_pdf_timeout: int = int(os.getenv("ACADEMIC_PDF_TIMEOUT", "30"))
    # Max PDF size in bytes (50 MB)
    academic_pdf_max_size: int = int(os.getenv("ACADEMIC_PDF_MAX_SIZE", str(50 * 1024 * 1024)))
    # HTTP timeout for API calls in seconds
    academic_api_timeout: int = int(os.getenv("ACADEMIC_API_TIMEOUT", "15"))


settings = Settings()
