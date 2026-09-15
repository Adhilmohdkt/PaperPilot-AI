"""Academic research-paper retrieval providers for PaperPilot."""
from .base import AcademicPaper, AcademicSearchProvider
from .arxiv_provider import ArxivProvider
from .openalex_provider import OpenAlexProvider

__all__ = [
    "AcademicPaper",
    "AcademicSearchProvider",
    "ArxivProvider",
    "OpenAlexProvider",
]
