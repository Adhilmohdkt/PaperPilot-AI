"""Academic search provider interface and shared data types."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional

from urllib.parse import quote_plus


@dataclass
class AcademicPaper:
    """Normalized representation of an academic paper from any provider."""
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    publication_date: str | None = None  # ISO format, e.g. "2024-01-15"
    year: int | None = None
    provider: str | None = None  # "arxiv" | "openalex" | etc.
    paper_url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    citation_count: int | None = None
    relevance_score: float | None = None  # assigned by the ranker


class AcademicSearchProvider(abc.ABC):
    """Abstract base class for academic paper search providers."""

    @abc.abstractmethod
    async def search(self, query: str, max_results: int) -> List[AcademicPaper]:
        """Search for academic papers matching the query.

        Args:
            query: The user's research query.
            max_results: Maximum number of papers to return.

        Returns:
            A list of AcademicPaper objects, potentially with empty relevance scores.
        """
        ...  # pragma: no cover