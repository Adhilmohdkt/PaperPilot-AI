"""ArXiv academic-paper provider for PaperPilot Phase 2."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional
import arxiv

from .base import AcademicPaper, AcademicSearchProvider

logger = logging.getLogger(__name__)

# The arXiv API supports up to 30,000 results, but PaperPilot intentionally
# uses a much smaller bounded result set.
ARXIV_API_MAX = 30000


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _build_search_query(query: str) -> str:
    """Build an arXiv search query.

    Explicit paper-title-style queries are searched against the title field
    to improve precision. Broader research queries continue to use the
    existing all-field search behavior.
    """
    normalized = " ".join(query.split()).strip()

    if not normalized:
        return "all:*"

    title_like_patterns = (
        "attention is all you need",
        "bert",
        "gpt",
        "resnet",
        "llama",
        "t5",
        "word2vec",
        "image is worth",
        "you only look once",
    )

    lowered = normalized.lower()

    if lowered in title_like_patterns:
        return f'ti:"{normalized}"'

    title_prefixes = (
        "find the paper ",
        "find paper ",
        "paper titled ",
        "find the paper titled ",
        "the paper ",
    )

    for prefix in title_prefixes:
        if lowered.startswith(prefix):
            title = normalized[len(prefix):].strip()

            if title:
                return f'ti:"{title}"'

    return normalized


# ---------------------------------------------------------------------------
# arXiv Client Request
# ---------------------------------------------------------------------------

def _fetch_arxiv_sync(
    query: str,
    max_results: int,
) -> List[AcademicPaper]:
    """Fetch papers synchronously via the arxiv client."""
    search_query = _build_search_query(query)
    effective_max = min(max_results, ARXIV_API_MAX)

    try:
        client = arxiv.Client(page_size=effective_max, delay_seconds=3.0, num_retries=3)
        search = arxiv.Search(
            query=search_query,
            max_results=effective_max,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: List[AcademicPaper] = []
        for result in client.results(search):
            title = " ".join(result.title.split()) if result.title else None
            authors = [author.name for author in result.authors] if result.authors else []
            abstract = result.summary.strip() if result.summary else None
            year = result.published.year if result.published else None
            pub_date = result.published.strftime("%Y-%m-%d") if result.published else None

            arxiv_id = result.get_short_id() if hasattr(result, "get_short_id") else None
            if not arxiv_id and result.entry_id:
                match = re.search(r"arxiv\.org/abs/([^/?#]+)", result.entry_id)
                if match:
                    arxiv_id = match.group(1)

            pdf_url = result.pdf_url
            paper_url = result.entry_id
            doi = result.doi

            if title:
                papers.append(
                    AcademicPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        publication_date=pub_date,
                        year=year,
                        provider="arxiv",
                        paper_url=paper_url,
                        pdf_url=pdf_url,
                        doi=doi,
                        arxiv_id=arxiv_id,
                    )
                )

        return papers

    except Exception as exc:
        logger.warning(f"ArXiv client search failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

async def search_arxiv(
    query: str,
    max_results: int = 50,
) -> List[AcademicPaper]:
    """Search arXiv for papers matching the query.

    Args:
        query: The search query string.
        max_results: Maximum number of papers to return.

    Returns:
        A list of AcademicPaper objects sorted by arXiv relevance.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_arxiv_sync, query, max_results)


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class ArxivProvider(AcademicSearchProvider):
    """Concrete provider for arXiv API searches."""

    def __init__(self, max_results: int = 50):
        self.max_results = max_results

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> List[AcademicPaper]:
        """Search arXiv using the configured query strategy."""
        effective_max = (
            max_results
            if max_results is not None
            else self.max_results
        )
        return await search_arxiv(
            query,
            effective_max,
        )