"""OpenAlex academic-paper provider for PaperPilot Phase 2."""
from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote_plus
from typing import Any

from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .base import AcademicPaper, AcademicSearchProvider

logger = logging.getLogger(__name__)

# OpenAlex API base endpoint
OPEN_ALEX_BASE = "https://api.openalex.org/works"
# Default per-page limit for OpenAlex
OPEN_ALEX_DEFAULT_PER_PAGE = 100
# OpenAlex does not have a strict hard cap but we limit to avoid huge requests
OPEN_ALEX_MAX_PER_REQUEST = 200


async def _fetch_openalex_json(query: str, max_results: int) -> Optional[dict[str, Any]]:
    """Fetch OpenALex API results as JSON.

    OpenAlex supports filtering by concepts, search, and other params.
    We use a simple search query with limit.
    """
    params: list[str] = []
    # Search across display_name, title, abstract, etc.
    params.append(f"search={quote_plus(query)}")
    # Limit results
    limit = min(max_results, OPEN_ALEX_MAX_PER_REQUEST)
    params.append(f"per_page={limit}")
    # Include "concepts" which may provide citation counts
    params.append("filter=")
    # Build URL
    url = f"{OPEN_ALEX_BASE}?{'&'.join(params)}"

    loop = asyncio.get_event_loop()
    try:
        def _request():
            with urllib_request.urlopen(url, timeout=15) as resp:
                body = resp.read()
                return json.loads(body)

        data = await loop.run_in_executor(None, _request)
        return data
    except (HTTPError, URLError, OSError) as e:
        logger.warning(f"OpenAlex API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"OpenAlex response not valid JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching OpenAlex: {e}")
        return None


def _parse_openalex_work(work: dict[str, Any], provider: str = "openalex") -> AcademicPaper | None:
    """Parse a single OpenAlex work record into an AcademicPaper.

    Returns None if the record has no useful title/abstract.
    """
    try:
        title = work.get("display_name") or work.get("title")
        if not title:
            return None

        # Authors
        authors: list[str] = []
        for author in work.get("authorships", []):
            author_name = author.get("author", {})
            if author_name and author_name.get("display_name"):
                authors.append(author_name["display_name"])

        # Abstract
        abstract = work.get("abstract")
        if not abstract:
            # Convert inverted index to string if needed
            abstract_inverted = work.get("abstract_inverted_index")
            if abstract_inverted and isinstance(abstract_inverted, dict):
                abstract = " ".join(abstract_inverted.keys())
            else:
                abstract = abstract_inverted

        # Publication date / year
        publication_date = work.get("publication_date")
        year: int | None = None
        if publication_date:
            # OpenAlex uses YYYY-MM-DD format
            try:
                year = int(publication_date[:4])
            except (ValueError, TypeError):
                year = None

        # Paper URL (OpenAlex always provides a PDF URL if available)
        paper_url = work.get("id")

        # PDF URL (best_oa_location gives open-access PDF if available)
        pdf_url = None
        oa_location = work.get("best_oa_location")
        if oa_location:
            pdf_url = oa_location.get("pdf_url")

        # DOI
        doi = work.get("doi")

        # OpenAlex ID
        openalex_id = work.get("id")

        # Citation count
        citation_count = work.get("cited_by_count")

        return AcademicPaper(
            title=title,
            authors=authors,
            abstract=abstract if abstract else None,
            publication_date=publication_date,
            year=year,
            provider=provider,
            paper_url=paper_url,
            pdf_url=pdf_url,
            doi=doi,
            openalex_id=openalex_id,
            citation_count=citation_count,
        )
    except Exception as e:
        logger.warning(f"Failed to parse OpenAlex work: {e}")
        return None


async def search_openalex(query: str, max_results: int = 100) -> List[AcademicPaper]:
    """Search OpenAlex for papers matching the query.

    Args:
        query: The search query string.
        max_results: Maximum number of papers to return.

    Returns:
        A list of AcademicPaper objects.
    """
    data = await _fetch_openalex_json(query, max_results)
    if data is None:
        return []

    results = data.get("results", [])
    papers: List[AcademicPaper] = []
    for work in results[:max_results]:
        paper = _parse_openalex_work(work, provider="openalex")
        if paper is not None:
            papers.append(paper)

    return papers


class OpenAlexProvider(AcademicSearchProvider):
    """Concrete provider for OpenAPI searches."""

    def __init__(self, max_results: int = 100):
        self.max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> List[AcademicPaper]:
        """See base class."""
        effective_max = max_results or self.max_results
        return await search_openalex(query, effective_max)