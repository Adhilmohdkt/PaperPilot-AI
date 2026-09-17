"""OpenAlex academic-paper provider for PaperPilot Phase 2."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus

from .base import AcademicPaper, AcademicSearchProvider


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAlex API configuration
# ---------------------------------------------------------------------------

OPEN_ALEX_BASE = "https://api.openalex.org/works"

# Default per-page limit for OpenAlex.
OPEN_ALEX_DEFAULT_PER_PAGE = 100

# OpenAlex does not have a strict hard cap, but we limit requests
# to avoid unnecessarily large responses.
OPEN_ALEX_MAX_PER_REQUEST = 200


# ---------------------------------------------------------------------------
# OpenAlex API request
# ---------------------------------------------------------------------------

async def _fetch_openalex_json(
    query: str,
    max_results: int,
) -> Optional[dict[str, Any]]:
    """Fetch OpenAlex API results as JSON.

    OpenAlex supports search across work titles, abstracts, and other
    indexed fields. PaperPilot uses a bounded result set.
    """

    params: list[str] = []

    # Search across OpenAlex's indexed work fields.
    params.append(
        f"search={quote_plus(query)}"
    )

    # Limit results.
    limit = min(
        max_results,
        OPEN_ALEX_MAX_PER_REQUEST,
    )

    params.append(
        f"per_page={limit}"
    )

    # Build request URL.
    url = (
        f"{OPEN_ALEX_BASE}?"
        f"{'&'.join(params)}"
    )

    loop = asyncio.get_event_loop()

    try:

        def _request():
            request = urllib_request.Request(
                url,
                headers={
                    "User-Agent": (
                        "PaperPilot-AI/1.0 "
                        "(academic research client)"
                    )
                },
            )

            with urllib_request.urlopen(
                request,
                timeout=15,
            ) as response:
                body = response.read()
                return json.loads(body)

        data = await loop.run_in_executor(
            None,
            _request,
        )

        return data

    except (HTTPError, URLError, OSError) as exc:
        logger.warning(
            f"OpenAlex API request failed: {exc}"
        )
        return None

    except json.JSONDecodeError as exc:
        logger.warning(
            f"OpenAlex response not valid JSON: {exc}"
        )
        return None

    except Exception as exc:
        logger.warning(
            f"Unexpected error fetching OpenAlex: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# arXiv identifier extraction
# ---------------------------------------------------------------------------

def _extract_arxiv_id(
    work: dict[str, Any],
) -> Optional[str]:
    """Extract an arXiv identifier from an OpenAlex work.

    OpenAlex can expose arXiv information through its locations.
    We inspect landing-page and PDF URLs for an arXiv identifier.

    Examples:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/abs/1706.03762v7
        https://arxiv.org/pdf/1706.03762
        https://arxiv.org/pdf/1706.03762v7.pdf
    """

    locations = work.get("locations") or []

    for location in locations:

        if not isinstance(location, dict):
            continue

        candidate_urls = (
            location.get("landing_page_url"),
            location.get("pdf_url"),
        )

        for candidate_url in candidate_urls:

            if not candidate_url:
                continue

            candidate_url = str(candidate_url)

            if "arxiv.org" not in candidate_url.lower():
                continue

            match = re.search(
                r"arxiv\.org/(?:abs|pdf)/([^/?#]+)",
                candidate_url,
                re.IGNORECASE,
            )

            if not match:
                continue

            arxiv_id = match.group(1)

            # Remove a possible .pdf suffix.
            arxiv_id = re.sub(
                r"\.pdf$",
                "",
                arxiv_id,
                flags=re.IGNORECASE,
            )

            if arxiv_id:
                return arxiv_id

    return None


# ---------------------------------------------------------------------------
# OpenAlex work parsing
# ---------------------------------------------------------------------------

def _parse_openalex_work(
    work: dict[str, Any],
    provider: str = "openalex",
) -> Optional[AcademicPaper]:
    """Parse a single OpenAlex work into an AcademicPaper.

    Returns None when the work does not contain a useful title.
    """

    try:

        # -------------------------------------------------------------------
        # Title
        # -------------------------------------------------------------------

        title = (
            work.get("display_name")
            or work.get("title")
        )

        if not title:
            return None

        # -------------------------------------------------------------------
        # Authors
        # -------------------------------------------------------------------

        authors: list[str] = []

        for authorship in work.get(
            "authorships",
            [],
        ):

            if not isinstance(authorship, dict):
                continue

            author = authorship.get(
                "author",
                {},
            )

            if (
                isinstance(author, dict)
                and author.get("display_name")
            ):
                authors.append(
                    author["display_name"]
                )

        # -------------------------------------------------------------------
        # Abstract
        # -------------------------------------------------------------------

        abstract = work.get("abstract")

        if not abstract:

            abstract_inverted = work.get(
                "abstract_inverted_index"
            )

            if (
                abstract_inverted
                and isinstance(
                    abstract_inverted,
                    dict,
                )
            ):
                abstract = " ".join(
                    abstract_inverted.keys()
                )

            else:
                abstract = abstract_inverted

        # -------------------------------------------------------------------
        # Publication date / year
        # -------------------------------------------------------------------

        publication_date = work.get(
            "publication_date"
        )

        year: int | None = None

        if publication_date:

            try:
                year = int(
                    publication_date[:4]
                )

            except (
                ValueError,
                TypeError,
            ):
                year = None

        # -------------------------------------------------------------------
        # Paper URL
        # -------------------------------------------------------------------

        paper_url = work.get("id")

        # -------------------------------------------------------------------
        # DOI
        # -------------------------------------------------------------------

        doi = work.get("doi")

        # -------------------------------------------------------------------
        # OpenAlex ID
        # -------------------------------------------------------------------

        openalex_id = work.get("id")

        # -------------------------------------------------------------------
        # Citation count
        # -------------------------------------------------------------------

        citation_count = work.get(
            "cited_by_count"
        )

        # -------------------------------------------------------------------
        # arXiv ID
        # -------------------------------------------------------------------

        arxiv_id = _extract_arxiv_id(
            work
        )

        # -------------------------------------------------------------------
        # PDF URL
        # -------------------------------------------------------------------

        pdf_url = None

        if arxiv_id:

            # Prefer the canonical arXiv PDF whenever OpenAlex exposes
            # an arXiv version of the paper.
            pdf_url = (
                f"https://arxiv.org/pdf/{arxiv_id}"
            )

        else:

            # Otherwise use OpenAlex's best available open-access PDF.
            oa_location = work.get(
                "best_oa_location"
            )

            if (
                isinstance(
                    oa_location,
                    dict,
                )
            ):
                pdf_url = oa_location.get(
                    "pdf_url"
                )

        # -------------------------------------------------------------------
        # Build normalized AcademicPaper
        # -------------------------------------------------------------------

        return AcademicPaper(
            title=title,
            authors=authors,
            abstract=(
                abstract
                if abstract
                else None
            ),
            publication_date=publication_date,
            year=year,
            provider=provider,
            paper_url=paper_url,
            pdf_url=pdf_url,
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            citation_count=citation_count,
        )

    except Exception as exc:

        logger.warning(
            f"Failed to parse OpenAlex work: {exc}"
        )

        return None


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

async def search_openalex(
    query: str,
    max_results: int = OPEN_ALEX_DEFAULT_PER_PAGE,
) -> List[AcademicPaper]:
    """Search OpenAlex for papers matching a query.

    Args:
        query:
            Academic search query.

        max_results:
            Maximum number of papers to return.

    Returns:
        A list of normalized AcademicPaper objects.
    """

    data = await _fetch_openalex_json(
        query,
        max_results,
    )

    if data is None:
        return []

    results = data.get(
        "results",
        [],
    )

    papers: List[AcademicPaper] = []

    for work in results[:max_results]:

        paper = _parse_openalex_work(
            work,
            provider="openalex",
        )

        if paper is not None:
            papers.append(paper)

    return papers


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class OpenAlexProvider(
    AcademicSearchProvider
):
    """Concrete provider for OpenAlex."""

    def __init__(
        self,
        max_results: int = OPEN_ALEX_DEFAULT_PER_PAGE,
    ):
        self.max_results = max_results

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> List[AcademicPaper]:
        """Search OpenAlex using the configured query strategy."""

        effective_max = (
            max_results
            if max_results is not None
            else self.max_results
        )

        return await search_openalex(
            query,
            effective_max,
        )