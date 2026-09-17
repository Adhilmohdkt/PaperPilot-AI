"""ArXiv academic-paper provider for PaperPilot Phase 2."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from xml.etree import ElementTree

from .base import AcademicPaper, AcademicSearchProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ArXiv API configuration
# ---------------------------------------------------------------------------

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# The arXiv API supports up to 30,000 results, but PaperPilot intentionally
# uses a much smaller bounded result set.
ARXIV_API_MAX = 30000

ARXIV_API_PARAMS = {
    "search_query": "all:{}",
    "start": 0,
    "sortBy": "relevance",
}


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _build_search_query(query: str) -> str:
    """Build an arXiv search query.

    Explicit paper-title-style queries are searched against the title field
    to improve precision. Broader research queries continue to use the
    existing all-field search behavior.

    Examples:
        "Attention Is All You Need"
            -> ti:"Attention Is All You Need"

        "recent papers about RAG evaluation"
            -> all:recent papers about RAG evaluation
    """
    normalized = " ".join(query.split()).strip()

    if not normalized:
        return "all:*"

    # A title-focused query is appropriate when the input looks like a
    # specific paper title rather than a broad research topic.
    #
    # Keep this intentionally conservative so normal academic discovery
    # queries continue using broad arXiv retrieval.
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

    # If the query explicitly asks to find a named paper, use title search.
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

    return ARXIV_API_PARAMS["search_query"].format(normalized)


# ---------------------------------------------------------------------------
# arXiv API request
# ---------------------------------------------------------------------------

async def _fetch_arxiv_xml(
    query: str,
    max_results: int,
) -> Optional[bytes]:
    """Fetch arXiv API results as XML bytes.

    Uses the asyncio event loop to run blocking HTTP in a thread.
    """
    search_query = _build_search_query(query)

    params = "&".join(
        (
            f"{key}={quote_plus(search_query)}"
            if key == "search_query"
            else f"{key}={value}"
        )
        for key, value in ARXIV_API_PARAMS.items()
    )

    params += f"&max_results={min(max_results, ARXIV_API_MAX)}"

    url = f"{ARXIV_API_URL}?{params}"

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

            with urllib_request.urlopen(request, timeout=15) as response:
                return response.read()

        xml_bytes = await loop.run_in_executor(None, _request)

        return xml_bytes

    except (HTTPError, URLError, OSError) as exc:
        logger.warning(f"ArXiv API request failed: {exc}")
        return None

    except Exception as exc:
        logger.warning(f"Unexpected error fetching arXiv: {exc}")
        return None


# ---------------------------------------------------------------------------
# arXiv response parsing
# ---------------------------------------------------------------------------

def _parse_arxiv_entry(entry: ElementTree.Element) -> AcademicPaper:
    """Parse a single <entry> element from the arXiv API response."""

    try:
        atom_namespace = "http://www.w3.org/2005/Atom"

        # Title
        title_elem = entry.find(
            f"{{{atom_namespace}}}title"
        )

        title = (
            " ".join(title_elem.text.split())
            if title_elem is not None and title_elem.text
            else None
        )

        # Authors
        authors: list[str] = []

        for author_elem in entry.findall(
            f"{{{atom_namespace}}}author"
        ):
            name_elem = author_elem.find(
                f"{{{atom_namespace}}}name"
            )

            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        # Abstract
        abstract_elem = entry.find(
            f"{{{atom_namespace}}}summary"
        )

        abstract = (
            abstract_elem.text.strip()
            if abstract_elem is not None and abstract_elem.text
            else None
        )

        # Publication date
        published_elem = entry.find(
            f"{{{atom_namespace}}}published"
        )

        publication_date = None
        year = None

        if published_elem is not None and published_elem.text:
            pub_text = published_elem.text.strip()

            match = re.match(r"(\d{4})", pub_text)

            if match:
                year = int(match.group(1))
                publication_date = pub_text

        # arXiv ID
        id_elem = entry.find(
            f"{{{atom_namespace}}}id"
        )

        arxiv_id = None

        if id_elem is not None and id_elem.text:
            abs_match = re.search(
                r"arxiv.org/abs/([^\s/]+)",
                id_elem.text,
            )

            if abs_match:
                arxiv_id = abs_match.group(1)

        # PDF URL
        pdf_url = None

        for link in entry.findall(
            f"{{{atom_namespace}}}link"
        ):
            title_attr = link.get("title", "")

            if title_attr == "pdf":
                href = link.get("href")

                if href:
                    pdf_url = href

        return AcademicPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=publication_date,
            year=year,
            provider="arxiv",
            paper_url=(
                id_elem.text
                if id_elem is not None and id_elem.text
                else None
            ),
            pdf_url=pdf_url,
            doi=None,
            arxiv_id=arxiv_id,
        )

    except Exception as exc:
        logger.warning(
            f"Failed to parse arXiv entry: {exc}"
        )

        return AcademicPaper()


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
    xml_bytes = await _fetch_arxiv_xml(
        query,
        max_results,
    )

    if xml_bytes is None:
        return []

    try:
        root = ElementTree.fromstring(xml_bytes)

        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        entries = root.findall(
            "atom:entry",
            namespace,
        )

        papers: List[AcademicPaper] = []

        for entry in entries[:max_results]:
            paper = _parse_arxiv_entry(entry)

            if paper.title:
                papers.append(paper)

        return papers

    except ElementTree.ParseError as exc:
        logger.warning(
            f"Failed to parse arXiv XML: {exc}"
        )

        return []


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