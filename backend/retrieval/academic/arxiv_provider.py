"""ArXiv academic-paper provider for PaperPilot Phase 2."""
from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote_plus
from xml.etree import ElementTree

from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .base import (
    AcademicPaper,
    AcademicSearchProvider,
)

logger = logging.getLogger(__name__)

# ArXiv API endpoint
ARXIV_API_URL = "https://export.arxiv.org/api/query"
# ArXiv default maximum; the API caps at 30000 but we use a smaller safe default
ARXIV_API_MAX = 30000
# arXiv sorts by relevance or last updated date; we ask for results sorted by relevance
ARXIV_API_PARAMS = {
    "search_query": "all:{}",
    "start": 0,
    "sortBy": "relevance",
}


async def _fetch_arxiv_xml(query: str, max_results: int) -> Optional[bytes]:
    """Fetch arXiv API results as XML bytes.

    Uses the asyncio event loop to run blocking HTTP in a thread.
    """
    # Only format the search_query parameter; others are already complete values
    search_query = ARXIV_API_PARAMS["search_query"].format(query)
    params = "&".join(
        f"{k}={quote_plus(search_query)}" if k == "search_query" else f"{k}={v}"
        for k, v in ARXIV_API_PARAMS.items()
    )
    params += f"&max_results={min(max_results, ARXIV_API_MAX)}"

    url = f"{ARXIV_API_URL}?{params}"

    loop = asyncio.get_event_loop()
    try:
        def _request():
            request = urllib_request.Request(
                url,
                headers={"User-Agent": "PaperPilot-AI/1.0 (academic research client)"},
            )
            with urllib_request.urlopen(request, timeout=15) as resp:
                return resp.read()

        xml_bytes = await loop.run_in_executor(None, _request)
        return xml_bytes
    except (HTTPError, URLError, OSError) as e:
        logger.warning(f"ArXiv API request failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching arXiv: {e}")
        return None


def _parse_arxiv_entry(entry: ElementTree.Element) -> AcademicPaper:
    """Parse a single <entry> element from the arXiv API response."""
    try:
        # Title
        title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
        title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else None

        # Authors
        authors: list[str] = []
        for author_elem in entry.findall(
            "{http://www.w3.org/2005/Atom}author"
        ):
            name_elem = author_elem.find("{http://www.w3.org/2005/Atom}name")
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        # Abstract
        abstract_elem = entry.find(
            "{http://www.w3.org/2005/Atom}summary"
        )
        abstract = abstract_elem.text.strip() if abstract_elem is not None else None

        # Published date
        published_elem = entry.find(
            "{http://www.w3.org/2005/Atom}published"
        )
        publication_date = None
        year = None
        if published_elem is not None and published_elem.text:
            pub_text = published_elem.text.strip()
            # Parse ISO date: "2024-01-15" or similar
            m = re.match(r"(\d{4})", pub_text)
            if m:
                year = int(m.group(1))
                publication_date = pub_text

        # arXiv ID from the id element
        id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
        arxiv_id = None
        if id_elem is not None and id_elem.text:
            # arXiv IDs look like http://arxiv.org/abs/2401.12345
            abs_match = re.search(r"arxiv.org/abs/([^\s/]+)", id_elem.text)
            if abs_match:
                arxiv_id = abs_match.group(1)

        # PDF URL
        pdf_url = None
        for link in entry.findall(
            "{http://www.w3.org/2005/Atom}link"
        ):
            title_attr = link.get("{http://www.w3.org/2005/Atom}title", "")
            if title_attr == "pdf":
                href = link.get("{http://www.w3.org/2005/Atom}href")
                if href:
                    pdf_url = href

        # DOI (often in the id or extensions, simplified)
        doi = None

        # OpenAlex ID - arXiv papers may have an OpenAle ID in the arXiv id

        return AcademicPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=publication_date,
            year=year,
            provider="arxiv",
            paper_url=id_elem.text if id_elem is not None else None,
            pdf_url=pdf_url,
            doi=doi,
            arxiv_id=arxiv_id,
        )
    except Exception as e:
        logger.warning(f"Failed to parse arXiv entry: {e}")
        return AcademicPaper()


async def search_arxiv(query: str, max_results: int = 50) -> List[AcademicPaper]:
    """Search arXiv for papers matching the query.

    Args:
        query: The search query string.
        max_results: Maximum number of papers to return (default 50).

    Returns:
        A list of AcademicPaper objects, sorted by arXiv relevance.
    """
    xml_bytes = await _fetch_arxiv_xml(query, max_results)
    if xml_bytes is None:
        return []

    try:
        root = ElementTree.fromstring(xml_bytes)
        # The Atom feed has entries in the namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        papers: List[AcademicPaper] = []
        for entry in entries[:max_results]:
            paper = _parse_arxiv_entry(entry)
            if paper.title:  # only add papers with a title
                papers.append(paper)

        return papers
    except ElementTree.ParseError as e:
        logger.warning(f"Failed to parse arXiv XML: {e}")
        return []


class ArxivProvider(AcademicSearchProvider):
    """Concrete provider for arXiv API searches."""

    def __init__(self, max_results: int = 50):
        self.max_results = max_results

    async def search(self, query: str, max_results: int | None = None) -> List[AcademicPaper]:
        """See base class."""
        effective_max = max_results or self.max_results
        return await search_arxiv(query, effective_max)
