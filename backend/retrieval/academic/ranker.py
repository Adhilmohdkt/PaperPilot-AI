"""Academic paper normalization, deduplication, and deterministic ranking."""

from __future__ import annotations

import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import List

from .base import AcademicPaper


# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------

LEXICAL_WEIGHT = 0.60
RECENCY_WEIGHT = 0.25
CITATION_WEIGHT = 0.15

RECENCY_WINDOW_YEARS = 5

MIN_CITATIONS_FOR_SCORE = 1

TITLE_DUPLICATE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def _normalize_text(text: str | None) -> str:
    """Normalize text for comparison."""

    if not text:
        return ""

    text = text.lower()

    # Normalize punctuation to spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace.
    return " ".join(text.split())


def _token_set(text: str) -> set[str]:
    """Return meaningful tokens from text."""

    normalized = _normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) > 2
    }


# ---------------------------------------------------------------------------
# Query relevance
# ---------------------------------------------------------------------------

_ACADEMIC_STOPWORDS = {
    "about",
    "and",
    "are",
    "based",
    "for",
    "from",
    "into",
    "methods",
    "model",
    "models",
    "of",
    "on",
    "paper",
    "papers",
    "research",
    "the",
    "this",
    "using",
    "with",
}


def _query_phrases(query: str) -> list[str]:
    """Extract meaningful multi-word phrases from an academic query."""

    normalized = _normalize_text(query)
    tokens = normalized.split()

    phrases: list[str] = []

    # Preserve common technical multi-word concepts when they occur
    # consecutively in the query.
    for size in (4, 3, 2):
        for index in range(len(tokens) - size + 1):
            phrase_tokens = tokens[index:index + size]

            if all(
                token not in _ACADEMIC_STOPWORDS
                for token in phrase_tokens
            ):
                phrases.append(" ".join(phrase_tokens))

    # Remove phrases contained inside longer phrases.
    unique_phrases: list[str] = []

    for phrase in phrases:
        if not any(
            phrase != other and phrase in other
            for other in phrases
        ):
            unique_phrases.append(phrase)

    return unique_phrases


def _lexical_score(
    query: str,
    paper: AcademicPaper,
) -> float:
    """Calculate phrase-aware lexical relevance."""

    normalized_query = _normalize_text(query)

    if not normalized_query:
        return 0.0

    normalized_title = _normalize_text(paper.title)
    normalized_abstract = _normalize_text(paper.abstract)

    if not normalized_title and not normalized_abstract:
        return 0.0

    query_tokens = _token_set(query)
    title_tokens = _token_set(paper.title or "")
    abstract_tokens = _token_set(paper.abstract or "")

    if not query_tokens:
        return 0.0

    # Query coverage.
    title_coverage = (
        len(query_tokens & title_tokens) / len(query_tokens)
    )

    abstract_coverage = (
        len(query_tokens & abstract_tokens) / len(query_tokens)
    )

    # Title is more informative than abstract for academic search.
    token_score = (
        0.70 * title_coverage
        + 0.30 * abstract_coverage
    )

    # Exact phrase matches provide stronger semantic evidence.
    phrase_score = 0.0

    for phrase in _query_phrases(query):
        if phrase in normalized_title:
            phrase_score = max(phrase_score, 1.0)
        elif phrase in normalized_abstract:
            phrase_score = max(phrase_score, 0.6)

    # Exact full-query match is a very strong signal.
    exact_query_bonus = 0.0

    if normalized_query in normalized_title:
        exact_query_bonus = 1.0
    elif normalized_query in normalized_abstract:
        exact_query_bonus = 0.5

    score = (
        0.70 * token_score
        + 0.25 * phrase_score
        + 0.05 * exact_query_bonus
    )

    return min(score, 1.0)


def _paper_text_tokens(
    paper: AcademicPaper,
) -> set[str]:
    """Return tokens from title and abstract."""

    title_tokens = _token_set(
        paper.title or ""
    )

    abstract_tokens = _token_set(
        paper.abstract or ""
    )

    return title_tokens | abstract_tokens


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------


def _year_score(
    year: int | None,
    current_year: int | None = None,
) -> float:
    """Return a deterministic recency score between 0.5 and 1.0."""

    if current_year is None:
        current_year = datetime.now().year

    if year is None:
        return 0.5

    age = max(0, current_year - year)

    if age >= RECENCY_WINDOW_YEARS:
        return 0.5

    return 1.0 - (
        age / RECENCY_WINDOW_YEARS
    ) * 0.5


# ---------------------------------------------------------------------------
# Citation scoring
# ---------------------------------------------------------------------------


def _citation_score(
    citations: int | None,
) -> float:
    """Return a logarithmically scaled citation score."""

    if citations is None or citations < MIN_CITATIONS_FOR_SCORE:
        return 0.5

    # Prevent extremely highly cited papers from dominating relevance.
    log_score = min(
        math.log10(max(citations, 1)),
        2.0,
    )

    # Map approximately:
    # 1 citation   -> 0.50
    # 10 citations -> 0.75
    # 100 citations -> 1.00
    return min(
        1.0,
        0.5 + (log_score / 4.0),
    )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def _normalize_title(
    title: str | None,
) -> str:
    """Normalize a title specifically for duplicate detection."""

    return _normalize_text(title)


def _same_identifier(
    paper_a: AcademicPaper,
    paper_b: AcademicPaper,
) -> bool:
    """Check whether two papers share a reliable identifier."""

    # arXiv identifier.
    if (
        paper_a.arxiv_id
        and paper_b.arxiv_id
        and paper_a.arxiv_id.lower()
        == paper_b.arxiv_id.lower()
    ):
        return True

    # DOI.
    if (
        paper_a.doi
        and paper_b.doi
        and paper_a.doi.lower().strip()
        == paper_b.doi.lower().strip()
    ):
        return True

    # OpenAlex ID.
    if (
        paper_a.openalex_id
        and paper_b.openalex_id
        and paper_a.openalex_id.lower()
        == paper_b.openalex_id.lower()
    ):
        return True

    return False


def _title_similarity(
    title_a: str,
    title_b: str,
) -> float:
    """Calculate normalized title similarity."""

    normalized_a = _normalize_title(title_a)
    normalized_b = _normalize_title(title_b)

    if not normalized_a or not normalized_b:
        return 0.0

    return SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()


def _is_duplicate(
    paper_a: AcademicPaper,
    paper_b: AcademicPaper,
) -> bool:
    """Determine whether two academic records represent the same paper."""

    # Reliable identifiers are the strongest signal.
    if _same_identifier(paper_a, paper_b):
        return True

    # Fall back to title similarity.
    similarity = _title_similarity(
        paper_a.title or "",
        paper_b.title or "",
    )

    return similarity >= TITLE_DUPLICATE_THRESHOLD


def _dedupe_papers(
    papers: List[AcademicPaper],
) -> List[AcademicPaper]:
    """Remove duplicate papers while preferring richer records."""

    if not papers:
        return []

    deduped: List[AcademicPaper] = []

    for candidate in papers:

        duplicate_index = None

        for index, existing in enumerate(deduped):

            if _is_duplicate(
                existing,
                candidate,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            deduped.append(candidate)
            continue

        existing = deduped[duplicate_index]

        # Prefer the record with more complete metadata.
        existing_completeness = _metadata_completeness(
            existing
        )

        candidate_completeness = _metadata_completeness(
            candidate
        )

        if candidate_completeness > existing_completeness:
            deduped[duplicate_index] = candidate

        elif (
            candidate_completeness
            == existing_completeness
            and (candidate.citation_count or 0)
            > (existing.citation_count or 0)
        ):
            deduped[duplicate_index] = candidate

    return deduped


def _metadata_completeness(
    paper: AcademicPaper,
) -> int:
    """Score how complete a paper record is."""

    fields = [
        paper.title,
        paper.abstract,
        paper.authors,
        paper.publication_date,
        paper.paper_url,
        paper.pdf_url,
        paper.doi,
        paper.arxiv_id,
        paper.openalex_id,
    ]

    return sum(
        1
        for field in fields
        if field
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank_papers(
    query: str,
    papers: List[AcademicPaper],
    top_k: int = 10,
) -> List[AcademicPaper]:
    """Rank papers using deterministic relevance signals."""

    if not papers:
        return []

    scored: List[
        tuple[AcademicPaper, float, float]
    ] = []

    for paper in papers:

        lexical = _lexical_score(
            query,
            paper,
        )

        recency = _year_score(
            paper.year
        )

        citation = _citation_score(
            paper.citation_count
        )

        composite = (
            LEXICAL_WEIGHT * lexical
            + RECENCY_WEIGHT * recency
            + CITATION_WEIGHT * citation
        )

        scored.append(
            (
                paper,
                composite,
                lexical,
            )
        )

    # Primary:
    #   composite relevance
    #
    # Secondary:
    #   lexical relevance
    #
    # Tertiary:
    #   citation count
    #
    # This keeps ranking deterministic.

    scored.sort(
        key=lambda item: (
            item[1],
            item[2],
            item[0].citation_count or 0,
        ),
        reverse=True,
    )

    ranked = [
        item[0]
        for item in scored[:top_k]
    ]

    # Store the actual composite relevance score.
    for paper, composite, _ in scored[:top_k]:
        paper.relevance_score = round(
            composite,
            3,
        )

    return ranked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_and_rank(
    query: str,
    papers: List[AcademicPaper],
    top_k: int = 5,
) -> List[AcademicPaper]:
    """Deduplicate, rank, and return the top academic papers."""

    if not papers:
        return []

    # 1. Remove duplicate records.
    deduped = _dedupe_papers(
        papers
    )

    # 2. Deterministically rank remaining papers.
    ranked = _rank_papers(
        query,
        deduped,
        top_k=top_k,
    )

    return ranked


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

rank_papers = normalize_and_rank
normalize_papers = normalize_and_rank