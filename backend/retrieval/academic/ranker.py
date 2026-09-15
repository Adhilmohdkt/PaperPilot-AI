"""Paper normalization, deduplication, and deterministic ranking for Phase 2."""
from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .base import AcademicPaper, AcademicSearchProvider


# ── Configuration ──────────────────────────────────────────────────────────

# Weight for lexical-relevance score (0.0–1.0)
LEXICAL_WEIGHT: float = 0.55

# Weight for recency score (newer papers get a small boost)
RECENCY_WEIGHT: float = 0.25

# Weight for citation-count score (more cited → slightly higher)
CITATION_WEIGHT: float = 0.20

# How many years back a paper is "recent" for the recency boost
RECENCY_WINDOW_YEARS: int = 5

# Minimum citation count to contribute to the citation score
MIN_CITATIONS_FOR_SCORE: int = 1

# Threshold for considering two papers "duplicates" (title similarity)
TITLE_DUPLICATE_THRESHOLD: float = 0.85


# ── Shared helpers ────────────────────────────────────────────────────────

def _normalize_text(txt: str | None) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not txt:
        return ""
    return " ".join(txt.lower().split())


def _token_set(text: str) -> set[str]:
    """Return a set of significant tokens from text."""
    return {t for t in _normalize_text(text).split() if len(t) > 2}


def _lexical_score(query_tokens: set[str], paper_tokens: set[str]) -> float:
    """Jaccard-like overlap between query tokens and paper tokens (title+abstract)."""
    if not query_tokens or not paper_tokens:
        return 0.0
    intersection = len(query_tokens & paper_tokens)
    union = len(query_tokens | paper_tokens)
    return intersection / union if union else 0.0


def _year_score(year: int | None, current_year: int = 2025) -> float:
    """Recency score: newer papers score higher.

    Linear decay: a paper from (current_year - RECENCY_WINDOW_YEARS) gets 0.5,
    older papers get proportionally less.
    """
    if year is None:
        return 0.5  # neutral — no penalty, no boost
    age = current_year - year
    if age <= 0:
        return 1.0
    if age >= RECENCY_WINDOW_YEARS:
        return 0.5
    # Linear interpolation: age 0→1.0, age N→0.5
    return 1.0 - (age / RECENCY_WINDOW_YEARS) * 0.5


def _citation_score(citations: int | None) -> float:
    """Citation-count score: logarithmic scaling so very-high doesn't dominate."""
    if citations is None or citations < MIN_CITATIONS_FOR_SCORE:
        return 0.5  # neutral
    # log10 scale, capped: 1 citation ≈ 0.8, 100 ≈ 1.0
    raw = min(math.log10(max(citations, 1)), 1.0)
    return 0.5 + raw * 0.5


def _paper_text_tokens(paper: AcademicPaper) -> set[str]:
    """Token set from a paper's title and abstract."""
    tokens: set[str] = _token_set(paper.title or "")
    tokens.update(_token_set(paper.abstract or ""))
    return tokens


def _is_duplicate(paper_a: AcademicPaper, paper_b: AcademicPaper) -> bool:
    """Heuristic: two papers are likely duplicates if their title tokens overlap heavily."""
    tokens_a = _paper_text_tokens(paper_a)
    tokens_b = _paper_text_tokens(paper_b)
    if not tokens_a or not tokens_b:
        return False
    return _lexical_score(tokens_a, tokens_b) >= TITLE_DUPLICATE_THRESHOLD


def _dedupe_papers(papers: List[AcademicPaper]) -> List[AcademicPaper]:
    """Remove near-duplicate papers based on title similarity.

    Keeps the paper with the higher citation count (or earlier year as tiebreaker).
    """
    if not papers:
        return []
    deduped: List[AcademicPaper] = [papers[0]]
    for candidate in papers[1:]:
        is_dup = False
        for existing in deduped:
            if _is_duplicate(existing, candidate):
                # Keep the one with more citations; if tied, earlier year
                if (candidate.citation_count or 0) > (existing.citation_count or 0):
                    # Replace existing with candidate
                    deduped[deduped.index(existing)] = candidate
                is_dup = True
                break
        if not is_dup:
            deduped.append(candidate)
    return deduped


def _rank_papers(
    query: str,
    papers: List[AcademicPaper],
    top_k: int = 10,
) -> List[AcademicPaper]:
    """Rank papers by a deterministic composite score and return top_k.

    The composite score combines:
      - Lexical query relevance (Jaccard overlap of query vs. title+abstract)
      - Recency (newer = better)
      - Citation count (more cited = slightly better)
    """
    if not papers:
        return []

    query_tokens = _token_set(query)

    # Compute a composite score for each paper
    scored: List[tuple[AcademicPaper, float]] = []
    for paper in papers:
        lexical = _lexical_score(query_tokens, _paper_text_tokens(paper))
        recency = _year_score(paper.year)
        citation = _citation_score(paper.citation_count)

        composite = (
            LEXICAL_WEIGHT * lexical
            + RECENCY_WEIGHT * recency
            + CITATION_WEIGHT * citation
        )
        scored.append((paper, composite))

    # Sort descending by composite score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top_k
    top_n = min(top_k, len(scored))
    return [paper for paper, _ in scored[:top_n]]


# ── Public API ────────────────────────────────────────────────────────────

def normalize_and_rank(
    query: str,
    papers: List[AcademicPaper],
    top_k: int = 5,
) -> List[AcademicPaper]:
    """Full pipeline: dedupe → rank → return top_k.

    Args:
        query: The original user query.
        papers: Raw papers from one or more providers (may contain duplicates).
        top_k: Number of final papers to return.

    Returns:
        A deduplicated, reranked list of AcademicPaper objects, sorted by
        relevance to the query.
    """
    # Step 1: deduplicate
    deduped = _dedupe_papers(papers)

    # Step 2: rank
    ranked = _rank_papers(query, deduped, top_k=top_k)

    # Step 3: assign final relevance scores (for downstream use)
    query_tokens = _token_set(query)
    for i, paper in enumerate(ranked, start=1):
        paper.relevance_score = round(
            _lexical_score(query_tokens, _paper_text_tokens(paper)), 3
        )

    return ranked

# Backwards-compatibility aliases for nodes.py import
rank_papers = normalize_and_rank
normalize_papers = normalize_and_rank