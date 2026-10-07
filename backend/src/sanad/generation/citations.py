"""Citation extraction and validation for generated legal answers."""

import re

from pydantic import BaseModel

from sanad.corpus.arabic_text import to_western_digits
from sanad.retrieval.reranker import SearchResult


class CitationItem(BaseModel):
    """Structured citation referencing an Egyptian Civil Code article."""

    article_number: int
    citation: str
    text_ar_snippet: str
    text_en_snippet: str
    source_page: int
    is_grounded: bool = True
    is_repealed: bool = False


def extract_citations(
    answer: str,
    retrieved_chunks: list[SearchResult],
) -> list[CitationItem]:
    """Extract cited article numbers from text and map to retrieved chunk records."""
    if not answer:
        return []

    # Map retrieved chunks by article number
    retrieved_map: dict[int, SearchResult] = {}
    for item in retrieved_chunks:
        num = item.chunk.article_number
        if num not in retrieved_map:
            retrieved_map[num] = item

    # Extract cited article numbers from brackets [المادة N] and [Article N]
    normalized_text = to_western_digits(answer)

    found_numbers: set[int] = set()

    # Pattern for bracketed citations: [المادة 147], [Article 147]
    for m in re.finditer(r"\[(?:المادة|Article)\s*(\d+)\]", normalized_text, re.IGNORECASE):
        found_numbers.add(int(m.group(1)))

    # Fallback pattern for unbracketed mentions: المادة 147, Article 147
    if not found_numbers:
        for m in re.finditer(r"\b(?:المادة|Article)\s*(\d+)\b", normalized_text, re.IGNORECASE):
            found_numbers.add(int(m.group(1)))

    citations: list[CitationItem] = []
    for num in sorted(found_numbers):
        if num in retrieved_map:
            chunk = retrieved_map[num].chunk
            citations.append(
                CitationItem(
                    article_number=num,
                    citation=chunk.citation,
                    text_ar_snippet=chunk.text_ar[:200]
                    + ("..." if len(chunk.text_ar) > 200 else ""),
                    text_en_snippet=chunk.text_en[:200]
                    + ("..." if len(chunk.text_en) > 200 else ""),
                    source_page=chunk.source_page,
                    is_grounded=True,
                    is_repealed=chunk.is_repealed,
                )
            )
        else:
            # Ungrounded citation hallucinated by model!
            citations.append(
                CitationItem(
                    article_number=num,
                    citation=f"Egyptian Civil Code, Article {num}",
                    text_ar_snippet="",
                    text_en_snippet="",
                    source_page=0,
                    is_grounded=False,
                    is_repealed=False,
                )
            )

    return citations
