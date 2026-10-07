"""Post-generation citation and content grounding verification guardrail."""

import re

from pydantic import BaseModel, Field

from sanad.corpus.arabic_text import normalize_arabic, to_western_digits
from sanad.generation.prompts import REFUSAL_AR, REFUSAL_EN
from sanad.retrieval.reranker import SearchResult


class GroundingCheckResult(BaseModel):
    """Result of citation and content grounding evaluation."""

    is_grounded: bool
    hallucinated_citations: list[int] = Field(default_factory=list)
    grounded_citations: list[int] = Field(default_factory=list)
    confidence_score: float = 1.0
    reason: str = "Grounded"


def _extract_cited_article_numbers(text: str) -> list[int]:
    """Extract article numbers cited in generated text."""
    norm_text = to_western_digits(text)
    cited: set[int] = set()

    for m in re.finditer(r"\[(?:المادة|Article)\s*(\d+)\]", norm_text, re.IGNORECASE):
        cited.add(int(m.group(1)))

    if not cited:
        for m in re.finditer(r"\b(?:المادة|Article)\s*(\d+)\b", norm_text, re.IGNORECASE):
            cited.add(int(m.group(1)))

    return sorted(cited)


def check_grounding(
    answer: str,
    retrieved_chunks: list[SearchResult],
    overlap_threshold: float = 0.25,
) -> GroundingCheckResult:
    """Verify that generated answer is strictly grounded in the retrieved context."""
    if not answer or not answer.strip():
        return GroundingCheckResult(
            is_grounded=False,
            confidence_score=0.0,
            reason="Empty answer produced",
        )

    # 1. If answer is the expected refusal message, it is correctly grounded
    clean_ans = answer.strip()
    if REFUSAL_AR in clean_ans or REFUSAL_EN in clean_ans:
        return GroundingCheckResult(
            is_grounded=True,
            confidence_score=1.0,
            reason="Valid refusal",
        )

    # If no context was provided but answer was not a refusal -> ungrounded hallucination!
    if not retrieved_chunks:
        return GroundingCheckResult(
            is_grounded=False,
            confidence_score=0.0,
            reason="Answer provided without retrieved context",
        )

    # 2. Check cited article numbers against retrieved article numbers
    retrieved_numbers = {item.chunk.article_number for item in retrieved_chunks}
    cited_numbers = _extract_cited_article_numbers(answer)

    grounded_citations = [n for n in cited_numbers if n in retrieved_numbers]
    hallucinated_citations = [n for n in cited_numbers if n not in retrieved_numbers]

    # If any citation points to an unretrieved article -> FAIL
    if hallucinated_citations:
        return GroundingCheckResult(
            is_grounded=False,
            hallucinated_citations=hallucinated_citations,
            grounded_citations=grounded_citations,
            confidence_score=0.0,
            reason=f"Hallucinated citations not in retrieved context: {hallucinated_citations}",
        )

    # 3. Check lexical overlap between answer and retrieved text
    answer_norm = normalize_arabic(answer)
    answer_tokens = set(answer_norm.split())
    # Filter short stopwords
    substantive_tokens = {t for t in answer_tokens if len(t) > 2}

    if not substantive_tokens:
        return GroundingCheckResult(
            is_grounded=True,
            grounded_citations=grounded_citations,
            confidence_score=1.0,
            reason="No substantive tokens to verify",
        )

    context_words: set[str] = set()
    for item in retrieved_chunks:
        c_norm = normalize_arabic(item.chunk.text_ar_normalized or item.chunk.text_ar)
        context_words.update(c_norm.split())
        en_words = item.chunk.text_en.lower().split()
        context_words.update(en_words)

    overlap_count = sum(1 for t in substantive_tokens if t in context_words)
    overlap_ratio = overlap_count / len(substantive_tokens)

    # If model answers substantively without any citation and very low overlap, flag it
    if not cited_numbers and overlap_ratio < overlap_threshold:
        return GroundingCheckResult(
            is_grounded=False,
            confidence_score=overlap_ratio,
            reason=f"Answer has low overlap ({overlap_ratio:.2f}) and lacks citations",
        )

    return GroundingCheckResult(
        is_grounded=True,
        grounded_citations=grounded_citations,
        confidence_score=overlap_ratio,
        reason="Grounded in retrieved context",
    )


def enforce_grounding_guardrail(
    answer: str,
    retrieved_chunks: list[SearchResult],
    refusal_message_ar: str = REFUSAL_AR,
    refusal_message_en: str = REFUSAL_EN,
    is_arabic: bool = True,
    overlap_threshold: float = 0.25,
) -> tuple[str, GroundingCheckResult]:
    """Run grounding check and replace ungrounded answers with the canonical refusal."""
    result = check_grounding(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        overlap_threshold=overlap_threshold,
    )

    if not result.is_grounded:
        fallback = refusal_message_ar if is_arabic else refusal_message_en
        return fallback, result

    return answer, result
