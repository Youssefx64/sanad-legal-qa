"""Retrieval metadata filters and intent detection."""

from pydantic import BaseModel

from sanad.corpus.arabic_text import extract_article_references, to_western_digits


class RetrievalFilter(BaseModel):
    """Metadata filtering criteria for hybrid search."""

    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    exclude_repealed: bool = True
    article_numbers: list[int] | None = None


def detect_query_filters(query: str) -> tuple[RetrievalFilter, list[int]]:
    """Detect filter criteria and explicit article numbers from user query.

    Returns (RetrievalFilter, list[explicit_article_numbers]).
    """
    clean_q = to_western_digits(query).lower()

    # Check if query explicitly asks about repealed articles
    repealed_keywords = ["ملغاة", "ملغاه", "الغيت", "ألغيت", "repealed", "repeal", "annulled"]
    asks_repealed = any(kw in clean_q for kw in repealed_keywords)

    # Extract explicit article references
    explicit_articles = extract_article_references(query, query)

    # If asking about known repealed articles directly (e.g. Article 54), don't exclude repealed
    if any(54 <= num <= 80 or 389 <= num <= 417 for num in explicit_articles):
        asks_repealed = True

    filters = RetrievalFilter(
        exclude_repealed=not asks_repealed,
        article_numbers=explicit_articles if explicit_articles else None,
    )

    return filters, explicit_articles
