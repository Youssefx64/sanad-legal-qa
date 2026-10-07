"""Corpus exploration and individual article lookup endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter

from sanad.api.errors import ArticleNotFoundError
from sanad.api.schemas import CorpusStatsResponse
from sanad.corpus.schema import ArticleRecord

router = APIRouter(tags=["Corpus"])

# Cached articles mapping
_articles_cache: dict[int, ArticleRecord] | None = None


def _get_articles_cache() -> dict[int, ArticleRecord]:
    global _articles_cache
    if _articles_cache is None:
        path = Path("data/processed/articles.json")
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _articles_cache = {
                item["article_number"]: ArticleRecord.model_validate(item) for item in data
            }
        else:
            _articles_cache = {}
    return _articles_cache


@router.get("/articles/{number}", response_model=ArticleRecord)
async def get_article_by_number(number: int) -> ArticleRecord:
    """Retrieve full text and metadata for a specific Egyptian Civil Code article."""
    if number < 1 or number > 1149:
        raise ArticleNotFoundError(number)

    cache = _get_articles_cache()
    if number not in cache:
        raise ArticleNotFoundError(number)

    return cache[number]


@router.get("/corpus/stats", response_model=CorpusStatsResponse)
async def get_corpus_statistics() -> CorpusStatsResponse:
    """Retrieve summary statistics, total articles, and book hierarchy."""
    cache = _get_articles_cache()
    articles = list(cache.values())

    repealed_count = sum(1 for a in articles if a.is_repealed)
    books = sorted({a.book for a in articles if a.book})

    return CorpusStatsResponse(
        total_articles=len(articles),
        active_articles=len(articles) - repealed_count,
        repealed_articles=repealed_count,
        books=books,
    )
