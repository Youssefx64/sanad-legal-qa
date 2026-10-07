"""Tests for incremental batch re-indexing."""

import json
from pathlib import Path
from typing import Any

import pytest

from sanad.corpus.schema import ArticleRecord
from sanad.indexing.batch_reindex import run_batch_reindex
from sanad.indexing.vector_store import VectorStore
from sanad.providers.base import EmbeddingProvider


class MockVectorStore(VectorStore):
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.upserted_chunks: list[Any] = []
        self.upserted_vectors: list[list[float]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection_if_not_exists(self, collection_name: str, dimension: int) -> None:
        self.collections.add(collection_name)

    def upsert(self, collection_name: str, chunks: list[Any], vectors: list[list[float]]) -> None:
        self.upserted_chunks.extend(chunks)
        self.upserted_vectors.extend(vectors)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[Any]:
        return []

    def count(self, collection_name: str) -> int:
        return len(self.upserted_chunks)


class MockEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 4

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.mark.asyncio
async def test_batch_reindex_with_new_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create new document JSON
    new_article = ArticleRecord(
        article_number=1150,
        book="New Amendments",
        part="Part 1",
        chapter="Chapter 1",
        section="Electronic Commerce",
        topic="Digital Contracts",
        text_ar="يجوز إبرام العقود بالوسائل الإلكترونية وتكون لها ذات القوة الملزمة.",
        text_ar_raw="يجوز إبرام العقود بالوسائل الإلكترونية...",
        text_en="Contracts may be concluded electronically and have equivalent binding force.",
        is_repealed=False,
        source_page=120,
        citation="Egyptian Civil Code, Article 1150 (Supplementary)",
        references=[147],
    )

    new_doc_file = tmp_path / "new_article_1150.json"
    new_doc_file.write_text(
        json.dumps([new_article.model_dump()], ensure_ascii=False), encoding="utf-8"
    )

    vector_store = MockVectorStore()

    res = await run_batch_reindex(
        new_documents_path=new_doc_file,
        provider=MockEmbeddingProvider(),
        vector_store=vector_store,
    )

    assert res["status"] == "reindexed_successfully"
    assert res["new_articles_count"] == 1
    assert res["new_chunks_count"] >= 1
    assert len(vector_store.upserted_chunks) >= 1
    assert vector_store.upserted_chunks[0].article_number == 1150
