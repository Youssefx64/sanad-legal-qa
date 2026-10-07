"""Hybrid retriever combining Dense vector search, BM25 keyword search, and exact article lookup."""

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from sanad.corpus.arabic_text import normalize_arabic
from sanad.indexing.chunker import ChunkRecord
from sanad.indexing.vector_store import VectorStore
from sanad.providers.base import EmbeddingProvider
from sanad.retrieval.filters import RetrievalFilter, detect_query_filters
from sanad.retrieval.reranker import NoOpReranker, Reranker, SearchResult

logger = logging.getLogger(__name__)


class BM25Index:
    """In-memory BM25 index over normalized Arabic chunk texts."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [
            normalize_arabic(c.text_ar_normalized or c.text_ar).split() for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        # Quick lookup map by article number
        self.chunks_by_article: dict[int, list[ChunkRecord]] = {}
        for c in chunks:
            self.chunks_by_article.setdefault(c.article_number, []).append(c)

    def search(
        self,
        query: str,
        limit: int = 20,
        filter_criteria: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = normalize_arabic(query).split()
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_pairs = list(zip(self.chunks, scores, strict=True))

        # Filter criteria
        filtered: list[tuple[ChunkRecord, float]] = []
        for chunk, score in scored_pairs:
            if filter_criteria:
                if filter_criteria.exclude_repealed and chunk.is_repealed:
                    continue
                if filter_criteria.book and chunk.book != filter_criteria.book:
                    continue
                if filter_criteria.chapter and chunk.chapter != filter_criteria.chapter:
                    continue
                if (
                    filter_criteria.article_numbers
                    and chunk.article_number not in filter_criteria.article_numbers
                ):
                    continue
            filtered.append((chunk, float(score)))

        # Sort descending by BM25 score
        filtered.sort(key=lambda x: x[1], reverse=True)
        top = filtered[:limit]

        # Normalize BM25 scores to [0, 1] range for combination
        max_score = top[0][1] if top and top[0][1] > 0 else 1.0

        results: list[SearchResult] = []
        for chunk, score in top:
            norm_score = max(0.0, score / max_score) if max_score > 0 else 0.0
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=norm_score,
                    source="bm25",
                )
            )

        return results

    def get_chunks_for_articles(self, article_numbers: list[int]) -> list[ChunkRecord]:
        """Fetch all chunks belonging to the given article numbers."""
        found: list[ChunkRecord] = []
        for num in article_numbers:
            found.extend(self.chunks_by_article.get(num, []))
        return found


class HybridRetriever:
    """Hybrid legal retriever combining Dense, BM25, and Exact Article Lookup."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        collection_name: str,
        bm25_index: BM25Index | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
        enable_exact_lookup: bool = True,
        min_relevance_score: float = 0.0,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.bm25_index = bm25_index
        self.reranker = reranker or NoOpReranker()
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.enable_exact_lookup = enable_exact_lookup
        self.min_relevance_score = min_relevance_score

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_override: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        """Execute hybrid search pipeline."""
        limit = top_k if top_k is not None else self.top_k

        # 1. Detect query intent, filters, and explicit article numbers
        auto_filters, explicit_articles = detect_query_filters(query)
        filters = filter_override or auto_filters

        exact_results: list[SearchResult] = []
        # 2. Exact article lookup if explicit numbers detected
        if self.enable_exact_lookup and explicit_articles and self.bm25_index:
            exact_chunks = self.bm25_index.get_chunks_for_articles(explicit_articles)
            for c in exact_chunks:
                exact_results.append(
                    SearchResult(
                        chunk=c,
                        score=1.0,
                        source="exact_lookup",
                    )
                )

        # 3. Dense search via Qdrant
        dense_candidates: list[SearchResult] = []
        try:
            q_vecs = await self.embedding_provider.embed([query], is_query=True)
            if q_vecs and self.vector_store.collection_exists(self.collection_name):
                q_vec = q_vecs[0]
                filter_dict: dict[str, Any] = {
                    "exclude_repealed": filters.exclude_repealed,
                }
                if filters.book:
                    filter_dict["book"] = filters.book
                if filters.chapter:
                    filter_dict["chapter"] = filters.chapter

                scored_chunks = self.vector_store.search(
                    collection_name=self.collection_name,
                    query_vector=q_vec,
                    limit=max(limit * 3, 20),
                    filter_criteria=filter_dict,
                )
                for sc in scored_chunks:
                    dense_candidates.append(
                        SearchResult(
                            chunk=sc.chunk,
                            score=sc.score,
                            source="dense",
                        )
                    )
        except Exception as e:
            logger.warning("Dense search failed: %s. Falling back to BM25 only.", e)

        # 4. BM25 search
        bm25_candidates: list[SearchResult] = []
        if self.bm25_index:
            bm25_candidates = self.bm25_index.search(
                query=query,
                limit=max(limit * 3, 20),
                filter_criteria=filters,
            )

        # 5. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, ChunkRecord] = {}

        # Dense ranks
        for rank, res in enumerate(dense_candidates):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                self.dense_weight / (self.rrf_k + rank + 1)
            )

        # BM25 ranks
        for rank, res in enumerate(bm25_candidates):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank + 1)
            )

        # Build fused list
        fused_candidates: list[SearchResult] = []
        for cid, score in rrf_scores.items():
            fused_candidates.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=score,
                    source="hybrid",
                )
            )

        fused_candidates.sort(key=lambda x: x.score, reverse=True)

        # 6. Merge exact lookups at top (deduplicating by chunk_id)
        seen_ids = set()
        final_candidates: list[SearchResult] = []

        for item in exact_results:
            if item.chunk.chunk_id not in seen_ids:
                seen_ids.add(item.chunk.chunk_id)
                final_candidates.append(item)

        for item in fused_candidates:
            if item.chunk.chunk_id not in seen_ids:
                seen_ids.add(item.chunk.chunk_id)
                final_candidates.append(item)

        # 7. Apply reranker
        reranked = await self.reranker.rerank(query, final_candidates, top_k=limit)

        # 8. Filter by min_relevance_score if set
        if self.min_relevance_score > 0.0:
            reranked = [
                r
                for r in reranked
                if r.score >= self.min_relevance_score or r.source == "exact_lookup"
            ]

        return reranked[:limit]
