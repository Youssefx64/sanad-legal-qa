"""Sanad RAG facade unifying retrieval, generation, citations, and guardrails."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sanad.config.registry import ModelRegistry, get_model_registry
from sanad.config.settings import Settings, get_settings
from sanad.corpus.schema import ArticleRecord
from sanad.generation.answerer import LegalAnswerer, StreamEvent
from sanad.generation.citations import CitationItem
from sanad.guardrails.grounding import GroundingCheckResult
from sanad.guardrails.pii import redact_pii
from sanad.indexing.chunker import chunk_corpus
from sanad.indexing.ingest import compute_corpus_hash, run_ingestion
from sanad.indexing.vector_store import (
    QdrantVectorStore,
    VectorStore,
    build_collection_name,
)
from sanad.providers.base import ChatUsage
from sanad.providers.factory import ProviderFactory, get_provider_factory
from sanad.retrieval.filters import RetrievalFilter
from sanad.retrieval.reranker import NoOpReranker, Reranker, SearchResult
from sanad.retrieval.retriever import BM25Index, HybridRetriever

logger = logging.getLogger(__name__)


class QueryResponse(BaseModel):
    """Full structured response from a Sanad legal Q&A query."""

    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    retrieved_articles: list[SearchResult] = Field(default_factory=list)
    grounding: GroundingCheckResult
    usage: ChatUsage
    latency_ms: float
    chat_model_id: str
    embedding_model_id: str
    trace_id: str
    pii_redacted: bool = False


class SanadRAG:
    """End-to-end RAG facade for the Egyptian Civil Code Q&A system."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
        factory: ProviderFactory | None = None,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        answerer: LegalAnswerer | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_model_registry()
        self.factory = factory or get_provider_factory()
        self.vector_store = vector_store or QdrantVectorStore(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        self.answerer = answerer or LegalAnswerer()
        self.reranker = reranker or NoOpReranker()
        self.bm25_index = bm25_index

        # Lazily load BM25Index from articles.json if not passed
        if self.bm25_index is None:
            self._init_bm25()

    def _init_bm25(self) -> None:
        """Initialize BM25 index from processed articles.json on disk."""
        articles_file = Path("data/processed/articles.json")
        if articles_file.exists():
            try:
                with articles_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                articles = [ArticleRecord.model_validate(x) for x in data]
                chunks = chunk_corpus(articles)
                self.bm25_index = BM25Index(chunks)
                logger.info("Initialized BM25 index with %d chunks.", len(chunks))
            except Exception as e:
                logger.warning("Could not initialize BM25 index from %s: %s", articles_file, e)

    def _get_collection_name(self, embedding_model_id: str, dim: int) -> str:
        """Resolve collection name for given embedding model."""
        articles_file = Path("data/processed/articles.json")
        corpus_hash = compute_corpus_hash(articles_file) if articles_file.exists() else "00000000"
        return build_collection_name(embedding_model_id, dim, corpus_hash)

    def _get_retriever(self, embedding_model_id: str) -> HybridRetriever:
        """Build or get a HybridRetriever for the target embedding model."""
        embed_provider = self.factory.get_embedding_provider(embedding_model_id)
        dim = embed_provider.dimension or 1536
        collection_name = self._get_collection_name(embedding_model_id, dim)

        return HybridRetriever(
            vector_store=self.vector_store,
            embedding_provider=embed_provider,
            collection_name=collection_name,
            bm25_index=self.bm25_index,
            reranker=self.reranker,
            top_k=5,
        )

    async def query(
        self,
        question: str,
        chat_model_id: str | None = None,
        embedding_model_id: str | None = None,
        top_k: int = 5,
        language_mode: str = "auto",
        filter_override: RetrievalFilter | None = None,
        enable_pii_masking: bool = True,
        enable_grounding: bool = True,
    ) -> QueryResponse:
        """Execute complete Q&A pipeline: PII scrub -> retrieve -> answer -> verify."""
        start_time = time.perf_counter()
        trace_id = str(uuid.uuid4())

        # Resolve model IDs
        c_model_id = chat_model_id or self.registry.defaults.chat_model
        e_model_id = embedding_model_id or self.registry.defaults.embedding_model

        # 1. PII Scrubbing
        scrubbed_query = question
        pii_was_redacted = False
        if enable_pii_masking:
            scrubbed_query, redactions = redact_pii(question)
            pii_was_redacted = len(redactions) > 0

        # 2. Retrieval
        retriever = self._get_retriever(e_model_id)
        retrieved_chunks = await retriever.retrieve(
            query=scrubbed_query,
            top_k=top_k,
            filter_override=filter_override,
        )

        # 3. Answer Generation
        chat_provider = self.factory.get_chat_provider(c_model_id)
        ans_result = await self.answerer.generate(
            question=scrubbed_query,
            retrieved_chunks=retrieved_chunks,
            provider=chat_provider,
            language_mode=language_mode,
            enable_grounding=enable_grounding,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return QueryResponse(
            answer=ans_result.answer,
            citations=ans_result.citations,
            retrieved_articles=retrieved_chunks,
            grounding=ans_result.grounding,
            usage=ans_result.usage,
            latency_ms=round(latency_ms, 2),
            chat_model_id=c_model_id,
            embedding_model_id=e_model_id,
            trace_id=trace_id,
            pii_redacted=pii_was_redacted,
        )

    async def stream_query(
        self,
        question: str,
        chat_model_id: str | None = None,
        embedding_model_id: str | None = None,
        top_k: int = 5,
        language_mode: str = "auto",
        filter_override: RetrievalFilter | None = None,
        enable_pii_masking: bool = True,
        enable_grounding: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream answer tokens and citations asynchronously."""
        c_model_id = chat_model_id or self.registry.defaults.chat_model
        e_model_id = embedding_model_id or self.registry.defaults.embedding_model

        scrubbed_query = question
        if enable_pii_masking:
            scrubbed_query, _ = redact_pii(question)

        retriever = self._get_retriever(e_model_id)
        retrieved_chunks = await retriever.retrieve(
            query=scrubbed_query,
            top_k=top_k,
            filter_override=filter_override,
        )

        chat_provider = self.factory.get_chat_provider(c_model_id)
        async for event in self.answerer.stream_generate(
            question=scrubbed_query,
            retrieved_chunks=retrieved_chunks,
            provider=chat_provider,
            language_mode=language_mode,
            enable_grounding=enable_grounding,
        ):
            yield event

    async def ingest(
        self,
        embedding_model_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Ingest or re-index processed articles into Qdrant."""
        target_model = embedding_model_id or self.registry.defaults.embedding_model
        return await run_ingestion(
            embedding_model_id=target_model,
            vector_store=self.vector_store,
            force=force,
        )
