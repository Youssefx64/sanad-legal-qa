"""Incremental batch re-indexing for new or amended legal documents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sanad.config.registry import get_model_registry
from sanad.config.settings import get_settings
from sanad.corpus.schema import ArticleRecord
from sanad.indexing.chunker import chunk_corpus
from sanad.indexing.embedder import embed_chunks
from sanad.indexing.ingest import compute_corpus_hash
from sanad.indexing.vector_store import (
    QdrantVectorStore,
    VectorStore,
    build_collection_name,
)
from sanad.providers.base import EmbeddingProvider
from sanad.providers.factory import get_provider_factory

logger = logging.getLogger(__name__)


async def run_batch_reindex(
    new_documents_path: str | Path,
    base_articles_path: str | Path = Path("data/processed/articles.json"),
    embedding_model_id: str | None = None,
    provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    rag_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Incrementally index new or updated articles into the active collection."""
    new_path = Path(new_documents_path)
    if not new_path.exists():
        raise FileNotFoundError(f"New documents file not found at: {new_path}")

    with open(new_path, encoding="utf-8") as f:
        new_data = json.load(f)

    if isinstance(new_data, dict):
        new_data = [new_data]

    new_articles = [ArticleRecord.model_validate(item) for item in new_data]

    settings = get_settings()
    registry = get_model_registry()

    target_model_id = embedding_model_id or registry.defaults.embedding_model
    model_cfg = registry.get_embedding_model(target_model_id)

    if provider is None:
        factory = get_provider_factory()
        provider = factory.get_embedding_provider(target_model_id)

    dim = provider.dimension
    if dim is None:
        probe = await provider.embed(["اختبار"], is_query=False)
        dim = len(probe[0]) if probe else 1536

    base_path = Path(base_articles_path)
    corpus_hash = compute_corpus_hash(base_path) if base_path.exists() else "dynamic"
    collection_name = build_collection_name(target_model_id, dim, corpus_hash)

    if vector_store is None:
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    # Ensure collection exists
    vector_store.create_collection_if_not_exists(collection_name, dimension=dim)

    # Chunk new articles
    chunk_strategy = (rag_config or {}).get("chunk_strategy", "article_paragraph")
    max_chars = (rag_config or {}).get("max_chars", 1200)
    overlap = (rag_config or {}).get("overlap", 100)

    chunks = chunk_corpus(
        new_articles,
        strategy=chunk_strategy,
        max_chars=max_chars,
        overlap=overlap,
    )

    # Embed and upsert
    content_mode = (rag_config or {}).get("embedding_content", "ar")
    vectors = await embed_chunks(
        chunks,
        provider=provider,
        content_mode=content_mode,
        batch_size=model_cfg.batch_size,
    )

    vector_store.upsert(collection_name, chunks=chunks, vectors=vectors)

    return {
        "status": "reindexed_successfully",
        "collection_name": collection_name,
        "new_articles_count": len(new_articles),
        "new_chunks_count": len(chunks),
    }
