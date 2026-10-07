"""Corpus ingestion pipeline into Qdrant vector database."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from sanad.config.registry import get_model_registry
from sanad.config.settings import get_settings
from sanad.corpus.schema import ArticleRecord
from sanad.indexing.chunker import chunk_corpus
from sanad.indexing.embedder import embed_chunks
from sanad.indexing.vector_store import (
    QdrantVectorStore,
    VectorStore,
    build_collection_name,
)
from sanad.providers.base import EmbeddingProvider
from sanad.providers.factory import get_provider_factory

logger = logging.getLogger(__name__)


def compute_corpus_hash(articles_path: str | Path) -> str:
    """Compute sha256 hash digest prefix of processed articles.json."""
    content = Path(articles_path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:8]


async def run_ingestion(
    articles_path: str | Path = Path("data/processed/articles.json"),
    embedding_model_id: str | None = None,
    vector_store: VectorStore | None = None,
    force: bool = False,
    rag_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute corpus chunking, embedding, and vector database ingestion.

    Idempotent: skips if target collection exists and is populated unless force=True.
    """
    path = Path(articles_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    corpus_hash = compute_corpus_hash(path)
    settings = get_settings()
    registry = get_model_registry()

    target_model_id = embedding_model_id or registry.defaults.embedding_model
    model_cfg = registry.get_embedding_model(target_model_id)

    factory = get_provider_factory()
    provider: EmbeddingProvider = factory.get_embedding_provider(target_model_id)

    # Auto-detect dimension if not yet known
    dim = provider.dimension
    if dim is None:
        logger.info("Detecting embedding dimension for model '%s'...", target_model_id)
        probe = await provider.embed(["اختبار الأبعاد"], is_query=False)
        dim = len(probe[0]) if probe else 1536

    collection_name = build_collection_name(target_model_id, dim, corpus_hash)

    # Initialize vector store if not injected
    if vector_store is None:
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    # Check idempotency
    if not force and vector_store.collection_exists(collection_name):
        existing_count = vector_store.count(collection_name)
        if existing_count > 0:
            logger.info(
                "Collection '%s' already indexed with %d chunks. Skipping (idempotent).",
                collection_name,
                existing_count,
            )
            return {
                "collection_name": collection_name,
                "model_id": target_model_id,
                "dimension": dim,
                "corpus_hash": corpus_hash,
                "chunks_count": existing_count,
                "status": "skipped_already_exists",
            }

    # Load articles
    with path.open("r", encoding="utf-8") as f:
        articles_data = json.load(f)
    articles = [ArticleRecord.model_validate(item) for item in articles_data]

    # Chunk corpus
    chunk_strategy = (rag_config or {}).get("chunk_strategy", "article_paragraph")
    max_chars = (rag_config or {}).get("max_chars", 1200)
    overlap = (rag_config or {}).get("overlap", 100)

    chunks = chunk_corpus(
        articles,
        strategy=chunk_strategy,
        max_chars=max_chars,
        overlap=overlap,
    )
    logger.info("Chunked %d articles into %d chunks.", len(articles), len(chunks))

    # Embed chunks
    content_mode = (rag_config or {}).get("embedding_content", "ar")
    logger.info(
        "Computing embeddings for %d chunks with model '%s'...", len(chunks), target_model_id
    )
    vectors = await embed_chunks(
        chunks,
        provider=provider,
        content_mode=content_mode,
        batch_size=model_cfg.batch_size,
    )

    # Create collection and upsert
    vector_store.create_collection_if_not_exists(collection_name, dimension=dim)
    vector_store.upsert(collection_name, chunks=chunks, vectors=vectors)
    logger.info("Upserted %d vectors into collection '%s'.", len(vectors), collection_name)

    return {
        "collection_name": collection_name,
        "model_id": target_model_id,
        "dimension": dim,
        "corpus_hash": corpus_hash,
        "chunks_count": len(chunks),
        "articles_count": len(articles),
        "status": "indexed_successfully",
    }
