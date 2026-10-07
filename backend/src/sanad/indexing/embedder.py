"""Chunk text embedding pipeline."""

import logging

from sanad.indexing.chunker import ChunkRecord
from sanad.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


def prepare_chunk_embedding_text(chunk: ChunkRecord, content_mode: str = "ar") -> str:
    """Format chunk text for embedding based on configured content mode."""
    if content_mode == "en":
        return chunk.text_en.strip()
    elif content_mode == "ar+en":
        return f"{chunk.text_ar_normalized}\n{chunk.text_en}".strip()
    # Default is normalized Arabic
    return chunk.text_ar_normalized.strip()


async def embed_chunks(
    chunks: list[ChunkRecord],
    provider: EmbeddingProvider,
    content_mode: str = "ar",
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed all chunks in batches using the specified embedding provider."""
    texts = [prepare_chunk_embedding_text(c, content_mode=content_mode) for c in chunks]
    return await provider.embed(texts, is_query=False)
