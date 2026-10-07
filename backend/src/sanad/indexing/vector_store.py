"""Vector store interface and Qdrant implementation."""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import qdrant_client
from pydantic import BaseModel
from qdrant_client.http import models as qmodels

from sanad.indexing.chunker import ChunkRecord

logger = logging.getLogger(__name__)


class ScoredChunk(BaseModel):
    """A retrieved chunk paired with its vector similarity score."""

    chunk: ChunkRecord
    score: float


def sanitize_collection_name(raw_name: str) -> str:
    """Sanitize string to valid Qdrant collection name (alphanumeric, dashes, underscores)."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_name)


def build_collection_name(model_id: str, dimension: int, corpus_hash: str) -> str:
    """Build standardized collection name combining model ID, dim, and corpus hash."""
    raw = f"sanad_{model_id}_{dimension}_{corpus_hash}"
    return sanitize_collection_name(raw)


class VectorStore(ABC):
    """Abstract interface for vector database storage and similarity search."""

    @abstractmethod
    def create_collection_if_not_exists(self, collection_name: str, dimension: int) -> None:
        """Create a collection if it does not already exist."""

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists."""

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of vectors in a collection."""

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
    ) -> None:
        """Upsert chunk records with their corresponding vector embeddings."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Perform cosine similarity vector search with optional metadata filters."""


class QdrantVectorStore(VectorStore):
    """Qdrant vector store supporting both containerized and in-memory instances."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
    ) -> None:
        """Initialize Qdrant client.

        If location==':memory:' or url is empty, initializes in-memory client.
        """
        if location == ":memory:" or not url:
            self.client = qdrant_client.QdrantClient(":memory:")
        else:
            self.client = qdrant_client.QdrantClient(url=url, api_key=api_key)

    def create_collection_if_not_exists(self, collection_name: str, dimension: int) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", collection_name, dimension)

    def collection_exists(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def count(self, collection_name: str) -> int:
        if not self.collection_exists(collection_name):
            return 0
        info = self.client.get_collection(collection_name)
        return info.points_count or 0

    def _generate_point_id(self, chunk_id: str) -> int:
        """Generate a deterministic 64-bit integer ID from string chunk_id for Qdrant."""
        return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:15], 16)

    def upsert(
        self,
        collection_name: str,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
    ) -> None:
        if not chunks or not vectors:
            return

        points: list[qmodels.PointStruct] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            point_id = self._generate_point_id(chunk.chunk_id)
            payload = chunk.model_dump()
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        # Upsert in batches of 250
        batch_size = 250
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        qfilter: qmodels.Filter | None = None
        if filter_criteria:
            conditions: list[qmodels.Condition] = []

            # Exclude repealed articles unless requested
            if filter_criteria.get("exclude_repealed", True):
                conditions.append(
                    qmodels.FieldCondition(
                        key="is_repealed",
                        match=qmodels.MatchValue(value=False),
                    )
                )

            # Filter by book
            if "book" in filter_criteria and filter_criteria["book"]:
                conditions.append(
                    qmodels.FieldCondition(
                        key="book",
                        match=qmodels.MatchValue(value=filter_criteria["book"]),
                    )
                )

            # Filter by chapter
            if "chapter" in filter_criteria and filter_criteria["chapter"]:
                conditions.append(
                    qmodels.FieldCondition(
                        key="chapter",
                        match=qmodels.MatchValue(value=filter_criteria["chapter"]),
                    )
                )

            # Filter by specific article numbers
            if "article_numbers" in filter_criteria and filter_criteria["article_numbers"]:
                conditions.append(
                    qmodels.FieldCondition(
                        key="article_number",
                        match=qmodels.MatchAny(any=filter_criteria["article_numbers"]),
                    )
                )

            if conditions:
                qfilter = qmodels.Filter(must=conditions)

        if hasattr(self.client, "search"):
            points = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qfilter,
            )
        else:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qfilter,
            )
            points = response.points

        scored: list[ScoredChunk] = []
        for res in points:
            if res.payload:
                chunk = ChunkRecord.model_validate(res.payload)
                scored.append(ScoredChunk(chunk=chunk, score=res.score))

        return scored
