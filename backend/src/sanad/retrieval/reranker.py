"""Reranking interfaces and implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from sanad.indexing.chunker import ChunkRecord

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Retrieved chunk paired with ranking score and retrieval provenance."""

    chunk: ChunkRecord
    score: float
    source: str = "dense"  # "dense" | "bm25" | "exact_lookup" | "hybrid" | "reranked"


class Reranker(ABC):
    """Abstract interface for passage rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank candidates based on query relevance and return top_k."""


class NoOpReranker(Reranker):
    """Pass-through reranker that preserves existing candidate ordering."""

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        return candidates[:top_k]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using local neural model with graceful fallback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._init_done = False

    def _load_model(self) -> None:
        if self._init_done:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Loaded CrossEncoder model: %s", self.model_name)
        except Exception as e:
            logger.warning(
                "Could not load CrossEncoder '%s': %s. Falling back to NoOp.", self.model_name, e
            )

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]

        self._load_model()
        if self._model is None:
            return candidates[:top_k]

        # Form query-document pairs
        pairs = [(query, c.chunk.text_ar_normalized or c.chunk.text_en) for c in candidates]
        scores = self._model.predict(pairs)

        reranked = [
            SearchResult(
                chunk=c.chunk,
                score=float(score),
                source="reranked",
            )
            for c, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


def build_reranker(reranker_type: str = "none", model_name: str | None = None) -> Reranker:
    """Build a Reranker instance based on type."""
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return NoOpReranker()
