"""Local HuggingFace / SentenceTransformers embedding provider."""

import hashlib
import logging
from typing import Any

from sanad.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalHFEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers with offline fallback."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dimension: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._dimension = dimension
        self.batch_size = batch_size
        self._model: Any = None
        self._initialized = False

    def _init_model(self) -> None:
        if self._initialized:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            detected_dim = self._model.get_sentence_embedding_dimension()
            if self._dimension is None:
                self._dimension = detected_dim
        except ImportError:
            logger.warning(
                "sentence_transformers not installed. LocalHFEmbeddingProvider will run in deterministic fallback mode."
            )
            if self._dimension is None:
                self._dimension = 384
        except Exception as e:
            logger.warning(
                "Failed to load SentenceTransformer '%s': %s. Using deterministic fallback.",
                self.model_name,
                e,
            )
            if self._dimension is None:
                self._dimension = 384

    @property
    def dimension(self) -> int | None:
        if not self._initialized:
            self._init_model()
        return self._dimension

    async def embed(
        self,
        texts: list[str],
        is_query: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            return []

        if not self._initialized:
            self._init_model()

        if self._model is not None:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vectors.tolist()  # type: ignore[no-any-return]

        # Deterministic pseudo-embedding for testing/fallback
        dim = self._dimension or 384
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # Generate deterministic unit-normalized floats
            raw_floats = [(h[i % len(h)] / 255.0) - 0.5 for i in range(dim)]
            norm = (sum(x * x for x in raw_floats) or 1.0) ** 0.5
            results.append([x / norm for x in raw_floats])
        return results
