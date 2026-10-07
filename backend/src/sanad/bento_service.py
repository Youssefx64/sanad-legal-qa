"""BentoML deployment service packaging the Sanad RAG engine."""

import logging
from typing import Any

from sanad.rag import SanadRAG

logger = logging.getLogger(__name__)

try:
    import bentoml

    @bentoml.service(
        name="sanad_legal_qa",
        resources={"cpu": "2"},
    )
    class SanadService:
        """BentoML service packaging Sanad legal question answering pipeline."""

        def __init__(self) -> None:
            self.rag = SanadRAG()
            logger.info("Initialized SanadService within BentoML container.")

        @bentoml.api
        async def ask(
            self,
            question: str,
            chat_model: str = "",
            embedding_model: str = "",
            top_k: int = 5,
        ) -> dict[str, Any]:
            """Ask a legal question over the Egyptian Civil Code."""
            response = await self.rag.query(
                question=question,
                chat_model_id=chat_model or None,
                embedding_model_id=embedding_model or None,
                top_k=top_k,
            )
            return response.model_dump()

        @bentoml.api
        def health(self) -> dict[str, str]:
            """Service health check."""
            return {"status": "ok", "service": "sanad_legal_qa"}

except ImportError:
    # Graceful fallback when bentoml is not installed
    class SanadService:  # type: ignore[no-redef]
        """Fallback SanadService when bentoml package is not installed."""

        def __init__(self) -> None:
            self.rag = SanadRAG()

        async def ask(self, question: str, **kwargs: Any) -> dict[str, Any]:
            resp = await self.rag.query(question=question)
            return resp.model_dump()

        def health(self) -> dict[str, str]:
            return {"status": "ok", "service": "sanad_legal_qa"}
