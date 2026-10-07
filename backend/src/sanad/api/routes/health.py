"""Health check endpoint."""

from fastapi import APIRouter, Depends

from sanad.api.deps import get_rag
from sanad.api.schemas import HealthResponse
from sanad.rag import SanadRAG

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag: SanadRAG = Depends(get_rag),
) -> HealthResponse:
    """Verify application health and connectivity to vector database."""
    qdrant_healthy = False
    try:
        # Check if vector store is initialized and queryable
        rag.vector_store.collection_exists("probe")
        qdrant_healthy = True
    except Exception:
        qdrant_healthy = False

    models_count = len(rag.registry.enabled_chat_models) + len(
        rag.registry.enabled_embedding_models
    )

    return HealthResponse(
        status="ok",
        qdrant=qdrant_healthy,
        models_loaded=models_count,
        version="0.1.0",
    )
