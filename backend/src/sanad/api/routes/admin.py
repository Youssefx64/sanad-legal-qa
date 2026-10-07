"""Administrative endpoints for operations, re-indexing, and ingestion."""

from fastapi import APIRouter, Depends

from sanad.api.deps import get_rag, verify_admin_key
from sanad.api.schemas import AdminIngestRequest, AdminIngestResponse
from sanad.rag import SanadRAG

router = APIRouter(tags=["Admin"])


@router.post(
    "/admin/ingest",
    response_model=AdminIngestResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def trigger_ingestion(
    request: AdminIngestRequest,
    rag: SanadRAG = Depends(get_rag),
) -> AdminIngestResponse:
    """Trigger vector database ingestion (protected by X-Admin-Key)."""
    res = await rag.ingest(
        embedding_model_id=request.embedding_model,
        force=request.force,
    )
    return AdminIngestResponse(
        status=res.get("status", "completed"),
        collection_name=res.get("collection_name", ""),
        chunks_count=res.get("chunks_count", 0),
        model_id=res.get("model_id", ""),
        dimension=res.get("dimension"),
        articles_count=res.get("articles_count"),
    )

