"""Legal Q&A endpoints supporting synchronous REST and Server-Sent Events (SSE)."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from sanad.api.deps import get_rag
from sanad.api.schemas import AskRequest, AskResponse
from sanad.rag import SanadRAG

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Legal Q&A"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    rag: SanadRAG = Depends(get_rag),
) -> AskResponse:
    """Submit a legal question and receive a grounded answer with citations."""
    resp = await rag.query(
        question=request.question,
        chat_model_id=request.chat_model,
        embedding_model_id=request.embedding_model,
        top_k=request.top_k,
        language_mode=request.language,
        enable_pii_masking=request.enable_pii_masking,
        enable_grounding=request.enable_grounding,
    )
    return AskResponse(
        answer=resp.answer,
        citations=resp.citations,
        retrieved_articles=resp.retrieved_articles,
        grounding=resp.grounding,
        usage=resp.usage,
        latency_ms=resp.latency_ms,
        chat_model_id=resp.chat_model_id,
        embedding_model_id=resp.embedding_model_id,
        trace_id=resp.trace_id,
        pii_redacted=resp.pii_redacted,
    )


@router.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    rag: SanadRAG = Depends(get_rag),
) -> EventSourceResponse:
    """Stream answer tokens and citation events via Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            async for ev in rag.stream_query(
                question=request.question,
                chat_model_id=request.chat_model,
                embedding_model_id=request.embedding_model,
                top_k=request.top_k,
                language_mode=request.language,
                enable_pii_masking=request.enable_pii_masking,
                enable_grounding=request.enable_grounding,
            ):
                payload_data = (
                    ev.data if isinstance(ev.data, str) else json.dumps(ev.data, ensure_ascii=False)
                )
                yield {
                    "event": ev.event,
                    "data": payload_data,
                }
        except Exception as e:
            logger.exception("Error in SSE stream generation: %s", e)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
