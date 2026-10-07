"""Pydantic request and response schemas for the FastAPI application."""

from typing import Any

from pydantic import BaseModel, Field

from sanad.generation.citations import CitationItem
from sanad.guardrails.grounding import GroundingCheckResult
from sanad.providers.base import ChatUsage
from sanad.retrieval.reranker import SearchResult


class AskRequest(BaseModel):
    """Payload for submitting a question to Sanad Legal Q&A."""

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="Legal question about the Egyptian Civil Code",
        examples=["ما هو مبدأ العقد شريعة المتعاقدين في القانون المدني؟"],
    )
    chat_model: str | None = Field(
        default=None,
        description="Override chat model ID from models.yaml",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Override embedding model ID from models.yaml",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant articles/paragraphs to retrieve",
    )
    language: str = Field(
        default="auto",
        pattern="^(auto|ar|en)$",
        description="Response language mode: auto, ar, or en",
    )
    enable_pii_masking: bool = Field(
        default=True,
        description="Scrub Egyptian national IDs, phone numbers, and emails",
    )
    enable_grounding: bool = Field(
        default=True,
        description="Enforce citation grounding guardrail",
    )


class AskResponse(BaseModel):
    """Complete structured response for a legal Q&A query."""

    answer: str
    citations: list[CitationItem]
    retrieved_articles: list[SearchResult]
    grounding: GroundingCheckResult
    usage: ChatUsage
    latency_ms: float
    chat_model_id: str
    embedding_model_id: str
    trace_id: str
    pii_redacted: bool


class HealthResponse(BaseModel):
    """System health check status."""

    status: str = "ok"
    qdrant: bool
    models_loaded: int
    version: str = "0.1.0"


class ModelsResponse(BaseModel):
    """Model registry catalog and current defaults."""

    defaults: dict[str, str]
    chat_models: list[dict[str, Any]]
    embedding_models: list[dict[str, Any]]


class CorpusStatsResponse(BaseModel):
    """Statistics and structure of the loaded Egyptian Civil Code."""

    total_articles: int
    active_articles: int
    repealed_articles: int
    books: list[str]


class AdminIngestRequest(BaseModel):
    """Payload to trigger vector database ingestion."""

    embedding_model: str | None = None
    force: bool = False


class AdminIngestResponse(BaseModel):
    """Result of an administrative ingestion trigger."""

    status: str
    collection_name: str
    chunks_count: int
    model_id: str
    dimension: int | None = None
    articles_count: int | None = None

