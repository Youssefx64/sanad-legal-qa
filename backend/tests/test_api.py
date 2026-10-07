"""Integration tests for all FastAPI application routes and middleware."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from sanad.api.app import create_app
from sanad.config.settings import Settings
from sanad.corpus.schema import ArticleRecord
from sanad.generation.answerer import LegalAnswerer
from sanad.indexing.chunker import chunk_article
from sanad.indexing.vector_store import QdrantVectorStore
from sanad.providers.base import (
    ChatMessage,
    ChatProvider,
    ChatResponse,
    ChatStreamChunk,
    ChatUsage,
)
from sanad.rag import SanadRAG
from sanad.retrieval.retriever import BM25Index


class MockChatProvider(ChatProvider):
    """Mock ChatProvider returning predetermined answers."""

    def __init__(self, answer_text: str = "العقد شريعة المتعاقدين [المادة 147]") -> None:
        self.answer_text = answer_text

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> ChatResponse:
        return ChatResponse(
            content=self.answer_text,
            usage=ChatUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            model="mock-chat-1",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ):
        words = self.answer_text.split()
        for w in words:
            yield ChatStreamChunk(content=w + " ", is_final=False)
        yield ChatStreamChunk(
            content="",
            is_final=True,
            usage=ChatUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
        )


@pytest.fixture
def test_client() -> TestClient:
    """Create a TestClient with mock RAG backend."""
    app = create_app()

    # Create mock article and vector store
    art147 = ArticleRecord(
        article_number=147,
        book="Obligations Generally",
        chapter="Contracts",
        text_ar="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
        text_ar_raw="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
        text_en="The contract makes the law of the parties.",
        is_repealed=False,
        source_page=34,
        citation="Egyptian Civil Code, Article 147",
        references=[],
    )
    chunks = chunk_article(art147)

    vstore = QdrantVectorStore(location=":memory:")
    col_name = "sanad_local-hf-embed-1_16_00000000"
    vstore.create_collection_if_not_exists(col_name, dimension=16)
    vstore.upsert(col_name, chunks, [[0.1] * 16])

    bm25 = BM25Index(chunks)

    rag = SanadRAG(
        vector_store=vstore,
        bm25_index=bm25,
        answerer=LegalAnswerer(),
    )
    # Inject mock chat provider
    rag.factory._chat_cache["openrouter-chat-1"] = MockChatProvider()

    app.state.rag = rag
    return TestClient(app)


def test_headers_middleware(test_client: TestClient) -> None:
    """Verify X-Request-ID and X-Response-Time-Ms headers are injected."""
    res = test_client.get("/health")
    assert res.status_code == 200
    assert "x-request-id" in res.headers
    assert "x-response-time-ms" in res.headers


def test_health_endpoint(test_client: TestClient) -> None:
    """Test /health endpoint returns status ok and vector db status."""
    res = test_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["qdrant"] is True
    assert data["models_loaded"] > 0
    assert data["version"] == "0.1.0"


def test_models_endpoint(test_client: TestClient) -> None:
    """Test /models endpoint returns chat and embedding models catalog."""
    res = test_client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "chat_models" in data
    assert "embedding_models" in data
    assert "defaults" in data
    assert len(data["chat_models"]) > 0


def test_corpus_article_endpoint(test_client: TestClient) -> None:
    """Test /articles/{number} endpoint for valid and invalid numbers."""
    # Valid article 147 (if data/processed/articles.json exists)
    if Path("data/processed/articles.json").exists():
        res147 = test_client.get("/articles/147")
        assert res147.status_code == 200
        data147 = res147.json()
        assert data147["article_number"] == 147
        assert "العقد" in data147["text_ar"]

    # Out of range 9999
    res_bad = test_client.get("/articles/9999")
    assert res_bad.status_code == 404

    # Out of range 0
    res_zero = test_client.get("/articles/0")
    assert res_zero.status_code == 404


def test_corpus_stats_endpoint(test_client: TestClient) -> None:
    """Test /corpus/stats endpoint returns article counts."""
    res = test_client.get("/corpus/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_articles" in data
    assert "active_articles" in data
    assert "repealed_articles" in data
    assert "books" in data


def test_ask_endpoint(test_client: TestClient) -> None:
    """Test POST /ask synchronous legal query."""
    payload = {
        "question": "ما هو نص المادة 147 من القانون المدني؟",
        "chat_model": "openrouter-chat-1",
        "embedding_model": "local-hf-embed-1",
        "top_k": 3,
        "language": "ar",
    }
    res = test_client.post("/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "citations" in data
    assert "grounding" in data
    assert data["grounding"]["is_grounded"] is True
    assert data["latency_ms"] > 0
    assert len(data["trace_id"]) > 0


def test_ask_stream_endpoint(test_client: TestClient) -> None:
    """Test POST /ask/stream SSE streaming events."""
    payload = {
        "question": "ما هو أثر العقد في المادة 147؟",
        "chat_model": "openrouter-chat-1",
        "embedding_model": "local-hf-embed-1",
    }
    with test_client.stream("POST", "/ask/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [
            line if isinstance(line, str) else line.decode("utf-8")
            for line in response.iter_lines()
            if line
        ]
        assert any(
            "token" in line_text or "citations" in line_text or "done" in line_text
            for line_text in lines
        )


def test_admin_ingest_auth(test_client: TestClient) -> None:
    """Test /admin/ingest requires valid X-Admin-Key."""
    payload = {"force": False}

    # Missing header -> 401
    res_no_auth = test_client.post("/admin/ingest", json=payload)
    assert res_no_auth.status_code == 401

    # Wrong header -> 401
    res_wrong_auth = test_client.post(
        "/admin/ingest",
        json=payload,
        headers={"X-Admin-Key": "wrong-secret"},
    )
    assert res_wrong_auth.status_code == 401

    # Valid header -> 200 (mocking rag.ingest)
    settings = Settings()
    test_client.app.state.rag.ingest = AsyncMock(
        return_value={
            "status": "indexed_successfully",
            "collection_name": "test_col",
            "chunks_count": 10,
            "model_id": "local-hf-embed-1",
            "dimension": 16,
        }
    )
    res_auth = test_client.post(
        "/admin/ingest",
        json=payload,
        headers={"X-Admin-Key": settings.admin_api_key},
    )
    assert res_auth.status_code == 200
    assert res_auth.json()["status"] == "indexed_successfully"


def test_metrics_endpoint(test_client: TestClient) -> None:
    """Test /metrics endpoint returns Prometheus metrics format."""
    res = test_client.get("/metrics")
    assert res.status_code == 200
    assert "sanad_requests_total" in res.text
