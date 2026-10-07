"""Unit and integration tests for generation, prompt formatting, citations, and guardrails."""

from collections.abc import AsyncIterator

import pytest

from sanad.corpus.schema import ArticleRecord
from sanad.generation.answerer import LegalAnswerer
from sanad.generation.citations import extract_citations
from sanad.generation.prompts import (
    REFUSAL_AR,
    REFUSAL_EN,
    build_rag_messages,
    is_arabic_text,
)
from sanad.guardrails.grounding import check_grounding, enforce_grounding_guardrail
from sanad.guardrails.pii import redact_pii
from sanad.indexing.chunker import chunk_article
from sanad.providers.base import (
    ChatMessage,
    ChatProvider,
    ChatResponse,
    ChatStreamChunk,
    ChatUsage,
)
from sanad.rag import SanadRAG
from sanad.retrieval.reranker import SearchResult


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
            usage=ChatUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            model="mock-llm",
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ChatStreamChunk]:
        words = self.answer_text.split()
        for w in words:
            yield ChatStreamChunk(content=w + " ", is_final=False)
        yield ChatStreamChunk(
            content="",
            is_final=True,
            usage=ChatUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )


@pytest.fixture
def mock_retrieved_chunks() -> list[SearchResult]:
    """Fixture returning sample retrieved search results for Article 147."""
    article = ArticleRecord(
        article_number=147,
        book="Obligations Generally",
        chapter="Contracts",
        text_ar="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
        text_ar_raw="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
        text_en="The contract makes the law of the parties. It cannot be revoked or altered except by mutual consent.",
        is_repealed=False,
        source_page=34,
        citation="Egyptian Civil Code, Article 147",
        references=[],
    )
    chunks = chunk_article(article)
    return [SearchResult(chunk=chunks[0], score=0.95, source="dense")]


def test_language_detection() -> None:
    """Test Arabic vs English language detection."""
    assert is_arabic_text("ما هو العقد؟") is True
    assert is_arabic_text("What is a contract in Egyptian Law?") is False
    assert is_arabic_text("147") is True  # Default to Arabic


def test_prompt_construction(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test system prompt and user message generation."""
    msgs = build_rag_messages("ما هو نص المادة 147؟", mock_retrieved_chunks, language_mode="ar")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert "سند" in msgs[0].content
    assert "[المادة N]" in msgs[0].content
    assert msgs[1].role == "user"
    assert "147" in msgs[1].content
    assert "العقد شريعة المتعاقدين" in msgs[1].content


def test_citations_extraction(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test extracting citations and verifying groundedness."""
    answer = "تنص المادة على أن العقد شريعة المتعاقدين [المادة 147]، بينما تنص مادة أخرى [المادة 999] على أمر مختلف."
    citations = extract_citations(answer, mock_retrieved_chunks)

    assert len(citations) == 2
    c147 = next(c for c in citations if c.article_number == 147)
    assert c147.is_grounded is True
    assert "Egyptian Civil Code, Article 147" in c147.citation
    assert len(c147.text_ar_snippet) > 0

    c999 = next(c for c in citations if c.article_number == 999)
    assert c999.is_grounded is False


def test_pii_redaction() -> None:
    """Test redaction of sensitive personal data."""
    text = (
        "موكلي رقمه القومي 29501011234567 ورقمه التليفوني 01012345678 "
        "وبريده الإلكتروني test.lawyer@example.com ورقم حسابه EG123456789012345678901234567."
    )
    redacted, items = redact_pii(text)
    assert "[NATIONAL_ID_REDACTED]" in redacted
    assert "[PHONE_REDACTED]" in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "[IBAN_REDACTED]" in redacted
    assert "29501011234567" not in redacted
    assert "01012345678" not in redacted
    assert len(items) == 4


def test_grounding_guardrail_success(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test grounded answer passes check."""
    good_answer = "العقد شريعة المتعاقدين ولا يجوز تعديله إلا باتفاق الطرفين [المادة 147]."
    res = check_grounding(good_answer, mock_retrieved_chunks)
    assert res.is_grounded is True
    assert res.grounded_citations == [147]
    assert res.hallucinated_citations == []


def test_grounding_guardrail_catches_hallucinated_citation(
    mock_retrieved_chunks: list[SearchResult],
) -> None:
    """Test grounding catches unretrieved article citations."""
    bad_answer = "ينص القانون على ذلك في [المادة 999]."
    res = check_grounding(bad_answer, mock_retrieved_chunks)
    assert res.is_grounded is False
    assert 999 in res.hallucinated_citations

    # Enforce guardrail replaces with canonical refusal
    safe_answer, guard_res = enforce_grounding_guardrail(
        bad_answer, mock_retrieved_chunks, is_arabic=True
    )
    assert safe_answer == REFUSAL_AR
    assert guard_res.is_grounded is False


def test_grounding_guardrail_allows_refusal(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test that canonical refusal string passes grounding check."""
    res_ar = check_grounding(REFUSAL_AR, mock_retrieved_chunks)
    assert res_ar.is_grounded is True

    res_en = check_grounding(REFUSAL_EN, mock_retrieved_chunks)
    assert res_en.is_grounded is True


@pytest.mark.asyncio
async def test_legal_answerer_generation(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test Answerer end-to-end generation."""
    answerer = LegalAnswerer()
    mock_provider = MockChatProvider(answer_text="العقد شريعة المتعاقدين [المادة 147].")

    result = await answerer.generate(
        question="ما هو أثر العقد؟",
        retrieved_chunks=mock_retrieved_chunks,
        provider=mock_provider,
        language_mode="ar",
    )
    assert "العقد شريعة المتعاقدين" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].article_number == 147
    assert result.grounding.is_grounded is True


@pytest.mark.asyncio
async def test_legal_answerer_empty_context_immediate_refusal() -> None:
    """Test Answerer refuses immediately on empty context without querying provider."""
    answerer = LegalAnswerer()
    # Provider that would have hallucinated
    mock_provider = MockChatProvider(answer_text="إجابة مختلقة")

    result = await answerer.generate(
        question="سؤال خارج القانون المدني",
        retrieved_chunks=[],
        provider=mock_provider,
        language_mode="ar",
    )
    assert result.answer == REFUSAL_AR
    assert result.model == "guardrail-fallback"
    assert result.grounding.is_grounded is True


@pytest.mark.asyncio
async def test_legal_answerer_streaming(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test Answerer streaming token and citation events."""
    answerer = LegalAnswerer()
    mock_provider = MockChatProvider(answer_text="العقد شريعة المتعاقدين [المادة 147]")

    events = []
    async for ev in answerer.stream_generate(
        question="ما أثر العقد؟",
        retrieved_chunks=mock_retrieved_chunks,
        provider=mock_provider,
        language_mode="ar",
    ):
        events.append(ev)

    event_types = [e.event for e in events]
    assert "token" in event_types
    assert "citations" in event_types
    assert "done" in event_types


@pytest.mark.asyncio
async def test_sanad_rag_facade_end_to_end(mock_retrieved_chunks: list[SearchResult]) -> None:
    """Test SanadRAG facade end-to-end with PII masking, retrieval, and answering."""
    from sanad.indexing.vector_store import QdrantVectorStore
    from sanad.retrieval.retriever import BM25Index

    # Setup in-memory vector store with chunk for article 147
    vstore = QdrantVectorStore(location=":memory:")
    chunk = mock_retrieved_chunks[0].chunk
    col_name = "sanad_local-hf-embed-1_16_00000000"
    vstore.create_collection_if_not_exists(col_name, dimension=16)
    vstore.upsert(col_name, [chunk], [[0.1] * 16])

    bm25 = BM25Index([chunk])
    rag = SanadRAG(
        vector_store=vstore,
        bm25_index=bm25,
        answerer=LegalAnswerer(),
    )

    # Mock provider in factory
    mock_chat = MockChatProvider(answer_text="وفقاً للقانون: العقد شريعة المتعاقدين [المادة 147]")
    rag.factory._chat_cache["openrouter-chat-1"] = mock_chat

    # Query with PII inside question
    query_text = "موكلي صاحب الرقم القومي 29501011234567 يسأل: ما نص المادة 147؟"
    resp = await rag.query(
        question=query_text,
        chat_model_id="openrouter-chat-1",
        embedding_model_id="local-hf-embed-1",
        enable_pii_masking=True,
    )

    assert resp.pii_redacted is True
    assert "المادة 147" in resp.answer
    assert resp.grounding.is_grounded is True
    assert len(resp.citations) == 1
    assert resp.citations[0].article_number == 147
    assert resp.latency_ms > 0
    assert resp.trace_id is not None

    # Test stream_query
    stream_events = []
    async for ev in rag.stream_query(
        question=query_text,
        chat_model_id="openrouter-chat-1",
        embedding_model_id="local-hf-embed-1",
    ):
        stream_events.append(ev)

    assert len(stream_events) > 0
    assert any(e.event == "token" for e in stream_events)
    assert any(e.event == "done" for e in stream_events)
