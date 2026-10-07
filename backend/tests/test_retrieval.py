"""Unit and integration tests for providers, chunking, indexing, and hybrid retrieval."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from sanad.corpus.schema import ArticleRecord
from sanad.indexing.chunker import chunk_article, chunk_corpus
from sanad.indexing.embedder import prepare_chunk_embedding_text
from sanad.indexing.ingest import run_ingestion
from sanad.indexing.vector_store import (
    QdrantVectorStore,
)
from sanad.providers.base import ChatMessage
from sanad.providers.local_hf import LocalHFEmbeddingProvider
from sanad.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from sanad.retrieval.filters import detect_query_filters
from sanad.retrieval.reranker import NoOpReranker
from sanad.retrieval.retriever import BM25Index, HybridRetriever


@pytest.fixture
def sample_articles() -> list[ArticleRecord]:
    """Create a small set of mock articles for testing."""
    return [
        ArticleRecord(
            article_number=147,
            book="Obligations Generally",
            chapter="Contracts",
            text_ar="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين، أو للأسباب التي يقررها القانون.",
            text_ar_raw="العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين، أو للأسباب التي يقررها القانون.",
            text_en="The contract makes the law of the parties. It cannot be revoked or altered except by mutual consent or for causes provided by law.",
            is_repealed=False,
            source_page=34,
            citation="Egyptian Civil Code, Article 147",
            references=[148],
        ),
        ArticleRecord(
            article_number=148,
            book="Obligations Generally",
            chapter="Contracts",
            text_ar="(1) يجب تنفيذ العقد طبقا لما اشتمل عليه وبطريقة تتفق مع ما يوجبه حسن النية.\n(2) ولا يقتصر العقد على إلزام المتعاقد بما ورد فيه فحسب، بل يتناول أيضا ما هو من مستلزماته.",
            text_ar_raw="(1) يجب تنفيذ العقد طبقا لما اشتمل عليه وبطريقة تتفق مع ما يوجبه حسن النية.\n(2) ولا يقتصر العقد على إلزام المتعاقد بما ورد فيه فحسب، بل يتناول أيضا ما هو من مستلزماته.",
            text_en="(1) A contract must be performed in accordance with its contents and in good faith.\n(2) A contract binds the contracting party not only as to its contents, but also as to everything which is an essential consequence.",
            is_repealed=False,
            source_page=34,
            citation="Egyptian Civil Code, Article 148",
            references=[147],
        ),
        ArticleRecord(
            article_number=54,
            book="General Provisions",
            chapter="General",
            text_ar="ألغيت المادة 54 بالقرار الجمهوري بالقانون رقم 384 لسنة 1956.",
            text_ar_raw="ألغيت المادة 54 بالقرار الجمهوري بالقانون رقم 384 لسنة 1956.",
            text_en="Article 54 was repealed by Presidential Decree Law 384 of 1956.",
            is_repealed=True,
            source_page=8,
            citation="Egyptian Civil Code, Article 54 (Repealed)",
            references=[],
        ),
    ]


@pytest.mark.asyncio
@respx.mock
async def test_openai_compatible_chat_complete() -> None:
    """Test OpenAI-compatible chat completion."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "العقد شريعة المتعاقدين"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            },
        )
    )

    provider = OpenAICompatibleChatProvider(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_name="test-model",
    )

    resp = await provider.complete([ChatMessage(role="user", content="ما هي المادة 147؟")])
    assert resp.content == "العقد شريعة المتعاقدين"
    assert resp.model == "test-model"
    assert resp.usage.total_tokens == 23


@pytest.mark.asyncio
@respx.mock
async def test_openai_compatible_chat_stream() -> None:
    """Test OpenAI-compatible SSE streaming completion."""
    sse_lines = (
        'data: {"choices":[{"delta":{"content":"العقد"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" شريعة"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=sse_lines,
            headers={"content-type": "text/event-stream"},
        )
    )

    provider = OpenAICompatibleChatProvider(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_name="test-model",
    )

    chunks = []
    async for chunk in provider.stream([ChatMessage(role="user", content="اختبار")]):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].content == "العقد"
    assert chunks[1].content == " شريعة"
    assert chunks[2].is_final is True


@pytest.mark.asyncio
@respx.mock
async def test_openai_compatible_embedding() -> None:
    """Test OpenAI-compatible embedding provider."""
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0},
                    {"embedding": [0.5, 0.6, 0.7, 0.8], "index": 1},
                ]
            },
        )
    )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_name="text-embedding-3-small",
    )

    embeddings = await provider.embed(["نص 1", "نص 2"])
    assert len(embeddings) == 2
    assert provider.dimension == 4
    assert embeddings[0] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_local_hf_embedding_deterministic() -> None:
    """Test local HF embedding provider deterministic mode."""
    provider = LocalHFEmbeddingProvider(model_name="test-embedder", dimension=64)
    assert provider.dimension == 64

    vecs = await provider.embed(["العقد شريعة المتعاقدين", "القانون المدني"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 64
    # Check normalized vector
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_chunking_strategies(sample_articles: list[ArticleRecord]) -> None:
    """Test article-level and paragraph-level chunking."""
    art147 = sample_articles[0]
    art148 = sample_articles[1]

    # Article 147 is short -> 1 chunk
    chunks_147 = chunk_article(art147, strategy="article_paragraph", max_chars=1200)
    assert len(chunks_147) == 1
    assert chunks_147[0].chunk_id == "art_147"
    assert chunks_147[0].article_number == 147

    # Article 148 has paragraphs (1) and (2) when threshold is low
    chunks_148 = chunk_article(art148, strategy="article_paragraph", max_chars=50)
    assert len(chunks_148) == 2
    assert chunks_148[0].chunk_id == "art_148_p1"
    assert chunks_148[0].paragraph_number == 1
    assert chunks_148[1].chunk_id == "art_148_p2"
    assert chunks_148[1].paragraph_number == 2

    # Chunk corpus
    all_chunks = chunk_corpus(sample_articles, strategy="article")
    assert len(all_chunks) == len(sample_articles)


def test_prepare_chunk_embedding_text(sample_articles: list[ArticleRecord]) -> None:
    """Test formatting chunk text for different embedding content modes."""
    chunks = chunk_article(sample_articles[0])
    chunk = chunks[0]

    assert prepare_chunk_embedding_text(chunk, "ar") == chunk.text_ar_normalized
    assert prepare_chunk_embedding_text(chunk, "en") == chunk.text_en
    assert "العقد" in prepare_chunk_embedding_text(chunk, "ar+en")
    assert "contract" in prepare_chunk_embedding_text(chunk, "ar+en").lower()


def test_detect_query_filters() -> None:
    """Test query intent analysis and explicit article detection."""
    # Explicit single article
    f1, arts1 = detect_query_filters("ما نص المادة 147 من القانون؟")
    assert arts1 == [147]
    assert f1.exclude_repealed is True

    # Explicit multiple articles
    f2, arts2 = detect_query_filters("ما الفرق بين المادتين 147 و 148؟")
    assert 147 in arts2
    assert 148 in arts2

    # Query asking about repealed article
    f3, arts3 = detect_query_filters("هل المادة 54 ملغاة؟")
    assert 54 in arts3
    assert f3.exclude_repealed is False

    # English query with Article keyword
    f4, arts4 = detect_query_filters("What does Article 219 specify?")
    assert arts4 == [219]


def test_bm25_index(sample_articles: list[ArticleRecord]) -> None:
    """Test BM25 search over legal chunks."""
    chunks = chunk_corpus(sample_articles, strategy="article")
    bm25 = BM25Index(chunks)

    # Search for "شريعة المتعاقدين"
    results = bm25.search("شريعة المتعاقدين", limit=5)
    assert len(results) > 0
    assert results[0].chunk.article_number == 147

    # Test exact chunk retrieval by article number
    art148_chunks = bm25.get_chunks_for_articles([148])
    assert len(art148_chunks) == 1
    assert art148_chunks[0].article_number == 148


@pytest.mark.asyncio
async def test_qdrant_vector_store_in_memory(sample_articles: list[ArticleRecord]) -> None:
    """Test Qdrant in-memory vector store upsert and search with filters."""
    chunks = chunk_corpus(sample_articles, strategy="article")
    provider = LocalHFEmbeddingProvider(model_name="test", dimension=16)
    vecs = await provider.embed([c.text_ar_normalized for c in chunks])

    vstore = QdrantVectorStore(location=":memory:")
    col_name = "test_col"
    vstore.create_collection_if_not_exists(col_name, dimension=16)

    vstore.upsert(col_name, chunks, vecs)
    assert vstore.count(col_name) == len(chunks)

    # Search without filter
    q_vec = vecs[0]
    hits = vstore.search(col_name, q_vec, limit=5, filter_criteria={"exclude_repealed": False})
    assert len(hits) == len(chunks)
    assert hits[0].chunk.article_number == 147

    # Search with repealed exclusion filter
    hits_filtered = vstore.search(
        col_name, q_vec, limit=5, filter_criteria={"exclude_repealed": True}
    )
    assert all(not h.chunk.is_repealed for h in hits_filtered)


@pytest.mark.asyncio
async def test_hybrid_retriever_exact_article_boost(sample_articles: list[ArticleRecord]) -> None:
    """Test that explicit article mentions are deterministically ranked at rank 1."""
    chunks = chunk_corpus(sample_articles, strategy="article")
    provider = LocalHFEmbeddingProvider(model_name="test", dimension=16)
    vecs = await provider.embed([c.text_ar_normalized for c in chunks])

    vstore = QdrantVectorStore(location=":memory:")
    col_name = "test_legal_qa"
    vstore.create_collection_if_not_exists(col_name, dimension=16)
    vstore.upsert(col_name, chunks, vecs)

    bm25 = BM25Index(chunks)
    retriever = HybridRetriever(
        vector_store=vstore,
        embedding_provider=provider,
        collection_name=col_name,
        bm25_index=bm25,
        reranker=NoOpReranker(),
        top_k=3,
        enable_exact_lookup=True,
    )

    # Query explicitly mentioning Article 147
    results = await retriever.retrieve("ما هو نص المادة 147؟")
    assert len(results) > 0
    assert results[0].chunk.article_number == 147
    assert results[0].source == "exact_lookup"

    # Query mentioning Article 148
    results_148 = await retriever.retrieve("Article 148 good faith performance")
    assert len(results_148) > 0
    assert results_148[0].chunk.article_number == 148
    assert results_148[0].source == "exact_lookup"


@pytest.mark.asyncio
async def test_ingest_pipeline_idempotency(
    tmp_path: Path, sample_articles: list[ArticleRecord]
) -> None:
    """Test the ingestion pipeline creates collection and runs idempotently."""
    articles_file = tmp_path / "articles.json"
    articles_file.write_text(
        json.dumps([a.model_dump() for a in sample_articles], ensure_ascii=False),
        encoding="utf-8",
    )

    vstore = QdrantVectorStore(location=":memory:")

    # First run: should index successfully
    res1 = await run_ingestion(
        articles_path=articles_file,
        embedding_model_id="local-hf-embed-1",
        vector_store=vstore,
        force=False,
    )
    assert res1["status"] == "indexed_successfully"
    assert res1["chunks_count"] == len(sample_articles)

    # Second run: should detect existing collection and skip
    res2 = await run_ingestion(
        articles_path=articles_file,
        embedding_model_id="local-hf-embed-1",
        vector_store=vstore,
        force=False,
    )
    assert res2["status"] == "skipped_already_exists"

    # Forced run: should re-index
    res3 = await run_ingestion(
        articles_path=articles_file,
        embedding_model_id="local-hf-embed-1",
        vector_store=vstore,
        force=True,
    )
    assert res3["status"] == "indexed_successfully"


def test_cli_ingest(
    tmp_path: Path, sample_articles: list[ArticleRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test sanad ingest CLI command."""
    from typer.testing import CliRunner

    import sanad.indexing.ingest
    from sanad.cli import app

    articles_file = tmp_path / "articles.json"
    articles_file.write_text(
        json.dumps([a.model_dump() for a in sample_articles], ensure_ascii=False),
        encoding="utf-8",
    )

    # Mock QdrantVectorStore in ingest to use in-memory store
    mem_store = QdrantVectorStore(location=":memory:")
    original_run_ingest = sanad.indexing.ingest.run_ingestion

    async def mock_run_ingestion(*args, **kwargs):
        kwargs["vector_store"] = mem_store
        return await original_run_ingest(*args, **kwargs)

    monkeypatch.setattr(sanad.indexing.ingest, "run_ingestion", mock_run_ingestion)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "--embedding-model",
            "local-hf-embed-1",
            "--articles-path",
            str(articles_file),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Ingestion indexed_successfully" in result.stdout
