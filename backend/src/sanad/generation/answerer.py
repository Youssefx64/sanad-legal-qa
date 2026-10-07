"""Legal answer generation orchestrator with citations and grounding guardrails."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from sanad.generation.citations import CitationItem, extract_citations
from sanad.generation.prompts import (
    REFUSAL_AR,
    REFUSAL_EN,
    build_rag_messages,
    is_arabic_text,
)
from sanad.guardrails.grounding import (
    GroundingCheckResult,
    enforce_grounding_guardrail,
)
from sanad.providers.base import ChatProvider, ChatUsage
from sanad.retrieval.reranker import SearchResult

logger = logging.getLogger(__name__)


class AnswerResult(BaseModel):
    """Generated legal answer with citations and verification metadata."""

    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    grounding: GroundingCheckResult
    usage: ChatUsage
    model: str


class StreamEvent(BaseModel):
    """Streaming chunk event emitted during answer generation."""

    event: str  # "token" | "citations" | "done"
    data: str | list[dict[str, Any]] | dict[str, Any]


def build_statutory_synthesis(
    retrieved_chunks: list[SearchResult],
    is_ar: bool,
    refusal_message_ar: str,
    refusal_message_en: str,
) -> str:
    """Build a grounded multi-article synthesis when remote LLM provider is unavailable."""
    if not retrieved_chunks:
        return refusal_message_ar if is_ar else refusal_message_en

    seen_articles = set()
    distinct_chunks = []
    for sc in retrieved_chunks:
        if sc.chunk.article_number not in seen_articles:
            seen_articles.add(sc.chunk.article_number)
            distinct_chunks.append(sc.chunk)

    if not distinct_chunks:
        return refusal_message_ar if is_ar else refusal_message_en

    if is_ar:
        if len(distinct_chunks) == 1:
            c = distinct_chunks[0]
            return (
                f"وفقاً لأحكام [المادة {c.article_number}] من القانون المدني المصري:\n\n{c.text_ar}"
            )

        sections = ["وفقاً لأحكام القانون المدني المصري المسترجعة ذات الصلة:"]
        for c in distinct_chunks[:4]:
            sections.append(f"\n\n• **[المادة {c.article_number}]**:\n{c.text_ar}")
        return "".join(sections)
    else:
        if len(distinct_chunks) == 1:
            c = distinct_chunks[0]
            return f"According to [Article {c.article_number}] of the Egyptian Civil Code:\n\n{c.text_en}"

        sections = ["According to the relevant provisions of the Egyptian Civil Code:"]
        for c in distinct_chunks[:4]:
            sections.append(f"\n\n• **[Article {c.article_number}]**:\n{c.text_en}")
        return "".join(sections)


class LegalAnswerer:
    """Orchestrates LLM generation, citation linking, and grounding verification."""

    def __init__(
        self,
        refusal_message_ar: str = REFUSAL_AR,
        refusal_message_en: str = REFUSAL_EN,
        grounding_threshold: float = 0.25,
    ) -> None:
        self.refusal_message_ar = refusal_message_ar
        self.refusal_message_en = refusal_message_en
        self.grounding_threshold = grounding_threshold

    async def generate(
        self,
        question: str,
        retrieved_chunks: list[SearchResult],
        provider: ChatProvider,
        language_mode: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_grounding: bool = True,
    ) -> AnswerResult:
        """Generate a complete legal answer with verified citations."""
        is_ar = (
            True
            if language_mode == "ar"
            else (False if language_mode == "en" else is_arabic_text(question))
        )

        # If no chunks were retrieved at all, refuse immediately without LLM call
        if not retrieved_chunks:
            refusal = self.refusal_message_ar if is_ar else self.refusal_message_en
            return AnswerResult(
                answer=refusal,
                citations=[],
                grounding=GroundingCheckResult(
                    is_grounded=True,
                    confidence_score=1.0,
                    reason="Refused due to empty retrieval context",
                ),
                usage=ChatUsage(),
                model="guardrail-fallback",
            )

        messages = build_rag_messages(
            question=question,
            retrieved_chunks=retrieved_chunks,
            language_mode=language_mode,
        )

        try:
            resp = await provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_answer = resp.content
            usage = resp.usage
            model_name = resp.model
        except Exception as e:
            logger.warning(
                "Chat provider completion failed (%s); generating deterministic synthesis from retrieved articles.",
                e,
            )
            raw_answer = build_statutory_synthesis(
                retrieved_chunks, is_ar, self.refusal_message_ar, self.refusal_message_en
            )
            usage = ChatUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80)
            model_name = "offline-grounded-fallback"

        final_answer = raw_answer
        grounding_res = GroundingCheckResult(is_grounded=True, reason="Unguarded")

        if enable_grounding:
            final_answer, grounding_res = enforce_grounding_guardrail(
                answer=raw_answer,
                retrieved_chunks=retrieved_chunks,
                refusal_message_ar=self.refusal_message_ar,
                refusal_message_en=self.refusal_message_en,
                is_arabic=is_ar,
                overlap_threshold=self.grounding_threshold,
            )

        citations = extract_citations(final_answer, retrieved_chunks)

        return AnswerResult(
            answer=final_answer,
            citations=citations,
            grounding=grounding_res,
            usage=usage,
            model=model_name,
        )

    async def stream_generate(
        self,
        question: str,
        retrieved_chunks: list[SearchResult],
        provider: ChatProvider,
        language_mode: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_grounding: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream token chunks and emit final citations event upon completion."""
        is_ar = (
            True
            if language_mode == "ar"
            else (False if language_mode == "en" else is_arabic_text(question))
        )

        # Empty context refusal fallback
        if not retrieved_chunks:
            refusal = self.refusal_message_ar if is_ar else self.refusal_message_en
            yield StreamEvent(event="token", data=refusal)
            yield StreamEvent(
                event="done",
                data={
                    "is_grounded": True,
                    "citations": [],
                    "reason": "Refused due to empty retrieval context",
                },
            )
            return

        messages = build_rag_messages(
            question=question,
            retrieved_chunks=retrieved_chunks,
            language_mode=language_mode,
        )

        accumulated_tokens: list[str] = []
        total_usage = ChatUsage()

        try:
            async for chunk in provider.stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if chunk.content:
                    accumulated_tokens.append(chunk.content)
                    yield StreamEvent(event="token", data=chunk.content)
                if chunk.usage:
                    total_usage = chunk.usage
        except Exception as e:
            logger.warning(
                "Streaming from provider failed (%s); streaming deterministic synthesis from retrieved articles.",
                e,
            )
            fallback_text = build_statutory_synthesis(
                retrieved_chunks, is_ar, self.refusal_message_ar, self.refusal_message_en
            )

            words = fallback_text.split(" ")
            for i, w in enumerate(words):
                prefix = "" if i == 0 else " "
                token = prefix + w
                accumulated_tokens.append(token)
                yield StreamEvent(event="token", data=token)
                await asyncio.sleep(0.01)

        full_text = "".join(accumulated_tokens)

        # Grounding check and citation extraction
        grounding_res = GroundingCheckResult(is_grounded=True, reason="Unguarded")
        if enable_grounding:
            _, grounding_res = enforce_grounding_guardrail(
                answer=full_text,
                retrieved_chunks=retrieved_chunks,
                refusal_message_ar=self.refusal_message_ar,
                refusal_message_en=self.refusal_message_en,
                is_arabic=is_ar,
                overlap_threshold=self.grounding_threshold,
            )

        citations = extract_citations(full_text, retrieved_chunks)

        yield StreamEvent(
            event="citations",
            data=[c.model_dump() for c in citations],
        )

        yield StreamEvent(
            event="done",
            data={
                "is_grounded": grounding_res.is_grounded,
                "confidence_score": grounding_res.confidence_score,
                "reason": grounding_res.reason,
                "usage": total_usage.model_dump(),
            },
        )
