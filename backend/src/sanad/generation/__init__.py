"""Generation module for prompt construction, answering, and citation tracking."""

from sanad.generation.answerer import AnswerResult, LegalAnswerer, StreamEvent
from sanad.generation.citations import CitationItem, extract_citations
from sanad.generation.prompts import (
    REFUSAL_AR,
    REFUSAL_EN,
    SYSTEM_PROMPT,
    build_rag_messages,
    format_context,
    is_arabic_text,
)

__all__ = [
    "AnswerResult",
    "CitationItem",
    "LegalAnswerer",
    "REFUSAL_AR",
    "REFUSAL_EN",
    "SYSTEM_PROMPT",
    "StreamEvent",
    "build_rag_messages",
    "extract_citations",
    "format_context",
    "is_arabic_text",
]

