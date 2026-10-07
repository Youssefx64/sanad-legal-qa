"""Article-aware chunking strategies for the Egyptian Civil Code."""

import re

from pydantic import BaseModel, Field

from sanad.corpus.arabic_text import normalize_arabic
from sanad.corpus.schema import ArticleRecord


class ChunkRecord(BaseModel):
    """A single retrievable text chunk originating from a legal article."""

    chunk_id: str = Field(description="Unique chunk identifier, e.g. art_147_p1")
    article_number: int = Field(description="Egyptian Civil Code article number")
    book: str | None = None
    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    citation: str
    text_ar: str = Field(description="Arabic text for display and citations")
    text_ar_normalized: str = Field(description="Normalized Arabic text for embedding and BM25")
    text_en: str = Field(description="English text of the chunk")
    is_repealed: bool = False
    references: list[int] = Field(default_factory=list)
    source_page: int = 1
    paragraph_number: int | None = None
    total_paragraphs: int = 1


def _split_into_paragraphs_ar(text: str) -> list[str]:
    """Split Arabic article text into distinct paragraphs."""
    # Check for numbered paragraph markers: (1), (2), (١), (٢), 1 -, 2 -, 1., 2.
    num_pattern = re.compile(
        r"(?:^|\n)(?=(?:[\(\[]\s*[\d\u0660-\u0669]+\s*[\)\]]|\b[\d\u0660-\u0669]+[\s\.\-]+))"
    )
    parts = [p.strip() for p in num_pattern.split(text) if p.strip()]
    if len(parts) > 1:
        return parts

    # Otherwise split on double newlines
    lines = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(lines) > 1:
        return lines

    return [text.strip()]


def _split_into_paragraphs_en(text: str) -> list[str]:
    """Split English article text into distinct paragraphs."""
    num_pattern = re.compile(r"(?:^|\n)(?=(?:[\(\[]\s*\d+\s*[\)\]]|\b\d+[\s\.\-]+))")
    parts = [p.strip() for p in num_pattern.split(text) if p.strip()]
    if len(parts) > 1:
        return parts

    lines = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(lines) > 1:
        return lines

    return [text.strip()]


def chunk_article(
    article: ArticleRecord,
    strategy: str = "article_paragraph",
    max_chars: int = 1200,
    overlap: int = 100,
) -> list[ChunkRecord]:
    """Chunk a single ArticleRecord into one or more ChunkRecords."""
    num = article.article_number
    text_ar = article.text_ar.strip()
    text_en = article.text_en.strip()

    # If strategy is "article" or text is within max_chars, produce single chunk
    if strategy == "article" or len(text_ar) <= max_chars:
        return [
            ChunkRecord(
                chunk_id=f"art_{num}",
                article_number=num,
                book=article.book,
                part=article.part,
                chapter=article.chapter,
                section=article.section,
                topic=article.topic,
                citation=article.citation,
                text_ar=text_ar,
                text_ar_normalized=normalize_arabic(text_ar),
                text_en=text_en,
                is_repealed=article.is_repealed,
                references=article.references,
                source_page=article.source_page,
                paragraph_number=None,
                total_paragraphs=1,
            )
        ]

    # Split into paragraphs
    ar_paras = _split_into_paragraphs_ar(text_ar)
    en_paras = _split_into_paragraphs_en(text_en)

    total_paras = max(len(ar_paras), len(en_paras))
    chunks: list[ChunkRecord] = []

    for idx in range(total_paras):
        p_num = idx + 1
        p_ar = ar_paras[idx] if idx < len(ar_paras) else (ar_paras[-1] if ar_paras else text_ar)
        p_en = en_paras[idx] if idx < len(en_paras) else (en_paras[-1] if en_paras else text_en)

        chunks.append(
            ChunkRecord(
                chunk_id=f"art_{num}_p{p_num}",
                article_number=num,
                book=article.book,
                part=article.part,
                chapter=article.chapter,
                section=article.section,
                topic=article.topic,
                citation=f"{article.citation}, Paragraph {p_num}",
                text_ar=p_ar,
                text_ar_normalized=normalize_arabic(p_ar),
                text_en=p_en,
                is_repealed=article.is_repealed,
                references=article.references,
                source_page=article.source_page,
                paragraph_number=p_num,
                total_paragraphs=total_paras,
            )
        )

    return chunks


def chunk_corpus(
    articles: list[ArticleRecord],
    strategy: str = "article_paragraph",
    max_chars: int = 1200,
    overlap: int = 100,
) -> list[ChunkRecord]:
    """Chunk all articles in the corpus."""
    all_chunks: list[ChunkRecord] = []
    for article in articles:
        all_chunks.extend(
            chunk_article(
                article,
                strategy=strategy,
                max_chars=max_chars,
                overlap=overlap,
            )
        )
    return all_chunks
