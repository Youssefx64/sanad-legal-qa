"""Pydantic schemas for the Egyptian Civil Code corpus."""

from pydantic import BaseModel, Field


class ArticleRecord(BaseModel):
    """Normalized record for a single article of the Egyptian Civil Code."""

    article_number: int = Field(description="Article number (1..1149)")
    book: str | None = Field(default=None, description="Book name (e.g. Obligations Generally)")
    part: str | None = Field(default=None, description="Part name")
    chapter: str | None = Field(default=None, description="Chapter name")
    section: str | None = Field(default=None, description="Section name")
    topic: str | None = Field(default=None, description="Sub-topic heading")
    text_ar: str = Field(description="Cleaned, normalized Arabic text")
    text_ar_raw: str = Field(description="Raw extracted Arabic text")
    text_en: str = Field(description="Cleaned English text")
    is_repealed: bool = Field(default=False, description="True if article has been repealed")
    source_page: int = Field(
        description="First page number in the source PDF where article appears"
    )
    citation: str = Field(description="Standardized legal citation string")
    references: list[int] = Field(
        default_factory=list,
        description="Cross-referenced article numbers cited within this article",
    )


class IssuanceLawArticle(BaseModel):
    """Articles from the Issuance Law (قانون الإصدار) separate from the Code."""

    article_number: int = Field(description="Issuance law article number (1 or 2)")
    text_ar: str = Field(description="Arabic text of issuance law article")
    text_en: str = Field(description="English text of issuance law article")
    source_page: int = Field(default=1, description="Page number")
    citation: str = Field(description="Citation string")


class CorpusMetadata(BaseModel):
    """Corpus summary metadata."""

    total_articles: int
    active_articles: int
    repealed_articles: int
    min_article_number: int
    max_article_number: int
    issuance_law_articles: list[IssuanceLawArticle] = Field(default_factory=list)
