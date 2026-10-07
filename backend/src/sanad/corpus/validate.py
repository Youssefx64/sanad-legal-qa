"""Validation logic and spot-check reporting for the Egyptian Civil Code corpus."""

import json
import random
import sys
from pathlib import Path

from sanad.corpus.schema import ArticleRecord, CorpusMetadata, IssuanceLawArticle

# Expected constants
TOTAL_EXPECTED_ARTICLES = 1149
REPEALED_ARTICLES_54_80 = set(range(54, 81))
REPEALED_ARTICLES_389_417 = set(range(389, 418))
ALL_REPEALED_ARTICLES = REPEALED_ARTICLES_54_80 | REPEALED_ARTICLES_389_417


class CorpusValidationError(Exception):
    """Raised when corpus integrity checks fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"Corpus validation failed with {len(errors)} error(s):\n"
            + "\n".join(f" - {err}" for err in errors[:20])
            + (f"\n ... and {len(errors) - 20} more" if len(errors) > 20 else "")
        )


def validate_articles(
    articles: list[ArticleRecord],
    max_chars: int = 5000,
) -> list[str]:
    """Validate article records against all integrity constraints.

    Returns a list of error strings. Empty list indicates full validity.
    """
    errors: list[str] = []

    if not articles:
        return ["Corpus is empty: 0 articles found."]

    # 1. Contiguity check (1..1149)
    article_numbers = [a.article_number for a in articles]
    number_set = set(article_numbers)

    if len(article_numbers) != len(number_set):
        duplicates = [num for num in number_set if article_numbers.count(num) > 1]
        errors.append(f"Duplicate article numbers found: {sorted(duplicates)}")

    expected_set = set(range(1, TOTAL_EXPECTED_ARTICLES + 1))
    missing = expected_set - number_set
    if missing:
        errors.append(f"Missing {len(missing)} article numbers: {sorted(missing)[:10]}...")

    unexpected = number_set - expected_set
    if unexpected:
        errors.append(
            f"Unexpected article numbers out of bounds (not 1..1149): {sorted(unexpected)}"
        )

    # 2. Per-article validation
    for article in articles:
        num = article.article_number

        # Check types
        if not isinstance(num, int):
            errors.append(f"Article {num}: article_number is not an integer ({type(num)})")

        # Check citation
        if not article.citation or not article.citation.startswith("Egyptian Civil Code, Article "):
            errors.append(f"Article {num}: Malformed citation '{article.citation}'")

        # Check non-empty Arabic text
        if not article.text_ar or not article.text_ar.strip():
            errors.append(f"Article {num}: text_ar is empty")

        if not article.text_ar_raw or not article.text_ar_raw.strip():
            errors.append(f"Article {num}: text_ar_raw is empty")

        # Check non-empty English text
        if not article.text_en or not article.text_en.strip():
            errors.append(f"Article {num}: text_en is empty")

        # Check sane length bounds
        if len(article.text_ar) > max_chars:
            errors.append(
                f"Article {num}: text_ar length ({len(article.text_ar)}) exceeds threshold ({max_chars})"
            )
        if len(article.text_en) > max_chars:
            errors.append(
                f"Article {num}: text_en length ({len(article.text_en)}) exceeds threshold ({max_chars})"
            )

        # Check repealed status
        if num in ALL_REPEALED_ARTICLES and not article.is_repealed:
            errors.append(f"Article {num}: Known repealed article is not flagged is_repealed=True")
        elif num not in ALL_REPEALED_ARTICLES and article.is_repealed:
            errors.append(
                f"Article {num}: Article flagged as repealed but not in known repealed list"
            )

        # Check source page
        if article.source_page < 1:
            errors.append(f"Article {num}: Invalid source_page ({article.source_page})")

    return errors


def assert_valid_corpus(
    articles: list[ArticleRecord],
    max_chars: int = 5000,
) -> None:
    """Assert that corpus is valid, raising CorpusValidationError on failure."""
    errors = validate_articles(articles, max_chars=max_chars)
    if errors:
        raise CorpusValidationError(errors)


def build_corpus_metadata(
    articles: list[ArticleRecord],
    issuance_articles: list[IssuanceLawArticle] | None = None,
) -> CorpusMetadata:
    """Build summary metadata for the corpus."""
    repealed_count = sum(1 for a in articles if a.is_repealed)
    nums = [a.article_number for a in articles]
    return CorpusMetadata(
        total_articles=len(articles),
        active_articles=len(articles) - repealed_count,
        repealed_articles=repealed_count,
        min_article_number=min(nums) if nums else 0,
        max_article_number=max(nums) if nums else 0,
        issuance_law_articles=issuance_articles or [],
    )


def generate_spotcheck_report(
    articles: list[ArticleRecord],
    output_path: str | Path,
    sample_size: int = 20,
    seed: int = 42,
) -> Path:
    """Spot-check N random articles and write a formatted Markdown report."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    # Sort by article number first for deterministic indexing
    sorted_articles = sorted(articles, key=lambda a: a.article_number)
    sampled = sorted(
        rng.sample(sorted_articles, min(sample_size, len(sorted_articles))),
        key=lambda a: a.article_number,
    )

    lines: list[str] = [
        "# Egyptian Civil Code (1948) — Corpus Spot-Check Report",
        "",
        f"- **Total Articles**: {len(articles)} (Contiguous 1..{TOTAL_EXPECTED_ARTICLES})",
        f"- **Active Articles**: {sum(1 for a in articles if not a.is_repealed)}",
        f"- **Repealed Articles**: {sum(1 for a in articles if a.is_repealed)} (Articles 54–80 & 389–417)",
        f"- **Spot-Sample Size**: {len(sampled)} articles",
        f"- **Sampling Seed**: {seed}",
        "",
        "## Sampled Articles Verification",
        "",
    ]

    for item in sampled:
        status = "⚠️ REPEALED" if item.is_repealed else "✅ ACTIVE"
        ar_snippet = item.text_ar[:160].replace("\n", " ") + (
            "..." if len(item.text_ar) > 160 else ""
        )
        en_snippet = item.text_en[:160].replace("\n", " ") + (
            "..." if len(item.text_en) > 160 else ""
        )
        refs = ", ".join(str(r) for r in item.references) if item.references else "None"

        lines.extend(
            [
                f"### Article {item.article_number} ({status})",
                f"- **Citation**: `{item.citation}`",
                f"- **Source Page**: {item.source_page}",
                f"- **Hierarchy**: Book: `{item.book or 'N/A'}` | Part: `{item.part or 'N/A'}` | Chapter: `{item.chapter or 'N/A'}` | Section: `{item.section or 'N/A'}`",
                f"- **Topic**: `{item.topic or 'N/A'}`",
                f"- **Cross-References**: {refs}",
                f"- **Arabic Length**: {len(item.text_ar)} chars | **English Length**: {len(item.text_en)} chars",
                "",
                "**Arabic Text (Normalized):**",
                f"> {ar_snippet}",
                "",
                "**English Text:**",
                f"> {en_snippet}",
                "",
                "---",
                "",
            ]
        )

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def validate_file(
    json_path: str | Path,
    report_path: str | Path | None = None,
) -> bool:
    """Validate an articles.json file on disk and optionally generate spot check report."""
    path = Path(json_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return False

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    articles = [ArticleRecord.model_validate(item) for item in data]
    errors = validate_articles(articles)

    if errors:
        print(f"❌ Corpus Validation Failed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors[:10]:
            print(f"   - {err}", file=sys.stderr)
        return False

    print(f"✅ Corpus Validation Succeeded! All {len(articles)} articles valid.")
    if report_path:
        out = generate_spotcheck_report(articles, report_path)
        print(f"📄 Spot-check report generated: {out}")
    return True
