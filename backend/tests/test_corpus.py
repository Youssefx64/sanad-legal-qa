"""Unit and integration tests for the corpus extraction, normalization, and validation."""

from pathlib import Path

import pytest

from sanad.corpus.arabic_text import (
    clean_arabic_raw,
    extract_article_references,
    normalize_arabic,
    to_western_digits,
)
from sanad.corpus.parser import parse_civil_code_corpus
from sanad.corpus.schema import ArticleRecord
from sanad.corpus.validate import (
    ALL_REPEALED_ARTICLES,
    CorpusValidationError,
    assert_valid_corpus,
    build_corpus_metadata,
    generate_spotcheck_report,
    validate_articles,
)


def test_arabic_digit_conversion() -> None:
    """Test conversion of Arabic-Indic digits to Western digits."""
    assert to_western_digits("المادة ١٤٧") == "المادة 147"
    assert to_western_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"
    assert to_western_digits("No digits here") == "No digits here"


def test_arabic_normalization() -> None:
    """Test tatweel, diacritics, alef, and yaa normalization."""
    # Tashkeel removal
    assert normalize_arabic("القَانُونُ المَدَنِيُّ") == "القانون المدني"
    # Tatweel removal
    assert normalize_arabic("الـــــقانون") == "القانون"
    # Alef normalization
    assert normalize_arabic("أحمد وإبراهيم والآخر") == "احمد وابراهيم والاخر"
    # Yaa normalization
    assert normalize_arabic("على القاضى") == "علي القاضي"
    # Whitespace collapsing
    assert normalize_arabic("  مادة   147 \n\n العقد  ") == "مادة 147 العقد"


def test_clean_arabic_raw() -> None:
    """Test raw Arabic cleaning preserves letters but standardizes formatting."""
    cleaned = clean_arabic_raw("  العقد \t شريعة   المتعاقدين\n\n ")
    assert cleaned == "العقد شريعة المتعاقدين"


def test_extract_article_references() -> None:
    """Test extracting article references from both Arabic and English text."""
    text_ar = "تسري أحكام المادتين 147 و 148 والمادة 215 على هذا النزاع."
    text_en = "Subject to the provisions of Articles 219 and 220 as well as Article 147."

    refs = extract_article_references(text_ar, text_en)
    assert 147 in refs
    assert 148 in refs
    assert 215 in refs
    assert 219 in refs
    assert 220 in refs


def test_article_record_schema() -> None:
    """Test ArticleRecord model validation."""
    record = ArticleRecord(
        article_number=147,
        book="Obligations Generally",
        chapter="Contracts",
        text_ar="العقد شريعة المتعاقدين",
        text_ar_raw="العقد شريعة المتعاقدين",
        text_en="The contract makes the law of the parties",
        is_repealed=False,
        source_page=34,
        citation="Egyptian Civil Code, Article 147",
        references=[148],
    )
    assert record.article_number == 147
    assert record.citation == "Egyptian Civil Code, Article 147"
    assert not record.is_repealed
    assert record.references == [148]


def test_validate_articles_success() -> None:
    """Test that a completely valid contiguous corpus passes validation."""
    mock_articles = []
    for num in range(1, 1150):
        is_rep = num in ALL_REPEALED_ARTICLES
        mock_articles.append(
            ArticleRecord(
                article_number=num,
                book="Book 1",
                text_ar=f"نص المادة {num}",
                text_ar_raw=f"نص المادة {num}",
                text_en=f"Text of Article {num}",
                is_repealed=is_rep,
                source_page=num // 10 + 1,
                citation=f"Egyptian Civil Code, Article {num}",
                references=[],
            )
        )

    errors = validate_articles(mock_articles)
    assert errors == []
    assert_valid_corpus(mock_articles)

    metadata = build_corpus_metadata(mock_articles)
    assert metadata.total_articles == 1149
    assert metadata.repealed_articles == len(ALL_REPEALED_ARTICLES)
    assert metadata.active_articles == 1149 - len(ALL_REPEALED_ARTICLES)


def test_validate_articles_failure_cases(tmp_path: Path) -> None:
    """Test validation catches missing articles, empty text, and invalid repeal flags."""
    # Missing article 10
    incomplete = [
        ArticleRecord(
            article_number=n,
            text_ar="نص",
            text_ar_raw="نص",
            text_en="text",
            is_repealed=(n in ALL_REPEALED_ARTICLES),
            source_page=1,
            citation=f"Egyptian Civil Code, Article {n}",
        )
        for n in range(1, 1150)
        if n != 10
    ]
    errors = validate_articles(incomplete)
    assert any("Missing 1 article numbers" in e for e in errors)

    # Empty Arabic text
    bad_text = [
        ArticleRecord(
            article_number=1,
            text_ar="",
            text_ar_raw="",
            text_en="valid text",
            is_repealed=False,
            source_page=1,
            citation="Egyptian Civil Code, Article 1",
        )
    ]
    errors_text = validate_articles(bad_text)
    assert any("text_ar is empty" in e for e in errors_text)

    # Repealed not flagged
    bad_repeal = [
        ArticleRecord(
            article_number=54,
            text_ar="ملغاة",
            text_ar_raw="ملغاة",
            text_en="repealed",
            is_repealed=False,  # Should be True
            source_page=10,
            citation="Egyptian Civil Code, Article 54",
        )
    ]
    errors_rep = validate_articles(bad_repeal)
    assert any("Known repealed article is not flagged" in e for e in errors_rep)

    with pytest.raises(CorpusValidationError):
        assert_valid_corpus(bad_repeal)


def test_spotcheck_report_generation(tmp_path: Path) -> None:
    """Test markdown spot check report is written correctly."""
    mock_articles = [
        ArticleRecord(
            article_number=num,
            book="Obligations",
            part="General",
            chapter="Contracts",
            section="Formation",
            topic="Consent",
            text_ar=f"نص المادة {num} بالتفصيل",
            text_ar_raw=f"نص المادة {num} بالتفصيل",
            text_en=f"Text of Article {num} in detail",
            is_repealed=(num in ALL_REPEALED_ARTICLES),
            source_page=10,
            citation=f"Egyptian Civil Code, Article {num}",
            references=[1, 2],
        )
        for num in range(1, 51)
    ]

    report_file = tmp_path / "spotcheck.md"
    generated = generate_spotcheck_report(mock_articles, report_file, sample_size=5, seed=42)
    assert generated.exists()

    content = generated.read_text(encoding="utf-8")
    assert "# Egyptian Civil Code (1948) — Corpus Spot-Check Report" in content
    assert "Sampled Articles Verification" in content
    assert "Egyptian Civil Code, Article" in content


@pytest.mark.skipif(
    not Path("data/raw/egyptian_civil_code_1948.pdf").exists(),
    reason="Raw PDF not present in environment",
)
def test_end_to_end_pdf_parsing() -> None:
    """End-to-end integration test extracting the actual PDF corpus."""
    pdf_path = Path("data/raw/egyptian_civil_code_1948.pdf")
    articles, issuance_articles = parse_civil_code_corpus(pdf_path)

    # Contiguity and counts
    assert len(articles) == 1149
    assert articles[0].article_number == 1
    assert articles[-1].article_number == 1149
    assert len(issuance_articles) == 2

    # Check repealed articles
    rep_54_80 = [a for a in articles if 54 <= a.article_number <= 80]
    assert all(a.is_repealed for a in rep_54_80)
    rep_389_417 = [a for a in articles if 389 <= a.article_number <= 417]
    assert all(a.is_repealed for a in rep_389_417)

    # Check that Article 147 has key contract text
    art147 = next(a for a in articles if a.article_number == 147)
    assert not art147.is_repealed
    assert "العقد" in art147.text_ar
    assert "contract" in art147.text_en.lower()

    # Check Article 1022 text is populated
    art1022 = next(a for a in articles if a.article_number == 1022)
    assert len(art1022.text_ar.strip()) > 0
    assert len(art1022.text_en.strip()) > 0

    # Ensure validation passes completely
    errors = validate_articles(articles)
    assert errors == []


def test_cli_extract_and_validate(tmp_path: Path) -> None:
    """Test CLI commands extract and validate end-to-end."""
    from typer.testing import CliRunner

    from sanad.cli import app

    runner = CliRunner()
    out_dir = tmp_path / "processed"
    rep_file = tmp_path / "spotcheck.md"
    pdf_path = Path("data/raw/egyptian_civil_code_1948.pdf")

    if not pdf_path.exists():
        pytest.skip("PDF not found")

    # Run extract command
    res = runner.invoke(
        app,
        [
            "extract",
            "--pdf-path",
            str(pdf_path),
            "--output-dir",
            str(out_dir),
            "--report-path",
            str(rep_file),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (out_dir / "articles.json").exists()
    assert (out_dir / "issuance_law.json").exists()
    assert (out_dir / "corpus_metadata.json").exists()
    assert rep_file.exists()

    # Run validate command
    val_res = runner.invoke(
        app,
        [
            "validate",
            "--articles-path",
            str(out_dir / "articles.json"),
        ],
    )
    assert val_res.exit_code == 0, val_res.stdout
    assert "Validation PASSED" in val_res.stdout
