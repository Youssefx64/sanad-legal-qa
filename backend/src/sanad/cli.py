"""CLI entrypoint for Sanad Legal Q&A."""

import json
from pathlib import Path
from typing import Annotated

import typer

from sanad.corpus.parser import parse_civil_code_corpus
from sanad.corpus.schema import ArticleRecord
from sanad.corpus.validate import (
    assert_valid_corpus,
    build_corpus_metadata,
    generate_spotcheck_report,
    validate_articles,
)

app = typer.Typer(
    name="sanad",
    help="Sanad: Arabic Legal Q&A System over the Egyptian Civil Code",
    add_completion=False,
)


@app.command()
def extract(
    pdf_path: Annotated[
        Path,
        typer.Option("--pdf-path", "-p", help="Path to source Egyptian Civil Code PDF"),
    ] = Path("data/raw/egyptian_civil_code_1948.pdf"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Output directory for processed files"),
    ] = Path("data/processed"),
    report_path: Annotated[
        Path,
        typer.Option("--report-path", "-r", help="Path to write spot-check markdown report"),
    ] = Path("reports/corpus_spotcheck.md"),
) -> None:
    """Extract bilingual Egyptian Civil Code articles from PDF into structured JSON."""
    if not pdf_path.exists():
        typer.secho(f"Error: PDF file not found at {pdf_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Extracting corpus from {pdf_path}...", fg=typer.colors.CYAN)
    articles, issuance_articles = parse_civil_code_corpus(pdf_path)

    typer.secho(f"Validating {len(articles)} extracted articles...", fg=typer.colors.CYAN)
    assert_valid_corpus(articles)

    output_dir.mkdir(parents=True, exist_ok=True)
    articles_file = output_dir / "articles.json"
    issuance_file = output_dir / "issuance_law.json"
    metadata_file = output_dir / "corpus_metadata.json"

    # Save articles.json
    articles_data = [a.model_dump() for a in articles]
    articles_file.write_text(
        json.dumps(articles_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.secho(f"Saved {len(articles)} articles to {articles_file}", fg=typer.colors.GREEN)

    # Save issuance_law.json
    issuance_data = [i.model_dump() for i in issuance_articles]
    issuance_file.write_text(
        json.dumps(issuance_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.secho(
        f"Saved {len(issuance_articles)} issuance law articles to {issuance_file}",
        fg=typer.colors.GREEN,
    )

    # Save metadata
    metadata = build_corpus_metadata(articles, issuance_articles)
    metadata_file.write_text(
        json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Generate spotcheck report
    rep = generate_spotcheck_report(articles, report_path)
    typer.secho(f"Generated spot-check report at {rep}", fg=typer.colors.GREEN)

    repealed_count = sum(1 for a in articles if a.is_repealed)
    typer.secho(
        f"SUCCESS: Extracted 1..{len(articles)} articles ({len(articles) - repealed_count} active, {repealed_count} repealed).",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command()
def validate(
    articles_path: Annotated[
        Path,
        typer.Option("--articles-path", "-a", help="Path to articles.json to validate"),
    ] = Path("data/processed/articles.json"),
    report_path: Annotated[
        Path | None,
        typer.Option("--report-path", "-r", help="Optional spot-check report output path"),
    ] = None,
) -> None:
    """Validate extracted articles against all integrity and schema constraints."""
    if not articles_path.exists():
        typer.secho(f"Error: file not found at {articles_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    with articles_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    articles = [ArticleRecord.model_validate(item) for item in data]
    errors = validate_articles(articles)

    if errors:
        typer.secho(
            f"Validation FAILED with {len(errors)} error(s):",
            fg=typer.colors.RED,
            bold=True,
            err=True,
        )
        for err in errors[:20]:
            typer.secho(f"  - {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(
        f"Validation PASSED: All {len(articles)} articles conform to schema.",
        fg=typer.colors.GREEN,
        bold=True,
    )

    if report_path:
        rep = generate_spotcheck_report(articles, report_path)
        typer.secho(f"Generated spot-check report at {rep}", fg=typer.colors.GREEN)


@app.command()
def ingest(
    embedding_model: Annotated[
        str | None,
        typer.Option("--embedding-model", "-m", help="Embedding model ID to index with"),
    ] = None,
) -> None:
    """Index articles into Qdrant vector database (implemented in Phase 3)."""
    typer.secho("Ingest command will be executed in Phase 3.", fg=typer.colors.YELLOW)


@app.command()
def eval() -> None:
    """Run RAG evaluation suite (implemented in Phase 7)."""
    typer.secho("Eval command will be executed in Phase 7.", fg=typer.colors.YELLOW)


@app.command()
def drift() -> None:
    """Run embedding drift detection (implemented in Phase 8)."""
    typer.secho("Drift command will be executed in Phase 8.", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
