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
    articles_path: Annotated[
        Path,
        typer.Option("--articles-path", "-a", help="Path to articles.json to index"),
    ] = Path("data/processed/articles.json"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-indexing even if collection exists"),
    ] = False,
) -> None:
    """Index articles into Qdrant vector database."""
    import asyncio

    from sanad.indexing.ingest import run_ingestion

    typer.secho(
        f"Starting ingestion for model '{embedding_model or 'default'}'...", fg=typer.colors.CYAN
    )
    try:
        res = asyncio.run(
            run_ingestion(
                articles_path=articles_path,
                embedding_model_id=embedding_model,
                force=force,
            )
        )
        typer.secho(f"Ingestion {res['status']}:", fg=typer.colors.GREEN, bold=True)
        typer.secho(f"  - Collection: {res['collection_name']}")
        typer.secho(f"  - Model: {res['model_id']} (dim={res['dimension']})")
        typer.secho(f"  - Chunks: {res['chunks_count']}")
    except Exception as e:
        typer.secho(f"Ingestion failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@app.command(name="eval")
def evaluate(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to evaluation questions JSONL"),
    ] = Path("data/eval/eval_questions.jsonl"),
    subset: Annotated[
        str,
        typer.Option(
            "--subset", "-s", help="Subset to evaluate ('ci' for 20-sample gate, 'full' for all)"
        ),
    ] = "ci",
    gate: Annotated[
        bool,
        typer.Option("--gate", "-g", help="Enforce quality gate threshold (faithfulness >= 0.75)"),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Quality gate faithfulness threshold"),
    ] = 0.75,
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of retrieved articles"),
    ] = 5,
    chat_model: Annotated[
        str | None,
        typer.Option("--chat-model", help="Chat model ID to evaluate"),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option("--embedding-model", help="Embedding model ID to evaluate"),
    ] = None,
    experiment_name: Annotated[
        str,
        typer.Option("--experiment", "-e", help="MLflow experiment name"),
    ] = "chunking_and_embedding",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to save evaluation reports"),
    ] = Path("reports"),
) -> None:
    """Run RAG evaluation suite and quality gate."""
    import asyncio

    from sanad.evaluation.gate import evaluate_quality_gate
    from sanad.evaluation.ragas_runner import RagasRunner

    typer.secho(
        f"Starting Sanad evaluation (subset={subset}, top_k={top_k})...",
        fg=typer.colors.CYAN,
    )

    runner = RagasRunner()
    try:
        report = asyncio.run(
            runner.evaluate_dataset(
                dataset_path=dataset_path,
                subset=subset,
                top_k=top_k,
                chat_model_id=chat_model,
                embedding_model_id=embedding_model,
            )
        )
    except Exception as e:
        typer.secho(f"Evaluation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ragas_results.json"
    md_path = output_dir / "ragas_results.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    typer.secho(f"Saved evaluation reports to {json_path} and {md_path}", fg=typer.colors.GREEN)

    # Log to MLflow
    params = {
        "subset": subset,
        "top_k": top_k,
        "chat_model": chat_model or "default",
        "embedding_model": embedding_model or "default",
    }
    run_id = runner.log_to_mlflow(
        report=report,
        experiment_name=experiment_name,
        run_name=f"eval_{subset}",
        params=params,
    )
    if run_id:
        typer.secho(
            f"Logged run {run_id} to MLflow experiment '{experiment_name}'", fg=typer.colors.GREEN
        )

    # Print summary
    typer.echo(report.to_markdown())

    if gate:
        passed = evaluate_quality_gate(report, threshold=threshold)
        if not passed:
            typer.secho(
                f"Evaluation FAILED quality gate: Faithfulness {report.faithfulness:.4f} < {threshold:.2f}",
                fg=typer.colors.RED,
                bold=True,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.secho("Quality gate PASSED.", fg=typer.colors.GREEN, bold=True)


@app.command()
def drift(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for drift JSON report"),
    ] = Path("reports/drift_report.json"),
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Cosine distance drift threshold"),
    ] = 0.45,
) -> None:
    """Detect embedding distribution drift against the corpus centroid."""
    import asyncio

    from sanad.observability.drift import run_drift_check

    typer.secho("Running embedding drift detection...", fg=typer.colors.CYAN)
    try:
        report = asyncio.run(run_drift_check(output_path=output, threshold=threshold))
    except Exception as e:
        typer.secho(f"Drift detection failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    typer.echo(report.to_markdown())
    if report.drift_detected:
        typer.secho(
            "⚠️  ALERT: Semantic embedding drift detected!", fg=typer.colors.YELLOW, bold=True
        )
    else:
        typer.secho(
            "✅ Query embeddings are well-aligned with corpus centroid.",
            fg=typer.colors.GREEN,
            bold=True,
        )


if __name__ == "__main__":
    app()
