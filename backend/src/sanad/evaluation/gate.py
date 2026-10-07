"""Quality gate module for CI/CD and deployment gating."""

from __future__ import annotations

import logging
from pathlib import Path

from sanad.evaluation.ragas_runner import EvaluationReport, RagasRunner

logger = logging.getLogger(__name__)

DEFAULT_FAITHFULNESS_THRESHOLD = 0.75
DEFAULT_ALERT_THRESHOLD = 0.80


def evaluate_quality_gate(
    report: EvaluationReport,
    threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
) -> bool:
    """Evaluate whether evaluation report passes quality gate criteria.

    Fails if faithfulness < threshold (0.75).
    Issues alert if faithfulness < alert_threshold (0.80).
    """
    passed = report.faithfulness >= threshold
    alert = report.faithfulness < alert_threshold

    print("\n" + "=" * 60)
    print("SANAD EVALUATION QUALITY GATE")
    print("=" * 60)
    print(f"Evaluated Samples:     {report.sample_count}")
    print(f"Faithfulness Score:    {report.faithfulness:.4f}  (Threshold: >= {threshold:.2f})")
    print(f"Answer Relevancy:      {report.answer_relevancy:.4f}")
    print(f"Context Precision:     {report.context_precision:.4f}")
    print(f"Context Recall:        {report.context_recall:.4f}")
    print(f"Hit Rate@K:            {report.hit_rate:.4f}")
    print(f"Mean Reciprocal Rank:  {report.mrr:.4f}")
    print(f"Latency P95:           {report.latency_p95:.1f} ms")
    print("-" * 60)

    if passed:
        if alert:
            print(
                f"⚠️  QUALITY GATE PASSED WITH ALERT: Faithfulness {report.faithfulness:.4f} < {alert_threshold:.2f}"
            )
        else:
            print("✅ QUALITY GATE PASSED: All legal faithfulness standards satisfied.")
    else:
        print(f"❌ QUALITY GATE FAILED: Faithfulness {report.faithfulness:.4f} < {threshold:.2f}")

    print("=" * 60 + "\n")
    return passed


async def run_quality_gate(
    dataset_path: str | Path | None = None,
    subset: str = "ci",
    threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
    save_reports: bool = True,
) -> bool:
    """Run full evaluation on CI subset and evaluate gate."""
    runner = RagasRunner()
    report = await runner.evaluate_dataset(
        dataset_path=dataset_path,
        subset=subset,
    )

    if save_reports:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "ragas_results.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        (reports_dir / "ragas_results.md").write_text(report.to_markdown(), encoding="utf-8")
        runner.log_to_mlflow(report, run_name=f"gate_{subset}")

    return evaluate_quality_gate(report, threshold=threshold)
