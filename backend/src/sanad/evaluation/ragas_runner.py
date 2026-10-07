"""RAGAS Evaluation runner for Sanad legal Q&A."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sanad.evaluation.dataset import EvalQuestion, load_eval_dataset
from sanad.evaluation.judge import LLMJudge, SampleEvaluation
from sanad.rag import SanadRAG

logger = logging.getLogger(__name__)


class EvaluationReport(BaseModel):
    """Aggregated evaluation metrics report."""

    sample_count: int
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    latency_p95: float
    tokens_per_query: float
    results: list[SampleEvaluation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Format report as GitHub Flavored Markdown."""
        lines = [
            "# Sanad RAG Evaluation Report",
            f"**Evaluated Samples:** {self.sample_count}",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "",
            "## Aggregate Metrics",
            "| Metric | Score | Course Target | Gate Threshold |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Faithfulness** | **{self.faithfulness:.4f}** | >= 0.85 | >= 0.75 |",
            f"| **Answer Relevancy** | {self.answer_relevancy:.4f} | >= 0.80 | >= 0.70 |",
            f"| **Context Precision** | {self.context_precision:.4f} | >= 0.80 | >= 0.70 |",
            f"| **Context Recall** | {self.context_recall:.4f} | >= 0.80 | >= 0.70 |",
            f"| **Hit Rate@K** | {self.hit_rate:.4f} | >= 0.85 | >= 0.75 |",
            f"| **Mean Reciprocal Rank (MRR)** | {self.mrr:.4f} | >= 0.75 | - |",
            f"| **Latency P95** | {self.latency_p95:.1f} ms | < 2500 ms | - |",
            f"| **Tokens per Query** | {self.tokens_per_query:.1f} tokens | - | - |",
            "",
            "## Quality Gate Verdict",
            f"- **Status:** {'PASSED' if self.faithfulness >= 0.75 else 'FAILED'}",
            f"- **Alert:** {'Nominal' if self.faithfulness >= 0.80 else 'Alert (< 0.80)'}",
            "",
        ]
        return "\n".join(lines)


class RagasRunner:
    """Runner that evaluates Sanad RAG against questions dataset."""

    def __init__(
        self,
        rag: SanadRAG | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self.rag = rag or SanadRAG()
        self.judge = judge or LLMJudge()

    async def evaluate_dataset(
        self,
        questions: list[EvalQuestion] | None = None,
        dataset_path: str | Path | None = None,
        subset: str = "ci",
        top_k: int = 5,
        chat_model_id: str | None = None,
        embedding_model_id: str | None = None,
    ) -> EvaluationReport:
        """Run evaluation on dataset."""
        if questions is None:
            questions = load_eval_dataset(dataset_path, subset=subset)

        results: list[SampleEvaluation] = []
        latencies: list[float] = []
        tokens_list: list[int] = []

        for q in questions:
            t0 = time.perf_counter()
            try:
                resp = await self.rag.query(
                    question=q.question,
                    chat_model_id=chat_model_id,
                    embedding_model_id=embedding_model_id,
                    top_k=top_k,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)
                tokens_list.append(resp.usage.total_tokens)

                retrieved_articles = [r.chunk.article_number for r in resp.retrieved_articles]
                context_snippets = [r.chunk.text_ar for r in resp.retrieved_articles]

                sample_eval = await self.judge.evaluate_sample(
                    question_id=q.id,
                    question=q.question,
                    answer=resp.answer,
                    retrieved_articles=retrieved_articles,
                    context_snippets=context_snippets,
                    ground_truth_articles=q.ground_truth_articles,
                    is_refusal_expected=q.is_refusal_expected,
                )
                results.append(sample_eval)
            except Exception as e:
                logger.error(f"Error evaluating question {q.id}: {e}")
                results.append(
                    SampleEvaluation(
                        question_id=q.id,
                        faithfulness=0.0,
                        answer_relevancy=0.0,
                        context_precision=0.0,
                        context_recall=0.0,
                        hit_rate=0.0,
                        mrr=0.0,
                        details={"error": str(e)},
                    )
                )
                latencies.append(0.0)
                tokens_list.append(0)

        sample_count = len(results)
        if sample_count == 0:
            raise ValueError("No questions evaluated.")

        faithfulness_mean = float(np.mean([r.faithfulness for r in results]))
        relevancy_mean = float(np.mean([r.answer_relevancy for r in results]))
        precision_mean = float(np.mean([r.context_precision for r in results]))
        recall_mean = float(np.mean([r.context_recall for r in results]))
        hit_rate_mean = float(np.mean([r.hit_rate for r in results]))
        mrr_mean = float(np.mean([r.mrr for r in results]))
        latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        tokens_mean = float(np.mean(tokens_list)) if tokens_list else 0.0

        report = EvaluationReport(
            sample_count=sample_count,
            faithfulness=faithfulness_mean,
            answer_relevancy=relevancy_mean,
            context_precision=precision_mean,
            context_recall=recall_mean,
            hit_rate=hit_rate_mean,
            mrr=mrr_mean,
            latency_p95=latency_p95,
            tokens_per_query=tokens_mean,
            results=results,
            metadata={
                "subset": subset,
                "top_k": top_k,
                "chat_model_id": chat_model_id,
                "embedding_model_id": embedding_model_id,
            },
        )
        return report

    def log_to_mlflow(
        self,
        report: EvaluationReport,
        experiment_name: str = "chunking_and_embedding",
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str | None:
        """Log evaluation run to MLflow tracking server."""
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name) as run:
                # Log Parameters
                if params:
                    for k, v in params.items():
                        mlflow.log_param(k, v)

                # Log Metrics
                mlflow.log_metric("faithfulness", report.faithfulness)
                mlflow.log_metric("answer_relevancy", report.answer_relevancy)
                mlflow.log_metric("context_precision", report.context_precision)
                mlflow.log_metric("context_recall", report.context_recall)
                mlflow.log_metric("hit_rate@k", report.hit_rate)
                mlflow.log_metric("mrr", report.mrr)
                mlflow.log_metric("latency_p95", report.latency_p95)
                mlflow.log_metric("tokens_per_query", report.tokens_per_query)
                mlflow.log_metric("sample_count", float(report.sample_count))

                # Log report artifact
                report_path = Path("reports/ragas_results.json")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report.model_dump_json(indent=2))
                mlflow.log_artifact(str(report_path))

                md_path = Path("reports/ragas_results.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(report.to_markdown())
                mlflow.log_artifact(str(md_path))

                logger.info(
                    f"Logged run {run.info.run_id} to MLflow experiment '{experiment_name}'"
                )
                return str(run.info.run_id)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow ({e}). Continuing locally.")
            return None
