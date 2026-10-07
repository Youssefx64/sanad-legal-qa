"""Tests for evaluation dataset, judge, runner, and quality gate."""

import pytest

from sanad.evaluation.dataset import load_eval_dataset, validate_eval_dataset
from sanad.evaluation.gate import evaluate_quality_gate
from sanad.evaluation.judge import LLMJudge
from sanad.evaluation.ragas_runner import EvaluationReport


def test_load_and_validate_eval_dataset() -> None:
    questions = load_eval_dataset(subset="full")
    assert len(questions) >= 50

    stats = validate_eval_dataset(questions)
    assert stats["total_questions"] >= 50
    assert stats["bilingual_twins"] >= 20
    assert stats["repealed_questions"] >= 5
    assert stats["refusal_questions"] >= 5


def test_load_ci_subset() -> None:
    ci_questions = load_eval_dataset(subset="ci")
    assert 1 <= len(ci_questions) <= 20
    assert all(q.is_ci_subset for q in ci_questions)


def test_judge_retrieval_metrics() -> None:
    judge = LLMJudge()

    # Exact top-1 match
    metrics = judge.compute_retrieval_metrics(
        retrieved_articles=[147, 148, 157],
        ground_truth_articles=[147],
    )
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["context_recall"] == 1.0
    assert metrics["context_precision"] == pytest.approx(1.0 / 3.0)

    # Top-2 match
    metrics_rank2 = judge.compute_retrieval_metrics(
        retrieved_articles=[100, 147, 157],
        ground_truth_articles=[147],
    )
    assert metrics_rank2["hit_rate"] == 1.0
    assert metrics_rank2["mrr"] == 0.5
    assert metrics_rank2["context_recall"] == 1.0

    # No match
    metrics_miss = judge.compute_retrieval_metrics(
        retrieved_articles=[1, 2, 3],
        ground_truth_articles=[147],
    )
    assert metrics_miss["hit_rate"] == 0.0
    assert metrics_miss["mrr"] == 0.0
    assert metrics_miss["context_recall"] == 0.0
    assert metrics_miss["context_precision"] == 0.0

    # Out of scope (empty ground truth)
    metrics_oos = judge.compute_retrieval_metrics(
        retrieved_articles=[],
        ground_truth_articles=[],
    )
    assert metrics_oos["hit_rate"] == 1.0
    assert metrics_oos["context_precision"] == 1.0


@pytest.mark.asyncio
async def test_judge_faithfulness_and_relevancy_refusal() -> None:
    judge = LLMJudge()

    # Correct canonical refusal on out-of-scope query
    faith, rel, _ = await judge.judge_faithfulness_and_relevancy(
        question="What is criminal penalty for murder?",
        answer="لم أجد أساساً في القانون المدني للإجابة عن هذا السؤال.",
        context_snippets=[],
        is_refusal=True,
        is_refusal_expected=True,
    )
    assert faith == 1.0
    assert rel == 1.0

    # Hallucinated answer when refusal was expected
    faith_bad, rel_bad, _ = await judge.judge_faithfulness_and_relevancy(
        question="What is criminal penalty for murder?",
        answer="العقوبة هي الإعدام وفقاً لقانون العقوبات.",
        context_snippets=[],
        is_refusal=False,
        is_refusal_expected=True,
    )
    assert faith_bad == 0.0
    assert rel_bad == 0.0


def test_quality_gate_evaluation() -> None:
    # Passing report
    passing_report = EvaluationReport(
        sample_count=20,
        faithfulness=0.88,
        answer_relevancy=0.85,
        context_precision=0.82,
        context_recall=0.90,
        hit_rate=0.95,
        mrr=0.89,
        latency_p95=450.0,
        tokens_per_query=220.0,
    )
    assert evaluate_quality_gate(passing_report, threshold=0.75, alert_threshold=0.80) is True

    # Failing report
    failing_report = EvaluationReport(
        sample_count=20,
        faithfulness=0.68,
        answer_relevancy=0.70,
        context_precision=0.60,
        context_recall=0.60,
        hit_rate=0.65,
        mrr=0.55,
        latency_p95=500.0,
        tokens_per_query=200.0,
    )
    assert evaluate_quality_gate(failing_report, threshold=0.75, alert_threshold=0.80) is False


def test_evaluation_report_markdown_formatting() -> None:
    report = EvaluationReport(
        sample_count=10,
        faithfulness=0.92,
        answer_relevancy=0.88,
        context_precision=0.85,
        context_recall=0.90,
        hit_rate=0.95,
        mrr=0.90,
        latency_p95=320.0,
        tokens_per_query=150.0,
    )
    md = report.to_markdown()
    assert "# Sanad RAG Evaluation Report" in md
    assert "Faithfulness" in md
    assert "0.9200" in md
    assert "PASSED" in md
