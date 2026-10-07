"""Tests for observability modules: tracing, metrics, and drift detection."""

import numpy as np
import pytest

from sanad.observability.drift import DriftDetector, compute_ks_2samp_numpy
from sanad.observability.metrics import (
    record_faithfulness_score,
    record_refusal,
    record_retrieval_score,
    record_token_cost,
)
from sanad.observability.tracing import SanadTracer, SpanRecord
from sanad.providers.base import EmbeddingProvider


class MockDriftEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 8

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), 8)).tolist()


def test_compute_ks_2samp_numpy() -> None:
    # Identical distributions
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d_stat, p_val = compute_ks_2samp_numpy(data1, data2)
    assert d_stat == 0.0
    assert p_val >= 0.9

    # Highly separated distributions
    data_low = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    data_high = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
    d_stat_sep, p_val_sep = compute_ks_2samp_numpy(data_low, data_high)
    assert d_stat_sep == 1.0
    assert p_val_sep < 0.05


@pytest.mark.asyncio
async def test_drift_detector() -> None:
    detector = DriftDetector(threshold_distance=0.45)
    provider = MockDriftEmbeddingProvider()

    queries = [
        "ما هي شروط العقد؟",
        "ما هو فسخ العقد؟",
        "ما هي أحكام الإيجار؟",
        "ما هو حق الملكية؟",
    ]

    report = await detector.detect_drift(queries, provider=provider)
    assert report.sample_size == 4
    assert 0.0 <= report.mean_cosine_distance <= 2.0
    assert 0.0 <= report.ks_statistic <= 1.0
    assert report.verdict != ""
    assert "# Sanad Embedding Drift Detection Report" in report.to_markdown()


def test_span_record_telemetry() -> None:
    with SpanRecord(name="retrieval_span", parent_trace_id="test_123") as span:
        x = sum(i for i in range(1000))
        assert x > 0

    assert span.name == "retrieval_span"
    assert span.trace_id == "test_123"
    assert span.duration_ms >= 0.0


def test_sanad_tracer_lifecycle() -> None:
    tracer = SanadTracer()
    span = tracer.span(name="generation_span")
    with span:
        pass
    assert span.duration_ms >= 0.0

    # Test safe fallback logging without active server
    tracer.log_generation(
        trace_id="dummy_trace",
        name="llm_generate",
        model="test_model",
        prompt="hello",
        completion="world",
    )
    tracer.score_trace(trace_id="dummy_trace", name="faithfulness", value=0.95)


def test_metrics_instrumentation_helpers() -> None:
    # Verify helper calls execute cleanly
    record_token_cost(model_id="openrouter-chat-1", prompt_tokens=100, completion_tokens=50)
    record_retrieval_score(score=0.88)
    record_refusal(language="ar")
    record_faithfulness_score(score=0.92)
