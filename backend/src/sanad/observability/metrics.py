"""Prometheus metrics collectors and middleware instrumentation."""

import time
from collections.abc import Callable
from typing import Any, cast

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

from sanad.config.registry import get_model_registry

# HTTP Metrics definitions
REQUEST_COUNT = Counter(
    "sanad_requests_total",
    "Total HTTP requests handled by Sanad backend",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "sanad_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# LLM & Token Metrics
TOKEN_USAGE = Counter(
    "sanad_tokens_total",
    "Total tokens consumed across chat completions",
    ["model", "type"],  # type: 'prompt' or 'completion'
)

ESTIMATED_COST = Counter(
    "sanad_estimated_cost_usd_total",
    "Estimated cost in USD across LLM providers",
    ["model"],
)

GROUNDING_FAILURES = Counter(
    "sanad_grounding_interventions_total",
    "Number of times the grounding guardrail suppressed ungrounded hallucinations",
)

REFUSALS_TOTAL = Counter(
    "sanad_refusals_total",
    "Total queries where system refused due to lack of statutory grounding",
    ["language"],
)

RETRIEVAL_SCORES = Histogram(
    "sanad_retrieval_scores",
    "Distribution of article relevance scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

RAGAS_FAITHFULNESS_GAUGE = Gauge(
    "sanad_ragas_faithfulness",
    "Latest RAGAS faithfulness evaluation score",
)


def record_token_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Calculate and record token consumption and USD cost."""
    TOKEN_USAGE.labels(model=model_id, type="prompt").inc(prompt_tokens)
    TOKEN_USAGE.labels(model=model_id, type="completion").inc(completion_tokens)

    try:
        registry = get_model_registry()
        chat_cfg = registry.get_chat_model(model_id)
        pricing = chat_cfg.pricing
        cost = (
            prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m
        ) / 1_000_000.0
        ESTIMATED_COST.labels(model=model_id).inc(cost)
    except Exception:
        pass


def record_retrieval_score(score: float) -> None:
    """Record retrieved chunk score in Prometheus histogram."""
    RETRIEVAL_SCORES.observe(score)


def record_refusal(language: str = "ar") -> None:
    """Record canonical refusal event in Prometheus counter."""
    REFUSALS_TOTAL.labels(language=language).inc()


def record_faithfulness_score(score: float) -> None:
    """Update RAGAS faithfulness gauge."""
    RAGAS_FAITHFULNESS_GAUGE.set(score)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware collecting HTTP request count and latency metrics."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        start_time = time.perf_counter()
        endpoint = request.url.path

        # Avoid polluting metrics with high cardinality on article IDs
        if endpoint.startswith("/articles/"):
            endpoint = "/articles/{number}"

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)

        return cast(Response, response)


def metrics_endpoint() -> Response:
    """Return raw Prometheus metrics payload."""
    payload: bytes = generate_latest()
    return Response(
        content=payload,
        media_type=CONTENT_TYPE_LATEST,
    )
