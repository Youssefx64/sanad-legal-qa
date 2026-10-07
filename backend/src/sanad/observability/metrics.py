"""Prometheus metrics collectors and middleware instrumentation."""

import time
from collections.abc import Callable
from typing import Any, cast

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

# Metrics definitions
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

TOKEN_USAGE = Counter(
    "sanad_tokens_total",
    "Total tokens consumed across chat completions",
    ["model", "type"],  # type: 'prompt' or 'completion'
)

GROUNDING_FAILURES = Counter(
    "sanad_grounding_interventions_total",
    "Number of times the grounding guardrail suppressed ungrounded hallucinations",
)


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
