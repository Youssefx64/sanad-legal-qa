"""Langfuse distributed tracing for Sanad legal Q&A pipeline."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from sanad.config.settings import get_settings

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.info("Langfuse library not available; using local telemetry tracer.")


class SpanRecord:
    """Telemetry span recording execution time and metadata."""

    def __init__(self, name: str, parent_trace_id: str | None = None) -> None:
        self.name = name
        self.trace_id = parent_trace_id or f"trace_{int(time.time() * 1000)}"
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self.metadata: dict[str, Any] = {}

    def __enter__(self) -> SpanRecord:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


class SanadTracer:
    """Manages distributed tracing via Langfuse with local fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(
            LANGFUSE_AVAILABLE and settings.langfuse_public_key and settings.langfuse_secret_key
        )
        self.client: Any = None
        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("Initialized Langfuse distributed tracer at %s", settings.langfuse_host)
            except Exception as e:
                logger.warning("Could not initialize Langfuse client: %s", e)
                self.enabled = False

    def start_trace(
        self,
        trace_id: str,
        name: str = "sanad_legal_qa",
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create or return active trace."""
        if not self.enabled or not self.client:
            return None

        with contextlib.suppress(Exception):
            return self.client.trace(
                id=trace_id,
                name=name,
                user_id=user_id,
                metadata=metadata or {},
            )
        return None

    def span(self, name: str, trace_id: str | None = None) -> SpanRecord:
        """Create timed context span."""
        return SpanRecord(name=name, parent_trace_id=trace_id)

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: Any,
        completion: str,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log LLM generation step to Langfuse."""
        if not self.enabled or not self.client:
            return

        with contextlib.suppress(Exception):
            trace = self.client.trace(id=trace_id)
            trace.generation(
                name=name,
                model=model,
                input=prompt,
                output=completion,
                usage=usage,
                metadata=metadata or {},
            )

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Attach evaluation metric (e.g. RAGAS faithfulness) to Langfuse trace."""
        if not self.enabled or not self.client:
            return

        with contextlib.suppress(Exception):
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )


_tracer_instance: SanadTracer | None = None


def get_tracer() -> SanadTracer:
    """Get singleton tracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = SanadTracer()
    return _tracer_instance
