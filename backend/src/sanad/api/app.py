"""FastAPI application initialization, routing, and lifecycle management."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from sanad.api.errors import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from sanad.api.routes.admin import router as admin_router
from sanad.api.routes.ask import router as ask_router
from sanad.api.routes.corpus import router as corpus_router
from sanad.api.routes.health import router as health_router
from sanad.api.routes.models import router as models_router
from sanad.config.settings import Settings, get_settings
from sanad.observability.metrics import PrometheusMiddleware, metrics_endpoint
from sanad.rag import SanadRAG

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware attaching X-Request-ID and X-Response-Time-Ms headers."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        response: Response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifespan management."""
    settings = get_settings()
    logger.info("Starting Sanad Legal QA API in '%s' environment...", settings.environment)

    # Initialize RAG singleton on app state
    rag = SanadRAG(settings=settings)
    app.state.rag = rag
    app.state.settings = settings

    yield

    logger.info("Shutting down Sanad Legal QA API...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    active_settings = settings or get_settings()

    app = FastAPI(
        title="Sanad Legal Q&A API (سند)",
        description="Arabic Legal Question Answering System over the Egyptian Civil Code (1948).",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 1. Middlewares
    # CORS
    cors_origins = (
        active_settings.cors_origins
        if isinstance(active_settings.cors_origins, list)
        else [active_settings.cors_origins]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Request Tracking
    app.add_middleware(RequestContextMiddleware)
    # Prometheus Metrics
    app.add_middleware(PrometheusMiddleware)

    # 2. Exception Handlers
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_exception_handler)

    # 3. Router Registration
    app.include_router(health_router)
    app.include_router(ask_router)
    app.include_router(models_router)
    app.include_router(corpus_router)
    app.include_router(admin_router)

    # 4. Metrics Endpoint
    @app.get("/metrics", tags=["Observability"])
    async def get_metrics() -> Response:
        return metrics_endpoint()

    return app


app = create_app()
