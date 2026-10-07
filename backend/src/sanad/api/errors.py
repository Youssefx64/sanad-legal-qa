"""Error handling and custom API exceptions."""

import logging
from datetime import UTC, datetime

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ArticleNotFoundError(HTTPException):
    """Raised when a requested article number does not exist in the Code."""

    def __init__(self, article_number: int) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_number} not found in the Egyptian Civil Code (valid range: 1..1149).",
        )


class UnauthorizedAdminError(HTTPException):
    """Raised when an invalid admin API key is supplied."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
        )


def format_error_response(
    detail: str,
    error_code: str,
    status_code: int,
) -> JSONResponse:
    """Format standard error envelope."""
    payload = {
        "detail": detail,
        "error_code": error_code,
        "status_code": status_code,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(status_code=status_code, content=payload)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTPException with structured envelope."""
    error_code = f"HTTP_{exc.status_code}"
    return format_error_response(
        detail=str(exc.detail),
        error_code=error_code,
        status_code=exc.status_code,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with structured envelope."""
    error_msgs = [
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
    ]
    detail = "; ".join(error_msgs)
    return format_error_response(
        detail=detail,
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected internal exceptions."""
    logger.exception("Unhandled server error: %s", exc)
    return format_error_response(
        detail="An internal server error occurred while processing the request.",
        error_code="INTERNAL_SERVER_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
