"""FastAPI dependency injection utilities."""

from typing import Annotated

from fastapi import Header, Request

from sanad.api.errors import UnauthorizedAdminError
from sanad.config.registry import ModelRegistry, get_model_registry
from sanad.config.settings import Settings, get_settings
from sanad.rag import SanadRAG


def get_settings_dep() -> Settings:
    """Dependency provider for application settings."""
    return get_settings()


def get_model_registry_dep() -> ModelRegistry:
    """Dependency provider for model registry."""
    return get_model_registry()


def get_rag(request: Request) -> SanadRAG:
    """Retrieve the SanadRAG singleton stored on application state."""
    rag = getattr(request.app.state, "rag", None)
    if not isinstance(rag, SanadRAG):
        rag = SanadRAG()
        request.app.state.rag = rag
    return rag


def verify_admin_key(
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> str:
    """Verify administrative API key header."""
    settings = get_settings()
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise UnauthorizedAdminError()
    return x_admin_key
