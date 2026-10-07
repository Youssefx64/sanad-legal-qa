"""Application settings using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Sanad application settings loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & LLM endpoints
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openai_compat_base_url: str | None = Field(default=None, alias="OPENAI_COMPAT_BASE_URL")
    openai_compat_api_key: str | None = Field(default=None, alias="OPENAI_COMPAT_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # Security
    admin_api_key: str = Field(default="sanad-admin-secret-key", alias="ADMIN_API_KEY")
    cors_origins: list[str] | str = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        alias="CORS_ORIGINS",
    )

    # Vector store
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Observability & Tracking
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")

    # Runtime & Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Paths (relative to repo root or configurable)
    repo_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent.parent
    )
    models_config_path: Path | None = None
    rag_config_path: Path | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Convert comma-separated CORS origins into a list of strings."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    def get_models_path(self) -> Path:
        """Resolve path to models.yaml."""
        if self.models_config_path:
            return self.models_config_path
        candidate = self.repo_root / "config" / "models.yaml"
        if candidate.exists():
            return candidate
        # Fallback to current working directory
        return Path("config/models.yaml").resolve()

    def get_rag_path(self) -> Path:
        """Resolve path to rag.yaml."""
        if self.rag_config_path:
            return self.rag_config_path
        candidate = self.repo_root / "config" / "rag.yaml"
        if candidate.exists():
            return candidate
        return Path("config/rag.yaml").resolve()

    def get_data_dir(self) -> Path:
        """Resolve path to data directory."""
        candidate = self.repo_root / "data"
        if candidate.exists():
            return candidate
        return Path("data").resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
