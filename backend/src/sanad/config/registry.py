"""Model Registry loader and validator for config/models.yaml."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelPricing(BaseModel):
    """Token pricing configuration per 1 million tokens."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0


class ChatModelConfig(BaseModel):
    """Configuration for a chat LLM."""

    id: str
    label: str
    provider: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    supports_streaming: bool = True
    enabled: bool = True


class EmbeddingModelConfig(BaseModel):
    """Configuration for an embedding model."""

    id: str
    label: str
    provider: str
    model: str
    dimension: int | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 64
    device: str = "auto"
    enabled: bool = True


class ProviderConfig(BaseModel):
    """Configuration for a model provider endpoint."""

    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None

    def resolve_base_url(self) -> str | None:
        """Resolve base URL from static config or environment variable."""
        if self.base_url:
            return self.base_url
        if self.base_url_env:
            return os.getenv(self.base_url_env)
        return None

    def resolve_api_key(self) -> str | None:
        """Resolve API key from environment variable."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


class DefaultsConfig(BaseModel):
    """Default model identifiers."""

    chat_model: str
    embedding_model: str
    judge_model: str


class ModelRegistry(BaseModel):
    """Root model registry mapping."""

    defaults: DefaultsConfig
    chat_models: list[ChatModelConfig]
    embedding_models: list[EmbeddingModelConfig]
    providers: dict[str, ProviderConfig]

    @model_validator(mode="after")
    def validate_registry_integrity(self) -> "ModelRegistry":
        """Verify unique IDs, default model existence, and provider mapping."""
        # 1. Unique chat IDs
        chat_ids = [m.id for m in self.chat_models]
        if len(chat_ids) != len(set(chat_ids)):
            duplicates = [cid for cid in chat_ids if chat_ids.count(cid) > 1]
            raise ValueError(f"Duplicate chat_model id(s) detected: {set(duplicates)}")

        # 2. Unique embedding IDs
        embed_ids = [m.id for m in self.embedding_models]
        if len(embed_ids) != len(set(embed_ids)):
            duplicates = [eid for eid in embed_ids if embed_ids.count(eid) > 1]
            raise ValueError(f"Duplicate embedding_model id(s) detected: {set(duplicates)}")

        # 3. Default models exist
        chat_id_set = set(chat_ids)
        embed_id_set = set(embed_ids)

        if self.defaults.chat_model not in chat_id_set:
            raise ValueError(
                f"Default chat_model '{self.defaults.chat_model}' not found in chat_models"
            )
        if self.defaults.embedding_model not in embed_id_set:
            raise ValueError(
                f"Default embedding_model '{self.defaults.embedding_model}' not found in embedding_models"
            )
        if self.defaults.judge_model not in chat_id_set:
            raise ValueError(
                f"Default judge_model '{self.defaults.judge_model}' not found in chat_models"
            )

        # 4. Providers mapped for enabled models
        for chat in self.chat_models:
            if chat.enabled and chat.provider not in self.providers:
                raise ValueError(
                    f"Chat model '{chat.id}' specifies unknown provider '{chat.provider}'"
                )

        for embed in self.embedding_models:
            if embed.enabled and embed.provider not in self.providers:
                raise ValueError(
                    f"Embedding model '{embed.id}' specifies unknown provider '{embed.provider}'"
                )

        return self

    def validate_runtime_keys(self, allow_missing: bool = False) -> list[str]:
        """Check if active providers have required API keys set. Returns missing env var names."""
        missing = []
        for chat in self.chat_models:
            if not chat.enabled:
                continue
            prov = self.providers.get(chat.provider)
            if prov and prov.api_key_env and not os.getenv(prov.api_key_env):
                missing.append(f"{chat.id}: missing env {prov.api_key_env}")

        for embed in self.embedding_models:
            if not embed.enabled:
                continue
            prov = self.providers.get(embed.provider)
            if prov and prov.api_key_env and not os.getenv(prov.api_key_env):
                missing.append(f"{embed.id}: missing env {prov.api_key_env}")

        if missing and not allow_missing:
            raise RuntimeError("Missing API keys for enabled models:\n" + "\n".join(missing))
        return missing

    def get_chat_model(self, model_id: str | None = None) -> ChatModelConfig:
        """Fetch chat model config by ID or default."""
        target_id = model_id or self.defaults.chat_model
        for model in self.chat_models:
            if model.id == target_id:
                if not model.enabled:
                    raise ValueError(f"Chat model '{target_id}' is disabled.")
                return model
        raise KeyError(
            f"Chat model '{target_id}' not found. Available: {[m.id for m in self.chat_models if m.enabled]}"
        )

    def get_embedding_model(self, model_id: str | None = None) -> EmbeddingModelConfig:
        """Fetch embedding model config by ID or default."""
        target_id = model_id or self.defaults.embedding_model
        for model in self.embedding_models:
            if model.id == target_id:
                if not model.enabled:
                    raise ValueError(f"Embedding model '{target_id}' is disabled.")
                return model
        raise KeyError(
            f"Embedding model '{target_id}' not found. Available: {[m.id for m in self.embedding_models if m.enabled]}"
        )

    def get_provider(self, provider_name: str) -> ProviderConfig:
        """Fetch provider config by name."""
        if provider_name not in self.providers:
            raise KeyError(f"Provider '{provider_name}' not defined in registry")
        return self.providers[provider_name]

    @property
    def enabled_chat_models(self) -> list[ChatModelConfig]:
        """List all currently enabled chat models."""
        return [m for m in self.chat_models if m.enabled]

    @property
    def enabled_embedding_models(self) -> list[EmbeddingModelConfig]:
        """List all currently enabled embedding models."""
        return [m for m in self.embedding_models if m.enabled]


def load_model_registry(path: str | Path) -> ModelRegistry:
    """Load and validate models.yaml from the specified path."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model registry file not found at: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ModelRegistry(**data)
