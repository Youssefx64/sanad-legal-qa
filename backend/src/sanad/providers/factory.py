"""Provider factory to instantiate ChatProvider and EmbeddingProvider from configuration."""

import os

from sanad.config.registry import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelRegistry,
    get_model_registry,
)
from sanad.config.settings import Settings, get_settings
from sanad.providers.base import ChatProvider, EmbeddingProvider
from sanad.providers.local_hf import LocalHFEmbeddingProvider
from sanad.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from sanad.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from sanad.providers.openrouter import (
    OpenRouterChatProvider,
    OpenRouterEmbeddingProvider,
)


class ProviderFactory:
    """Factory to instantiate and cache provider instances based on model registry."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_model_registry()
        self._chat_cache: dict[str, ChatProvider] = {}
        self._embed_cache: dict[str, EmbeddingProvider] = {}

    def _resolve_api_key(self, api_key_env: str | None) -> str:
        """Resolve API key from environment variable or settings."""
        if not api_key_env:
            return ""
        val = os.environ.get(api_key_env, "")
        if not val and hasattr(self.settings, api_key_env.lower()):
            val = getattr(self.settings, api_key_env.lower()) or ""
        return val

    def _resolve_base_url(self, base_url: str | None, base_url_env: str | None) -> str:
        """Resolve base URL from config or environment variable."""
        if base_url_env:
            val = os.environ.get(base_url_env, "")
            if val:
                return val
            if hasattr(self.settings, base_url_env.lower()):
                val = getattr(self.settings, base_url_env.lower()) or ""
                if val:
                    return val
        if base_url:
            return base_url
        return ""

    def get_chat_provider(self, model_id: str | None = None) -> ChatProvider:
        """Get or instantiate a ChatProvider for the requested model ID."""
        target_id = model_id or self.registry.defaults.chat_model
        if target_id in self._chat_cache:
            return self._chat_cache[target_id]

        cfg: ChatModelConfig = self.registry.get_chat_model(target_id)
        prov_cfg = self.registry.providers.get(cfg.provider)

        api_key = self._resolve_api_key(prov_cfg.api_key_env if prov_cfg else None)
        base_url = self._resolve_base_url(
            prov_cfg.base_url if prov_cfg else None,
            prov_cfg.base_url_env if prov_cfg else None,
        )

        provider_instance: ChatProvider
        if cfg.provider == "openrouter":
            provider_instance = OpenRouterChatProvider(
                api_key=api_key or "placeholder_key",
                model_name=cfg.model,
                base_url=base_url or "https://openrouter.ai/api/v1",
                default_temperature=cfg.params.get("temperature", 0.1),
                default_max_tokens=cfg.params.get("max_tokens", 1024),
            )
        elif cfg.provider == "openai_compatible":
            provider_instance = OpenAICompatibleChatProvider(
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key or "placeholder_key",
                model_name=cfg.model,
                default_temperature=cfg.params.get("temperature", 0.1),
                default_max_tokens=cfg.params.get("max_tokens", 1024),
            )
        elif cfg.provider == "ollama":
            provider_instance = OllamaChatProvider(
                base_url=base_url or "http://localhost:11434",
                model_name=cfg.model,
                default_temperature=cfg.params.get("temperature", 0.1),
                default_max_tokens=cfg.params.get("max_tokens", 1024),
            )
        else:
            raise ValueError(f"Unsupported chat provider type: {cfg.provider}")

        self._chat_cache[target_id] = provider_instance
        return provider_instance

    def get_embedding_provider(self, model_id: str | None = None) -> EmbeddingProvider:
        """Get or instantiate an EmbeddingProvider for the requested model ID."""
        target_id = model_id or self.registry.defaults.embedding_model
        if target_id in self._embed_cache:
            return self._embed_cache[target_id]

        cfg: EmbeddingModelConfig = self.registry.get_embedding_model(target_id)
        prov_cfg = self.registry.providers.get(cfg.provider)

        api_key = self._resolve_api_key(prov_cfg.api_key_env if prov_cfg else None)
        base_url = self._resolve_base_url(
            prov_cfg.base_url if prov_cfg else None,
            prov_cfg.base_url_env if prov_cfg else None,
        )

        provider_instance: EmbeddingProvider
        if cfg.provider == "openrouter":
            provider_instance = OpenRouterEmbeddingProvider(
                api_key=api_key or "placeholder_key",
                model_name=cfg.model,
                base_url=base_url or "https://openrouter.ai/api/v1",
                dimension=cfg.dimension,
                query_prefix=cfg.query_prefix,
                document_prefix=cfg.document_prefix,
                batch_size=cfg.batch_size,
            )
        elif cfg.provider == "openai_compatible":
            provider_instance = OpenAICompatibleEmbeddingProvider(
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key or "placeholder_key",
                model_name=cfg.model,
                dimension=cfg.dimension,
                query_prefix=cfg.query_prefix,
                document_prefix=cfg.document_prefix,
                batch_size=cfg.batch_size,
            )
        elif cfg.provider == "ollama":
            provider_instance = OllamaEmbeddingProvider(
                base_url=base_url or "http://localhost:11434",
                model_name=cfg.model,
                dimension=cfg.dimension,
            )
        elif cfg.provider == "local_hf":
            provider_instance = LocalHFEmbeddingProvider(
                model_name=cfg.model,
                dimension=cfg.dimension,
                batch_size=cfg.batch_size,
            )
        else:
            raise ValueError(f"Unsupported embedding provider type: {cfg.provider}")

        self._embed_cache[target_id] = provider_instance
        return provider_instance

    def clear_cache(self) -> None:
        """Clear cached provider instances."""
        self._chat_cache.clear()
        self._embed_cache.clear()


_global_factory: ProviderFactory | None = None


def get_provider_factory() -> ProviderFactory:
    """Return the global ProviderFactory singleton."""
    global _global_factory
    if _global_factory is None:
        _global_factory = ProviderFactory()
    return _global_factory


def reset_provider_factory() -> None:
    """Reset the global ProviderFactory singleton (useful in tests)."""
    global _global_factory
    _global_factory = None
