"""OpenRouter provider implementation for Chat and Embeddings."""

from sanad.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)


class OpenRouterChatProvider(OpenAICompatibleChatProvider):
    """Chat completion provider for OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://openrouter.ai/api/v1",
        default_temperature: float = 0.1,
        default_max_tokens: int = 1024,
        site_url: str = "https://github.com/sanad-legal-qa",
        app_name: str = "Sanad Legal QA",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        }
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        )


class OpenRouterEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://openrouter.ai/api/v1",
        dimension: int | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        site_url: str = "https://github.com/sanad-legal-qa",
        app_name: str = "Sanad Legal QA",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        }
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            dimension=dimension,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            batch_size=batch_size,
            extra_headers=extra_headers,
            timeout=timeout,
            max_retries=max_retries,
        )
