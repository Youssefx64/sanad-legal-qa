"""Provider interfaces and implementations for Sanad Legal Q&A."""

from sanad.providers.base import (
    ChatMessage,
    ChatProvider,
    ChatResponse,
    ChatStreamChunk,
    ChatUsage,
    EmbeddingProvider,
    EmbeddingResponse,
)
from sanad.providers.factory import (
    ProviderFactory,
    get_provider_factory,
    reset_provider_factory,
)
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

__all__ = [
    "ChatMessage",
    "ChatProvider",
    "ChatResponse",
    "ChatStreamChunk",
    "ChatUsage",
    "EmbeddingProvider",
    "EmbeddingResponse",
    "LocalHFEmbeddingProvider",
    "OllamaChatProvider",
    "OllamaEmbeddingProvider",
    "OpenAICompatibleChatProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OpenRouterChatProvider",
    "OpenRouterEmbeddingProvider",
    "ProviderFactory",
    "get_provider_factory",
    "reset_provider_factory",
]

