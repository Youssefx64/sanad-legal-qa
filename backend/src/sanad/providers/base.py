"""Base interfaces and data models for LLM and Embedding providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatUsage(BaseModel):
    """Token usage tracking for a completion or embedding call."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    """Complete response returned by a ChatProvider."""

    content: str
    usage: ChatUsage
    model: str
    finish_reason: str | None = None


class ChatStreamChunk(BaseModel):
    """A streaming chunk emitted during token generation."""

    content: str
    is_final: bool = False
    usage: ChatUsage | None = None


class EmbeddingResponse(BaseModel):
    """Output from an EmbeddingProvider."""

    embeddings: list[list[float]]
    usage: ChatUsage | None = None
    dimension: int


class ChatProvider(ABC):
    """Abstract base class for all chat completion providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a complete response for the given conversation messages."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Stream response chunks asynchronously as they are generated."""


class EmbeddingProvider(ABC):
    """Abstract base class for all text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Return the vector dimensionality of this embedding model, or None if undetermined."""

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        is_query: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed a list of text strings into vector representations."""
