"""Ollama local model provider for Chat and Embeddings."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from sanad.providers.base import (
    ChatMessage,
    ChatProvider,
    ChatResponse,
    ChatStreamChunk,
    ChatUsage,
    EmbeddingProvider,
)

logger = logging.getLogger(__name__)


class OllamaChatProvider(ChatProvider):
    """Chat provider connecting to local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b-instruct",
        default_temperature: float = 0.1,
        default_max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
            },
            "stream": False,
        }
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "")
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            return ChatResponse(
                content=content,
                usage=ChatUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                model=self.model_name,
                finish_reason="stop" if data.get("done") else None,
            )

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatStreamChunk]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
            },
            "stream": True,
        }
        payload.update(kwargs)

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream("POST", url, json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                chunk_data = json.loads(line)
                content_piece = chunk_data.get("message", {}).get("content", "")
                done = chunk_data.get("done", False)

                usage = None
                if done:
                    prompt_tokens = chunk_data.get("prompt_eval_count", 0)
                    completion_tokens = chunk_data.get("eval_count", 0)
                    usage = ChatUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )

                yield ChatStreamChunk(
                    content=content_piece,
                    is_final=done,
                    usage=usage,
                )


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider connecting to local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        dimension: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._dimension = dimension
        self.timeout = timeout

    @property
    def dimension(self) -> int | None:
        return self._dimension

    async def embed(
        self,
        texts: list[str],
        is_query: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        url = f"{self.base_url}/api/embeddings"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                payload = {"model": self.model_name, "prompt": text}
                payload.update(kwargs)
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                vec = data.get("embedding", [])
                embeddings.append(vec)
                if self._dimension is None and vec:
                    self._dimension = len(vec)

        return embeddings
