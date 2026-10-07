"""OpenAI-compatible HTTP provider for Chat and Embeddings."""

import asyncio
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


def _is_unrecoverable_quota_error(e: Exception) -> bool:
    """Detect if an API error is due to an exhausted account quota (cannot be resolved by retrying)."""
    if isinstance(e, httpx.HTTPStatusError):
        resp = e.response
        if resp is not None:
            text = resp.text.lower()
            if any(
                term in text
                for term in ("free-models-per-day", "quota", "credit", "insufficient", "exceeded")
            ):
                return True
    return False


class OpenAICompatibleChatProvider(ChatProvider):
    """Chat provider speaking standard OpenAI `/chat/completions` protocol."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        default_temperature: float = 0.1,
        default_max_tokens: int = 1024,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Call `/chat/completions` with retries and exponential backoff."""
        payload = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": False,
        }
        payload.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload, headers=self._headers())
                    resp.raise_for_status()
                    data = resp.json()

                    choice = data["choices"][0]
                    content = choice["message"]["content"] or ""
                    finish_reason = choice.get("finish_reason")

                    usage_data = data.get("usage", {})
                    usage = ChatUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    )

                    return ChatResponse(
                        content=content,
                        usage=usage,
                        model=self.model_name,
                        finish_reason=finish_reason,
                    )
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                if _is_unrecoverable_quota_error(e):
                    logger.warning("Unrecoverable quota error encountered; skipping retries: %s", e)
                    raise
                # Retry on 429 or 5xx
                status = getattr(getattr(e, "response", None), "status_code", 500)
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Chat request failed with %s (attempt %d/%d). Retrying in %ds...",
                        e,
                        attempt + 1,
                        self.max_retries,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Chat request failed after retries")

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Stream SSE chunks from `/chat/completions`."""
        payload = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": True,
        }
        payload.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with (
                    httpx.AsyncClient(timeout=self.timeout) as client,
                    client.stream("POST", url, json=payload, headers=self._headers()) as response,
                ):
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            yield ChatStreamChunk(content="", is_final=True)
                            break

                        try:
                            chunk_json = json.loads(data_str)
                            choices = chunk_json.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content_piece = delta.get("content", "")
                            if content_piece:
                                yield ChatStreamChunk(content=content_piece, is_final=False)
                        except json.JSONDecodeError:
                            continue
                    return
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                if _is_unrecoverable_quota_error(e):
                    logger.warning(
                        "Unrecoverable quota error encountered in stream; skipping retries: %s", e
                    )
                    raise
                status = getattr(getattr(e, "response", None), "status_code", 500)
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Chat stream request failed with %s (attempt %d/%d). Retrying in %ds...",
                        e,
                        attempt + 1,
                        self.max_retries,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        if last_exception:
            raise last_exception


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding provider speaking standard OpenAI `/embeddings` protocol."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        dimension: int | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._dimension = dimension
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    async def embed(
        self,
        texts: list[str],
        is_query: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed a list of text strings in batches with retries."""
        if not texts:
            return []

        prefix = self.query_prefix if is_query else self.document_prefix
        prefixed_texts = [f"{prefix}{t}" if prefix else t for t in texts]

        all_embeddings: list[list[float]] = []
        url = f"{self.base_url}/embeddings"

        # Process in batches
        for i in range(0, len(prefixed_texts), self.batch_size):
            batch = prefixed_texts[i : i + self.batch_size]
            payload = {"model": self.model_name, "input": batch}
            payload.update(kwargs)

            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(url, json=payload, headers=self._headers())
                        resp.raise_for_status()
                        data = resp.json()

                        # OpenAI embeddings return sorted by index
                        batch_embeds = [
                            item["embedding"]
                            for item in sorted(data["data"], key=lambda x: x["index"])
                        ]
                        all_embeddings.extend(batch_embeds)

                        # Auto-detect dimension on first result if not set
                        if self._dimension is None and batch_embeds:
                            self._dimension = len(batch_embeds[0])
                        break
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    if _is_unrecoverable_quota_error(e):
                        logger.warning(
                            "Unrecoverable quota error encountered in embed; skipping retries: %s",
                            e,
                        )
                        raise
                    status = getattr(getattr(e, "response", None), "status_code", 500)
                    if status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                        wait_time = 2**attempt
                        await asyncio.sleep(wait_time)
                    else:
                        raise

        return all_embeddings
