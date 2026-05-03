"""Ollama local backend adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import ollama as ollama_sdk

from faust.adapters.base import LLMAdapter
from faust.core.models import AppConfig, Message
from faust.exceptions import BackendError


class OllamaAdapter(LLMAdapter):
    """LLM adapter for a locally running Ollama instance."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._client = ollama_sdk.AsyncClient()

    async def stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        payload = [m.to_dict() for m in messages]
        try:
            async for chunk in await self._client.chat(
                model=self.config.model,
                messages=payload,
                stream=True,
                options={"temperature": self.config.temperature},
            ):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as exc:  # pragma: no cover - network errors
            raise BackendError(f"Ollama stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            await self._client.list()
            return True
        except Exception:
            return False
