"""OpenAI-compatible local API adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from faust.adapters.base import LLMAdapter
from faust.core.models import AppConfig, Message
from faust.exceptions import BackendError


class OpenAICompatAdapter(LLMAdapter):
    """LLM adapter for any OpenAI-compatible local API server."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._client = AsyncOpenAI(
            base_url=config.openai_compat.base_url,
            api_key=config.openai_compat.api_key,
        )

    async def stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        payload = [m.to_dict() for m in messages]
        try:
            async for chunk in await self._client.chat.completions.create(
                model=self.config.model,
                messages=payload,
                stream=True,
                temperature=self.config.temperature,
            ):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:  # pragma: no cover - network errors
            raise BackendError(f"OpenAI-compat stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
