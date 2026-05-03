"""Abstract base class for all LLM backend adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from faust.core.models import AppConfig, Message


class LLMAdapter(ABC):
    """Abstract contract for all LLM backend adapters."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """Yield response tokens as an async generator."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
        ...
