"""Abstract base class for all LLM backend adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict, Iterator, List, Optional

from faust.core.models import AppConfig, Message


class LLMAdapter(ABC):
    """Abstract contract for all LLM backend adapters."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self, messages: List[Dict[str, Any]], stream: bool = True
    ) -> Iterator[str]:
        """Yield response tokens as a synchronous iterator."""
        ...

    def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a single parsed JSON object constrained by schema.

        Default implementation collects all tokens from generate() and
        json.loads() the result. Subclasses that support native structured
        outputs (e.g. Ollama format field, OpenAI response_format) should
        override this to use the backend's first-class schema enforcement.

        Args:
            messages:  Chat message list in {role, content} dict form.
            schema:    A JSON Schema object describing the expected output.
            model_override: Optional model name to use for this call only.

        Returns:
            Parsed dict matching the schema, or an empty dict on failure.
        """
        import json
        full = "".join(self.generate(messages, stream=False))
        try:
            # Strip markdown code fences if the model wrapped the JSON.
            clean = full.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(clean)
        except Exception:
            return {}

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """Yield response tokens as an async generator (OpenAI-compat path)."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
        ...
