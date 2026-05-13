"""Ollama adapter for Faust."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional

import httpx

from faust.adapters.base import LLMAdapter
from faust.core.models import AppConfig, Message


class OllamaAdapter(LLMAdapter):
    """Streams responses from a local Ollama instance."""

    def __init__(self, config: AppConfig, model_override: Optional[str] = None) -> None:
        super().__init__(config)
        self.base_url = config.ollama.base_url.rstrip("/")
        # model_override lets callers pin a specific model for a role
        # (e.g. coder → qwen2.5-coder:14b) without touching the base config.
        self.model = model_override or config.models.default
        self.timeout = config.ollama.request_timeout
        self.temperature = config.temperature
        self.context_window = config.context_window

    # ------------------------------------------------------------------
    # Synchronous streaming (used by graph nodes)
    # ------------------------------------------------------------------

    def generate(
        self, messages: List[Dict[str, Any]], stream: bool = True
    ) -> Iterator[str]:
        """Yield response tokens from Ollama /api/chat."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_window,
            },
        }
        url = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

    # ------------------------------------------------------------------
    # Structured output (JSON schema enforcement via Ollama format field)
    # ------------------------------------------------------------------

    def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a single parsed JSON object constrained by schema.

        Passes the schema directly to Ollama's ``format`` field, which uses
        grammar-constrained decoding to guarantee the output matches the
        schema without any post-hoc regex or prompt tricks.

        Args:
            messages:       Chat message list in {role, content} dict form.
            schema:         A JSON Schema object describing the expected output.
            model_override: Optional model name override for this call only.

        Returns:
            Parsed dict matching the schema, or an empty dict on failure.
        """
        model = model_override or self.model
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.0,   # deterministic for structured extraction
                "num_ctx": self.context_window,
            },
        }
        url = f"{self.base_url}/api/chat"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return json.loads(content)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Async interface (OpenAI-compat path, health checks)
    # ------------------------------------------------------------------

    async def stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """Async streaming — not used by graph nodes but satisfies the base contract."""
        import asyncio
        import functools
        loop = asyncio.get_event_loop()
        message_dicts = [m.to_dict() for m in messages]
        # Run the sync generator in a thread to avoid blocking the event loop.
        chunks = await loop.run_in_executor(
            None,
            functools.partial(list, self.generate(message_dicts, stream=True)),
        )
        for chunk in chunks:
            yield chunk

    async def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
