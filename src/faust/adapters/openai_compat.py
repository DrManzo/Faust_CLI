"""OpenAI-compatible local API adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, Dict, Iterator, List, Optional

import httpx
from openai import AsyncOpenAI

from faust.adapters.base import LLMAdapter
from faust.core.models import AppConfig, Message
from faust.exceptions import BackendError


class OpenAICompatAdapter(LLMAdapter):
    """LLM adapter for any OpenAI-compatible local API server.

    Supports both the async streaming path (used by health checks and future
    async callers) and the synchronous generate() / generate_structured()
    path used by the LangGraph nodes in graph.py.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._client = AsyncOpenAI(
            base_url=config.openai_compat.base_url,
            api_key=config.openai_compat.api_key,
        )
        self._base_url = config.openai_compat.base_url.rstrip("/")
        self._api_key = config.openai_compat.api_key
        self._model = config.models.default
        self._temperature = config.temperature
        self._context_window = config.context_window

    # ------------------------------------------------------------------
    # Synchronous interface (used by graph nodes via functools.partial)
    # ------------------------------------------------------------------

    def generate(
        self, messages: List[Dict[str, Any]], stream: bool = True
    ) -> Iterator[str]:
        """Yield response tokens via the OpenAI-compatible /chat/completions endpoint.

        Uses httpx directly for synchronous streaming so graph nodes can
        consume tokens without an event loop.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "temperature": self._temperature,
        }
        url = f"{self._base_url}/chat/completions"
        try:
            with httpx.Client(timeout=120) as client:
                if stream:
                    with client.stream("POST", url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                line = line[6:]
                            try:
                                data = json.loads(line)
                                delta = data["choices"][0]["delta"].get("content")
                                if delta:
                                    yield delta
                            except Exception:
                                continue
                else:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"].get("content", "")
                    yield content
        except Exception as exc:
            raise BackendError(f"OpenAI-compat generate failed: {exc}") from exc

    def generate_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a single parsed JSON object constrained by schema.

        Uses OpenAI's response_format with json_schema when supported,
        falling back to json_object mode if the schema is rejected.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        model = model_override or self._model
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        url = f"{self._base_url}/chat/completions"
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code == 400:
                    # Fall back to plain json_object mode
                    payload["response_format"] = {"type": "json_object"}
                    response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content", "")
                clean = content.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                return json.loads(clean)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Async interface (health checks, future async callers)
    # ------------------------------------------------------------------

    async def stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        payload = [m.to_dict() for m in messages]
        try:
            async for chunk in await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                stream=True,
                temperature=self._temperature,
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
