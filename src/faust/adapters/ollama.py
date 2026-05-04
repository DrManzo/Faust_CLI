"""Ollama adapter for Faust."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List

import httpx


class OllamaAdapter:
    """Streams responses from a local Ollama instance."""

    def __init__(self, config) -> None:
        self.base_url = config.ollama.base_url.rstrip("/")
        self.model = config.model
        self.timeout = config.ollama.request_timeout
        self.temperature = config.temperature
        self.context_window = config.context_window

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
