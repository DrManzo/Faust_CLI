"""Tests for OllamaAdapter using mocked httpx (no real Ollama needed)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

from faust.adapters.ollama import OllamaAdapter


class FakeOllamaConfig:
    """Minimal config stub matching the attributes used by OllamaAdapter."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        request_timeout: float = 30.0,
        temperature: float = 0.7,
        context_window: int = 8192,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.ollama = SimpleNamespace(
            base_url=base_url,
            request_timeout=request_timeout,
        )


class DummyStreamResponse:
    """Mimic the context-managed streaming response returned by httpx."""

    def __init__(self, lines: List[Dict[str, Any]]) -> None:
        self._raw_lines = [json.dumps(line) for line in lines]

    def __enter__(self) -> "DummyStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for line in self._raw_lines:
            yield line


def test_ollama_adapter_generate_yields_streamed_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    """generate() should yield content chunks from Ollama's streaming response."""

    messages_payload: List[Dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
    ]
    fake_chunks = [
        {"message": {"content": "Hello "}},
        {"message": {"content": "world"}},
    ]

    def fake_stream(self, method: str, url: str, json: Dict[str, Any]):
        assert method == "POST"
        assert url == "http://localhost:11434/api/chat"
        assert json["model"] == "llama3:8b"
        assert json["messages"] == messages_payload
        assert json["stream"] is True
        assert json["options"]["temperature"] == 0.7
        assert json["options"]["num_ctx"] == 8192
        return DummyStreamResponse(fake_chunks)

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)

    adapter = OllamaAdapter(FakeOllamaConfig())
    chunks = list(adapter.generate(messages_payload, stream=True))

    assert chunks == ["Hello ", "world"]


def test_ollama_adapter_ignores_lines_without_message_content(
    monkeypatch: pytest.MonkeyPatch,
):
    """generate() should ignore streamed lines that do not contain message.content."""

    messages_payload: List[Dict[str, Any]] = [
        {"role": "user", "content": "Test"},
    ]
    fake_chunks = [
        {"foo": "bar"},
        {"message": {"content": "valid"}},
        {"message": {}},
    ]

    def fake_stream(self, method: str, url: str, json: Dict[str, Any]):
        return DummyStreamResponse(fake_chunks)

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)

    adapter = OllamaAdapter(FakeOllamaConfig())
    chunks = list(adapter.generate(messages_payload, stream=True))

    assert chunks == ["valid"]