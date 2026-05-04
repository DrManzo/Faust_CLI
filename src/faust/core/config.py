"""Faust configuration loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Dict
import yaml


BackendType = Literal["ollama", "openai_compat"]


@dataclass
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    request_timeout: int = 120


@dataclass
class OpenAICompatSettings:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local"


@dataclass
class FaustConfig:
    backend: BackendType = "ollama"
    model: str = "llama3:8b"
    temperature: float = 0.7
    context_window: int = 8192
    system_prompt: str = ""

    ollama: OllamaSettings = OllamaSettings()
    openai_compat: OpenAICompatSettings = OpenAICompatSettings()


def load_config(path: Path | None = None) -> FaustConfig:
    if path is None:
        path = Path("configs/default.yaml")  # important: the YAML file, not a .py file

    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    backend = data.get("backend", "ollama")
    model = data.get("model", "llama3:8b")
    temperature = float(data.get("temperature", 0.7))
    context_window = int(data.get("context_window", 8192))
    system_prompt = data.get("system_prompt", "") or ""

    ollama_raw = data.get("ollama", {}) or {}
    ollama = OllamaSettings(
        base_url=ollama_raw.get("base_url", "http://localhost:11434"),
        request_timeout=int(ollama_raw.get("request_timeout", 120)),
    )

    openai_raw = data.get("openai_compat", {}) or {}
    openai_compat = OpenAICompatSettings(
        base_url=openai_raw.get("base_url", "http://localhost:1234/v1"),
        api_key=openai_raw.get("api_key", "local"),
    )

    return FaustConfig(
        backend=backend,
        model=model,
        temperature=temperature,
        context_window=context_window,
        system_prompt=system_prompt,
        ollama=ollama,
        openai_compat=openai_compat,
    )