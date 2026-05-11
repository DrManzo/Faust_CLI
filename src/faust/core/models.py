"""Core data models and LangGraph state definition for Faust."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TypedDict, Literal, List

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Return the format expected by Ollama/OpenAI-compatible APIs."""
        return {"role": self.role.value, "content": self.content}


class Turn(BaseModel):
    """One complete exchange: a user message and the assistant response."""

    user_message: Message
    assistant_message: Message | None = None


class Session(BaseModel):
    """A full conversation session."""

    id: str
    model: str
    turns: List[Turn] = Field(default_factory=list)
    system_prompt: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OllamaSettings(BaseModel):
    """Settings for the local Ollama backend."""

    base_url: str = "http://localhost:11434"
    request_timeout: int = 120


class OpenAICompatConfig(BaseModel):
    """Settings for an OpenAI-compatible local API server."""

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local"


class SqliteSettings(BaseModel):
    """Settings for optional SQLite checkpoint persistence."""

    path: str = "data/faust.db"


class AppConfig(BaseModel):
    """Runtime configuration loaded from configs/default.yaml."""

    # Core model/backend settings
    model: str = "llama3:8b"
    backend: Literal["ollama", "openai_compat"] = "ollama"
    temperature: float = 0.7
    context_window: int = 8192

    # High-level system prompt for the assistant
    system_prompt: str = "You are Faust, a local AI assistant."

    # Checkpointing / persistence
    checkpointer_backend: Literal["memory", "sqlite"] = "memory"

    # Backend-specific nested configs
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    sqlite: SqliteSettings = Field(default_factory=SqliteSettings)


class FaustState(TypedDict):
    """Shared state object passed between all LangGraph nodes.

    Every node reads from FaustState and returns a partial dict of updates.
    """

    session: Session
    config: AppConfig
    user_input: str
    messages: List[Message]
    response: str
    error: str | None