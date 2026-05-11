"""Core data models and LangGraph state definition for Faust."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field

try:
    from typing import NotRequired, TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired, TypedDict


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
    """Settings for optional SQLite-backed persistence."""

    path: str = "data/faust.db"


class MemorySettings(BaseModel):
    """Settings for long-term durable memory."""

    enabled: bool = True
    backend: Literal["memory", "sqlite"] = "memory"
    namespace: str = "memories"
    max_results: int = 3


class MemoryRecord(BaseModel):
    """A durable user memory stored across sessions."""

    key: str
    slot: str | None = None
    text: str
    category: Literal["profile", "preference", "constraint", "fact"] = "fact"
    source: Literal["explicit", "inferred"] = "explicit"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AppConfig(BaseModel):
    """Runtime configuration loaded from configs/default.yaml."""

    model: str = "llama3:8b"
    backend: Literal["ollama", "openai_compat"] = "ollama"
    temperature: float = 0.7
    context_window: int = 8192

    system_prompt: str = "You are Faust, a local AI assistant."

    checkpointer_backend: Literal["memory", "sqlite"] = "memory"
    memory: MemorySettings = Field(default_factory=MemorySettings)

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    sqlite: SqliteSettings = Field(default_factory=SqliteSettings)


class FaustState(TypedDict):
    """Shared state object passed between all LangGraph nodes.

    Bucket guide:
    - Checkpoint memory: resumable thread/session state and normalized outputs.
    - Long-term memory: durable user facts live in the memory store as MemoryRecord
      entries; recalled_memories is only the per-turn projection of that store.
    - Ephemeral state: transient routing hints, planning notes, and scoped test
      requests that should not be treated as durable memory.
    """

    # Checkpoint memory: session continuity and runtime configuration.
    session: Session
    config: AppConfig
    user_id: str
    user_input: str

    # Checkpoint memory: normalized turn outputs and routing outcomes.
    intent: str | None
    active_agent: str | None
    response: str
    error: str | None

    # Checkpoint memory: conversation context and retrieved durable memories.
    messages: List[Message]
    recalled_memories: List[MemoryRecord]
    artifacts: List[str]

    # Ephemeral state: transient routing and execution hints.
    memory_route: str | None
    requested_role: NotRequired[str | None]
    task_type: NotRequired[str | None]
    ephemeral_context: NotRequired[dict[str, str] | None]
    requested_tests: NotRequired[List[str]]
    execution_notes: NotRequired[str | None]