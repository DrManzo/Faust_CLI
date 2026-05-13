"""Core data models and LangGraph state definition for Faust."""


from __future__ import annotations


from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal


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



class ModelRouteConfig(BaseModel):
    """Per-role model routing configuration."""

    default: str = "llama3.3:8b"      # General chat, memory, assistant
    coder: str = "qwen2.5-coder:14b"  # Code-focused tasks, test drafting
    planner: str = "deepseek-r1:8b"   # Reasoning, planning, decomposition



class AppConfig(BaseModel):
    """Runtime configuration loaded from configs/default.yaml."""


    # Legacy single-model field kept for backward compatibility with tests and
    # any code that still reads config.model directly. When a models block is
    # present in the YAML this field is ignored in favour of models.default.
    model: str = "llama3.3:8b"

    models: ModelRouteConfig = Field(default_factory=ModelRouteConfig)

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
    - Ephemeral state: transient routing hints, retrieval diagnostics, planning notes,
      and scoped test requests that should not be treated as durable memory.

    Phase 3 additions (tool_call dispatch):
    - tool_name   : name of the registered PluginRegistry tool to dispatch.
    - tool_inputs : keyword arguments forwarded verbatim to the plugin function.
    - tool_result : raw return value written back by tool_call_node after dispatch.
    """


    # Checkpoint memory: session continuity and runtime configuration.
    session: Session
    config: AppConfig
    user_id: str
    user_input: str


    # Checkpoint memory: normalized turn outputs, routing outcomes, and resumable state.
    intent: str | None
    active_agent: str | None
    response: str
    error: str | None


    # Checkpoint memory: conversation context and recalled durable memories projected
    # into the current turn.
    messages: List[Message]
    recalled_memories: List[MemoryRecord]
    artifacts: List[str]


    # Ephemeral state: transient routing, retrieval, planning, correction, and
    # execution hints.
    memory_route: str | None
    requested_role: NotRequired[str | None]
    task_type: NotRequired[str | None]
    memory_query: NotRequired[str | None]
    memory_hits: NotRequired[List[MemoryRecord]]
    ephemeral_context: NotRequired[dict[str, str] | None]
    requested_tests: NotRequired[List[str]]
    execution_notes: NotRequired[str | None]

    # Step 9: tests-only workflow ephemeral state.
    # test_proposal: proposed test file content shown to the user for review.
    # test_approved: must be True before run_requested_tests executes anything.
    # test_report_path: path of the written report file for this turn.
    test_proposal: NotRequired[str | None]
    test_approved: NotRequired[bool]
    test_report_path: NotRequired[str | None]

    # Step 11 Phase 2: edit-and-propose cycle.
    # edit_proposal: an EditProposal instance or None; carried through the loop.
    edit_proposal: NotRequired[Any]

    # Step 11 Phase 3: plugin/tool dispatch fields.
    # tool_name   : registered PluginRegistry name to dispatch (str | None).
    # tool_inputs : kwargs forwarded to the plugin fn (empty dict if unused).
    # tool_result : raw return value after dispatch (Any | None).
    tool_name: NotRequired[str | None]
    tool_inputs: NotRequired[dict]
    tool_result: NotRequired[Any]
