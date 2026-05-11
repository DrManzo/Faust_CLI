"""LangGraph state graph for Faust."""

from __future__ import annotations

import re
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

try:
    from langgraph.store.sqlite import SqliteStore
except ImportError:  # pragma: no cover
    SqliteStore = None

from faust.core.models import AppConfig, FaustState, MemoryRecord, Message, Role

_GRAPH_RESOURCES: list[ExitStack] = []


def make_memory_store(config: AppConfig):
    """Create the configured long-term memory store."""
    if not config.memory.enabled:
        return None

    if config.memory.backend == "sqlite":
        if SqliteStore is None:
            raise RuntimeError(
                "SQLite long-term memory requires langgraph.store.sqlite.SqliteStore"
            )

        db_path = Path(config.sqlite.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        memory_db_path = db_path.with_name("faust_memory.db")

        stack = ExitStack()
        store = stack.enter_context(
            SqliteStore.from_conn_string(str(memory_db_path))
        )
        store.setup()
        _GRAPH_RESOURCES.append(stack)
        return store

    return InMemoryStore()


def _memory_namespace(state: FaustState) -> tuple[str, str]:
    """Return the namespace tuple for durable user memory."""
    config = state["config"]
    user_id = state.get("user_id", "default")
    return (config.memory.namespace, user_id)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _memory_record_from_item(item) -> MemoryRecord | None:
    value = getattr(item, "value", {}) or {}
    try:
        return MemoryRecord(
            key=getattr(item, "key", value.get("key", str(uuid4()))),
            slot=value.get("slot"),
            text=value.get("text", ""),
            category=value.get("category", "fact"),
            source=value.get("source", "explicit"),
            created_at=value.get("created_at", datetime.now(timezone.utc)),
            updated_at=value.get("updated_at", datetime.now(timezone.utc)),
        )
    except Exception:
        return None


def _normalize_memory_text(memory_text: str) -> tuple[str, str]:
    """Normalize first-person user facts into stable third-person memory text."""
    text = memory_text.strip().rstrip(".")
    lowered = text.lower()

    if lowered.startswith("my favorite "):
        remainder = text[3:].strip()
        return (f"The user's {remainder}.", "preference")

    if lowered.startswith("i prefer "):
        preference = text[9:].strip()
        return (f"The user prefers {preference}.", "preference")

    if lowered.startswith("my name is "):
        name = text[11:].strip()
        return (f"The user's name is {name}.", "profile")

    if lowered.startswith("i was born on "):
        birthdate = text[14:].strip()
        return (f"The user's birthdate is {birthdate}.", "profile")

    if lowered.startswith("my birthdate is "):
        birthdate = text[16:].strip()
        return (f"The user's birthdate is {birthdate}.", "profile")

    return (f"The user said: {text}.", "fact")


def _detect_slot(memory_text: str) -> tuple[str | None, str]:
    normalized = _normalize_text(memory_text)
    original = memory_text.strip().rstrip(".")

    favorite_editor = re.match(
        r"my favorite editor is\s+(.+)", original, flags=re.IGNORECASE
    )
    if favorite_editor and normalized.startswith("my favorite editor is "):
        return "preference.favorite_editor", favorite_editor.group(1).strip()

    preferred_editor = re.match(
        r"i prefer\s+(.+)", original, flags=re.IGNORECASE
    )
    if preferred_editor and "editor" in normalized:
        return "preference.favorite_editor", preferred_editor.group(1).strip()

    user_name = re.match(
        r"my name is\s+(.+)", original, flags=re.IGNORECASE
    )
    if user_name and normalized.startswith("my name is "):
        return "profile.name", user_name.group(1).strip()

    born_on = re.match(
        r"i was born on\s+(.+)", original, flags=re.IGNORECASE
    )
    if born_on and normalized.startswith("i was born on "):
        return "profile.birthdate", born_on.group(1).strip()

    birthdate_is = re.match(
        r"my birthdate is\s+(.+)", original, flags=re.IGNORECASE
    )
    if birthdate_is and normalized.startswith("my birthdate is "):
        return "profile.birthdate", birthdate_is.group(1).strip()

    return None, original


def _query_keywords(query: str) -> set[str]:
    stopwords = {
        "what", "is", "my", "what's", "do", "i", "you", "remember",
        "did", "say", "about", "the", "a", "an", "are", "to", "of",
        "when", "was", "who", "am",
    }
    words = re.findall(r"[a-z0-9]+", _normalize_text(query))
    return {word for word in words if word not in stopwords}


def _score_memory(query: str, memory: MemoryRecord) -> tuple[int, float]:
    normalized_query = _normalize_text(query)
    normalized_text = _normalize_text(memory.text)
    query_terms = _query_keywords(query)
    text_terms = set(re.findall(r"[a-z0-9]+", normalized_text))

    score = 0

    if normalized_query and normalized_query in normalized_text:
        score += 100

    if memory.slot == "preference.favorite_editor" and "favorite editor" in normalized_query:
        score += 200

    if memory.slot == "profile.name" and (
        "my name" in normalized_query or "who am i" in normalized_query
    ):
        score += 200

    if memory.slot == "profile.birthdate" and (
        "when was i born" in normalized_query
        or "what is my birthdate" in normalized_query
        or "what's my birthdate" in normalized_query
    ):
        score += 220

    if "favorite editor" in normalized_query and "favorite editor" in normalized_text:
        score += 80

    if "birthdate" in normalized_query and "birthdate" in normalized_text:
        score += 80

    if "born" in normalized_query and "birthdate" in normalized_text:
        score += 80

    if "favorite" in normalized_query and "favorite" in normalized_text:
        score += 40

    overlap = len(query_terms & text_terms)
    score += overlap * 10

    timestamp = memory.updated_at.timestamp() if memory.updated_at else 0.0
    return score, timestamp


def _dedupe_memories(memories: list[MemoryRecord]) -> list[MemoryRecord]:
    seen_slots: set[str] = set()
    seen_texts: set[str] = set()
    deduped: list[MemoryRecord] = []

    for memory in sorted(
        memories,
        key=lambda m: m.updated_at.timestamp() if m.updated_at else 0.0,
        reverse=True,
    ):
        if memory.slot:
            if memory.slot in seen_slots:
                continue
            seen_slots.add(memory.slot)
            deduped.append(memory)
            continue

        text_key = _normalize_text(memory.text)
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        deduped.append(memory)

    return deduped


def retrieve_memories(state: FaustState, *, store) -> dict:
    """Load durable memories for the current user, preferring exact slot lookups."""
    config = state["config"]

    if not config.memory.enabled or store is None:
        return {"recalled_memories": []}

    namespace = _memory_namespace(state)
    query = state.get("user_input", "").strip()
    candidates: list[MemoryRecord] = []

    slot_keys = (
        "preference.favorite_editor",
        "profile.name",
        "profile.birthdate",
    )

    try:
        for key in slot_keys:
            item = store.get(namespace, key)
            if item:
                memory = _memory_record_from_item(item)
                if memory:
                    candidates.append(memory)

        search_limit = max(config.memory.max_results, 10)
        for item in store.search(namespace, limit=search_limit):
            memory = _memory_record_from_item(item)
            if memory:
                candidates.append(memory)

    except Exception:
        return {"recalled_memories": []}

    deduped = _dedupe_memories(candidates)

    if not query:
        return {"recalled_memories": deduped[: config.memory.max_results]}

    ranked = sorted(
        deduped,
        key=lambda memory: _score_memory(query, memory),
        reverse=True,
    )

    return {"recalled_memories": ranked[: config.memory.max_results]}


def build_prompt(state: FaustState) -> dict:
    """Ensure system prompt is first and inject recalled memories."""
    current = list(state.get("messages", []))
    config = state.get("config")
    recalled = state.get("recalled_memories", [])

    system_parts: list[str] = []
    if config:
        system_parts.append(config.system_prompt)

    if recalled:
        memory_lines = [
            f"- {memory.text}"
            for memory in recalled
            if memory.text.strip()
        ]
        if memory_lines:
            system_parts.append(
                "Relevant long-term memory about the user:\n"
                + "\n".join(memory_lines)
            )

    if system_parts:
        system_text = "\n\n".join(system_parts)
        system_msg = Message(role=Role.SYSTEM, content=system_text)

        if current and current[0].role == Role.SYSTEM:
            current[0] = system_msg
        else:
            current = [system_msg] + current

    return {"messages": current}


def _is_memory_write(query: str) -> bool:
    normalized_query = _normalize_text(query)
    return normalized_query.startswith((
        "remember that ",
        "remember this ",
        "remember ",
    ))


def _extract_recall_answer(query: str, recalled: list[MemoryRecord]) -> str | None:
    """Return a deterministic answer for simple user-fact recall."""
    normalized_query = _normalize_text(query)
    if not normalized_query or not recalled:
        return None

    best_by_slot = {
        memory.slot: memory
        for memory in recalled
        if memory.slot
    }

    favorite_editor_memory = best_by_slot.get("preference.favorite_editor")
    if favorite_editor_memory and "favorite editor" in normalized_query:
        text = favorite_editor_memory.text.strip()
        prefix = "The user's favorite editor is "
        value = text.removeprefix(prefix).rstrip(".")
        return f"Your favorite editor is {value}."

    name_memory = best_by_slot.get("profile.name")
    if name_memory and (
        "my name" in normalized_query or "who am i" in normalized_query
    ):
        text = name_memory.text.strip()
        prefix = "The user's name is "
        value = text.removeprefix(prefix).rstrip(".")
        return f"Your name is {value}."

    birthdate_memory = best_by_slot.get("profile.birthdate")
    if birthdate_memory and (
        "when was i born" in normalized_query
        or "what is my birthdate" in normalized_query
        or "what's my birthdate" in normalized_query
    ):
        text = birthdate_memory.text.strip()
        prefix = "The user's birthdate is "
        value = text.removeprefix(prefix).rstrip(".")
        return f"You were born on {value}."

    return None


def memory_answer_node(state: FaustState) -> dict:
    """Return a grounded answer directly from recalled memory when possible."""
    query = state.get("user_input", "")
    recalled = state.get("recalled_memories", [])
    answer = _extract_recall_answer(query, recalled)

    if answer:
        return {
            "response": answer,
            "error": None,
            "intent": "memory_recall",
            "active_agent": "memory_answer",
        }

    return {}


def should_use_memory_answer(state: FaustState) -> str:
    """Route memory writes and deterministic recall queries before the LLM."""
    query = state.get("user_input", "")
    recalled = state.get("recalled_memories", [])

    if _is_memory_write(query):
        return "save_memory"

    if _extract_recall_answer(query, recalled) is not None:
        return "memory_answer"

    return "build_prompt"


def llm_node(state: FaustState, adapter) -> dict:
    """Call the LLM adapter and collect the full streaming response."""
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""

    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        return {
            "response": full_response,
            "error": None,
            "active_agent": "assistant",
        }
    except Exception as exc:
        return {
            "response": "",
            "error": str(exc),
            "active_agent": "assistant",
        }


def save_memory(state: FaustState, *, store) -> dict:
    """Persist explicit durable memory requests."""
    config = state["config"]
    user_input = state.get("user_input", "").strip()

    if not config.memory.enabled or store is None or not user_input:
        return {}

    lowered = user_input.lower()
    triggers = (
        "remember that ",
        "remember this ",
        "remember ",
    )

    matched_prefix = next(
        (prefix for prefix in triggers if lowered.startswith(prefix)),
        None,
    )
    if not matched_prefix:
        return {}

    memory_text = user_input[len(matched_prefix):].strip()
    if not memory_text:
        return {}

    normalized_text, normalized_category = _normalize_memory_text(memory_text)

    namespace = _memory_namespace(state)
    now = datetime.now(timezone.utc)
    slot, extracted_value = _detect_slot(memory_text)

    if slot == "preference.favorite_editor":
        stored_text = f"The user's favorite editor is {extracted_value}."
        key = slot
        category = "preference"
    elif slot == "profile.name":
        stored_text = f"The user's name is {extracted_value}."
        key = slot
        category = "profile"
    elif slot == "profile.birthdate":
        stored_text = f"The user's birthdate is {extracted_value}."
        key = slot
        category = "profile"
    else:
        stored_text = normalized_text
        key = str(uuid4())
        category = normalized_category

    record = MemoryRecord(
        key=key,
        slot=slot,
        text=stored_text,
        category=category,
        source="explicit",
        created_at=now,
        updated_at=now,
    )

    try:
        store.put(namespace, key, record.model_dump(mode="json"))
    except Exception:
        return {}

    recalled = [
        m for m in state.get("recalled_memories", [])
        if not (slot is not None and m.slot == slot)
    ]
    recalled.append(record)
    recalled = _dedupe_memories(recalled)

    response = ""
    if slot == "preference.favorite_editor":
        response = f"Okay — I'll remember that your favorite editor is {extracted_value}."
    elif slot == "profile.name":
        response = f"Okay — I'll remember that your name is {extracted_value}."
    elif slot == "profile.birthdate":
        response = f"Okay — I'll remember that you were born on {extracted_value}."
    else:
        response = "Okay — I'll remember that."

    return {
        "recalled_memories": recalled,
        "response": response,
        "error": None,
        "intent": "memory_write",
        "active_agent": "memory_write",
    }


def make_checkpointer(config: AppConfig):
    """Create the configured LangGraph checkpointer."""
    allowed_msgpack_modules = (
        ("faust.core.models", "AppConfig"),
        ("faust.core.models", "Message"),
        ("faust.core.models", "Role"),
        ("faust.core.models", "MemoryRecord"),
        ("faust.core.models", "Session"),
    )

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=allowed_msgpack_modules
    )

    if config.checkpointer_backend == "sqlite":
        db_path = Path(config.sqlite.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(db_path), serde=serde)

    return InMemorySaver(serde=serde)


def build_graph(adapter, config: AppConfig) -> CompiledStateGraph:
    """Build and compile the LangGraph state graph."""
    workflow = StateGraph(FaustState)
    store = make_memory_store(config)

    workflow.add_node(
        "retrieve_memories",
        partial(retrieve_memories, store=store),
    )
    workflow.add_node("memory_answer", memory_answer_node)
    workflow.add_node("build_prompt", build_prompt)
    workflow.add_node("llm", partial(llm_node, adapter=adapter))
    workflow.add_node("save_memory", partial(save_memory, store=store))

    workflow.set_entry_point("retrieve_memories")
    workflow.add_conditional_edges(
        "retrieve_memories",
        should_use_memory_answer,
        {
            "save_memory": "save_memory",
            "memory_answer": "memory_answer",
            "build_prompt": "build_prompt",
        },
    )
    workflow.add_edge("memory_answer", END)
    workflow.add_edge("build_prompt", "llm")
    workflow.add_edge("llm", "save_memory")
    workflow.add_edge("save_memory", END)

    checkpointer = make_checkpointer(config)

    return workflow.compile(checkpointer=checkpointer, store=store)