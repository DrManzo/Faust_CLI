"""LangGraph state graph for Faust."""

from __future__ import annotations

import re
import subprocess
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


def _strip_correction_prefixes(text: str) -> str:
    """Remove lightweight correction words from an extracted value."""
    value = text.strip().rstrip(".")
    value = re.sub(
        r"^(actually|no[, ]+|nope[, ]+|it's|it is)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


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

    if lowered.startswith("i am "):
        name = text[5:].strip()
        return (f"The user's name is {name}.", "profile")

    if lowered.startswith("i was born on "):
        birthdate = text[14:].strip()
        return (f"The user's birthdate is {birthdate}.", "profile")

    if lowered.startswith("my birthdate is "):
        birthdate = text[16:].strip()
        return (f"The user's birthdate is {birthdate}.", "profile")

    if lowered.startswith("my birthday is "):
        birthdate = text[15:].strip()
        return (f"The user's birthdate is {birthdate}.", "profile")

    return (f"The user said: {text}.", "fact")


def _detect_slot(memory_text: str) -> tuple[str | None, str]:
    normalized = _normalize_text(memory_text)
    original = memory_text.strip().rstrip(".")

    favorite_editor = re.match(
        r"my favorite editor is\s+(.+)", original, flags=re.IGNORECASE
    )
    if favorite_editor and normalized.startswith("my favorite editor is "):
        value = _strip_correction_prefixes(favorite_editor.group(1))
        return "preference.favorite_editor", value

    preferred_editor = re.match(
        r"i prefer\s+(.+)", original, flags=re.IGNORECASE
    )
    if preferred_editor and "editor" in normalized:
        value = _strip_correction_prefixes(preferred_editor.group(1))
        return "preference.favorite_editor", value

    user_name = re.match(
        r"my name is\s+(.+)", original, flags=re.IGNORECASE
    )
    if user_name and normalized.startswith("my name is "):
        value = _strip_correction_prefixes(user_name.group(1))
        return "profile.name", value

    i_am_name = re.match(
        r"i am\s+(.+)", original, flags=re.IGNORECASE
    )
    if i_am_name and normalized.startswith("i am "):
        value = _strip_correction_prefixes(i_am_name.group(1))
        return "profile.name", value

    born_on = re.match(
        r"i was born on\s+(.+)", original, flags=re.IGNORECASE
    )
    if born_on and normalized.startswith("i was born on "):
        value = _strip_correction_prefixes(born_on.group(1))
        return "profile.birthdate", value

    birthdate_is = re.match(
        r"my birthdate is\s+(.+)", original, flags=re.IGNORECASE
    )
    if birthdate_is and normalized.startswith("my birthdate is "):
        value = _strip_correction_prefixes(birthdate_is.group(1))
        return "profile.birthdate", value

    birthday_is = re.match(
        r"my birthday is\s+(.+)", original, flags=re.IGNORECASE
    )
    if birthday_is and normalized.startswith("my birthday is "):
        value = _strip_correction_prefixes(birthday_is.group(1))
        return "profile.birthdate", value

    return None, original


def _query_keywords(query: str) -> set[str]:
    stopwords = {
        "what",
        "is",
        "my",
        "what's",
        "do",
        "i",
        "you",
        "remember",
        "did",
        "say",
        "about",
        "the",
        "a",
        "an",
        "are",
        "to",
        "of",
        "when",
        "was",
        "who",
        "am",
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

    if (
        memory.slot == "preference.favorite_editor"
        and "favorite editor" in normalized_query
    ):
        score += 200

    if memory.slot == "profile.name" and (
        "my name" in normalized_query or "who am i" in normalized_query
    ):
        score += 200

    if memory.slot == "profile.birthdate" and (
        "when was i born" in normalized_query
        or "what is my birthdate" in normalized_query
        or "what's my birthdate" in normalized_query
        or "when is my birthday" in normalized_query
        or "what is my birthday" in normalized_query
        or "what's my birthday" in normalized_query
    ):
        score += 220

    if (
        "favorite editor" in normalized_query
        and "favorite editor" in normalized_text
    ):
        score += 80

    if "birthdate" in normalized_query and "birthdate" in normalized_text:
        score += 80

    if "birthday" in normalized_query and "birthdate" in normalized_text:
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


def classify_task(state: FaustState) -> dict:
    """Classify the turn into a narrow task type before role routing."""
    query = _normalize_text(state.get("user_input", ""))

    if not query:
        return {"task_type": "general"}

    if _is_memory_write(query) or _detect_recall_slot(query):
        return {"task_type": "memory"}

    coding_markers = (
        "write code",
        "implement",
        "code",
        "refactor",
        "debug",
        "fix this bug",
        "add tests",
        "test ",
        "pytest",
    )
    if any(marker in query for marker in coding_markers):
        return {"task_type": "coding"}

    reasoning_markers = (
        "plan this",
        "make a plan",
        "break this down",
        "think step by step",
        "reason through",
        "design this",
        "architecture",
    )
    if any(marker in query for marker in reasoning_markers):
        return {"task_type": "reasoning"}

    return {"task_type": "general"}


def route_memory(state: FaustState) -> dict:
    """Decide whether this turn is a memory write, memory recall, or normal flow."""
    query = state.get("user_input", "")
    recalled = state.get("recalled_memories", [])

    if _is_memory_write(query):
        return {
            "memory_route": "memory_write",
            "requested_role": None,
            "execution_notes": "Detected durable memory write.",
        }

    slot = _detect_recall_slot(query)
    if slot and _has_recalled_slot(recalled, slot):
        return {
            "memory_route": "memory_recall",
            "requested_role": None,
            "execution_notes": "Detected deterministic memory recall.",
        }

    return {
        "memory_route": "role_router",
        "execution_notes": "No deterministic memory path selected.",
    }


def should_route_after_memory(state: FaustState) -> str:
    """Route memory writes and deterministic recall queries before role dispatch."""
    route = state.get("memory_route")

    if route == "memory_write":
        return "save_memory"
    if route == "memory_recall":
        return "memory_answer"
    return "role_router"


def determine_role(state: FaustState) -> dict:
    """Pick a minimal model role from the normalized task type."""
    requested_role = state.get("requested_role")
    task_type = state.get("task_type")

    if requested_role in {"assistant", "reasoner", "coder"}:
        role = requested_role
    elif task_type == "coding":
        role = "coder"
    elif task_type == "reasoning":
        role = "reasoner"
    else:
        role = "assistant"

    return {
        "requested_role": role,
        "execution_notes": f"Selected role '{role}' for task type '{task_type}'.",
    }


def route_role(state: FaustState) -> str:
    """Return the graph node name for the selected role."""
    role = state.get("requested_role")

    if role == "coder":
        return "coder"
    if role == "reasoner":
        return "reasoner"
    return "assistant"


def build_prompt(state: FaustState) -> dict:
    """Ensure system prompt is first and inject recalled memories and role guidance."""
    current = list(state.get("messages", []))
    config = state.get("config")
    recalled = state.get("recalled_memories", [])
    requested_role = state.get("requested_role")
    task_type = state.get("task_type")

    system_parts: list[str] = []
    if config:
        system_parts.append(config.system_prompt)

    if requested_role:
        system_parts.append(
            f"Active role: {requested_role}. Task type: {task_type or 'general'}."
        )

    if requested_role == "reasoner":
        system_parts.append(
            "Focus on planning, decomposition, tradeoffs, and clear execution steps."
        )
    elif requested_role == "coder":
        system_parts.append(
            "Focus on implementation details, code changes, and relevant tests."
        )

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


def _extract_memory_candidate(query: str) -> tuple[str | None, str | None]:
    """Recognize explicit or implicit durable self-fact writes."""
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return None, None

    explicit_prefixes = (
        "remember that ",
        "remember this ",
        "remember ",
    )
    for prefix in explicit_prefixes:
        if normalized_query.startswith(prefix):
            raw = query[len(prefix):].strip()
            slot, value = _detect_slot(raw)
            if slot:
                return slot, value
            return None, raw

    slot, value = _detect_slot(query)
    if slot:
        return slot, value

    return None, None


def _is_memory_write(query: str) -> bool:
    slot, value = _extract_memory_candidate(query)
    return bool(slot and value)


def _detect_recall_slot(query: str) -> str | None:
    """Classify simple recall questions into a known memory slot."""
    q = _normalize_text(query)
    if not q:
        return None

    if any(
        phrase in q
        for phrase in (
            "favorite editor",
            "preferred editor",
            "what editor do i prefer",
            "which editor do i prefer",
            "what editor do i use",
        )
    ):
        return "preference.favorite_editor"

    if any(
        phrase in q
        for phrase in (
            "who am i",
            "what is my name",
            "what's my name",
            "do you know my name",
        )
    ):
        return "profile.name"

    if any(
        phrase in q
        for phrase in (
            "when was i born",
            "what is my birthdate",
            "what's my birthdate",
            "when is my birthday",
            "what is my birthday",
            "what's my birthday",
        )
    ):
        return "profile.birthdate"

    return None


def _has_recalled_slot(recalled: list[MemoryRecord], slot: str) -> bool:
    return any(memory.slot == slot for memory in recalled)


def _extract_recall_answer(query: str, recalled: list[MemoryRecord]) -> str | None:
    """Return a deterministic answer for simple user-fact recall."""
    slot = _detect_recall_slot(query)
    if not slot or not recalled:
        return None

    best_by_slot = {
        memory.slot: memory for memory in recalled if memory.slot
    }
    memory = best_by_slot.get(slot)
    if memory is None:
        return None

    text = memory.text.strip()

    if slot == "preference.favorite_editor":
        value = text.removeprefix("The user's favorite editor is ").rstrip(".")
        return f"Your favorite editor is {value}."

    if slot == "profile.name":
        value = text.removeprefix("The user's name is ").rstrip(".")
        return f"Your name is {value}."

    if slot == "profile.birthdate":
        value = text.removeprefix("The user's birthdate is ").rstrip(".")
        return f"Your birthdate is {value}."

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
            "task_type": "memory",
            "execution_notes": "Answered from deterministic long-term memory recall.",
            "requested_tests": [],
        }

    return {}


def _extract_requested_tests(query: str) -> list[str]:
    """Extract narrow pytest targets from plain-text requests."""
    matches = re.findall(
        r"(tests/[A-Za-z0-9_./-]+(?:::[A-Za-z0-9_./-]+)*)",
        query,
        flags=re.IGNORECASE,
    )
    deduped: list[str] = []
    for match in matches:
        if match not in deduped:
            deduped.append(match)
    return deduped


def assistant_node(state: FaustState, adapter) -> dict:
    """General conversation role."""
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""

    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        return {
            "response": full_response,
            "error": None,
            "intent": state.get("intent") or "general_response",
            "active_agent": "assistant",
            "requested_tests": [],
            "execution_notes": "Assistant role completed response generation.",
        }
    except Exception as exc:
        return {
            "response": "",
            "error": str(exc),
            "intent": state.get("intent") or "general_response",
            "active_agent": "assistant",
            "requested_tests": [],
            "execution_notes": "Assistant role failed during response generation.",
        }


def reasoner_node(state: FaustState, adapter) -> dict:
    """Planning and decomposition role."""
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""

    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        requested_tests = _extract_requested_tests(state.get("user_input", ""))
        return {
            "response": full_response,
            "error": None,
            "intent": "reasoning",
            "active_agent": "reasoner",
            "requested_tests": requested_tests,
            "execution_notes": "Reasoner role completed planning/decomposition.",
        }
    except Exception as exc:
        return {
            "response": "",
            "error": str(exc),
            "intent": "reasoning",
            "active_agent": "reasoner",
            "requested_tests": [],
            "execution_notes": "Reasoner role failed during planning/decomposition.",
        }


def coder_node(state: FaustState, adapter) -> dict:
    """Code-focused implementation role."""
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""

    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        requested_tests = _extract_requested_tests(state.get("user_input", ""))
        return {
            "response": full_response,
            "error": None,
            "intent": "coding",
            "active_agent": "coder",
            "requested_tests": requested_tests,
            "execution_notes": "Coder role completed implementation response.",
        }
    except Exception as exc:
        return {
            "response": "",
            "error": str(exc),
            "intent": "coding",
            "active_agent": "coder",
            "requested_tests": [],
            "execution_notes": "Coder role failed during implementation response.",
        }


def should_run_requested_tests(state: FaustState) -> str:
    """Only send coding flows with explicit scoped tests into the test node."""
    if state.get("active_agent") != "coder":
        return "save_memory"

    requested_tests = state.get("requested_tests", [])
    if requested_tests:
        return "run_requested_tests"

    return "save_memory"


def run_requested_tests(state: FaustState) -> dict:
    """Run scoped pytest targets requested by the coding workflow.

    Safety rules:
    - Only explicit pytest node IDs under tests/ are allowed.
    - No arbitrary shell commands are accepted.
    - Results are normalized back into execution_notes.
    """
    requested_tests = state.get("requested_tests", [])
    if not requested_tests:
        return {
            "execution_notes": "No scoped tests requested.",
        }

    valid_targets: list[str] = []
    rejected_targets: list[str] = []

    for target in requested_tests:
        cleaned = target.strip()
        if not cleaned:
            continue

        if not cleaned.startswith("tests/"):
            rejected_targets.append(cleaned)
            continue

        if any(token in cleaned for token in (";", "&&", "||", "|", "`", "$(", "..\\")):
            rejected_targets.append(cleaned)
            continue

        test_file = cleaned.split("::", 1)[0]
        if not Path(test_file).exists():
            rejected_targets.append(cleaned)
            continue

        valid_targets.append(cleaned)

    if not valid_targets:
        notes = ["Scoped test execution skipped: no valid pytest targets."]
        if rejected_targets:
            notes.append("Rejected targets: " + ", ".join(rejected_targets))
        return {
            "execution_notes": " ".join(notes),
        }

    command = ["pytest", *valid_targets, "-q"]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        note_parts = [
            f"Ran scoped tests: {', '.join(valid_targets)}.",
            f"Exit code: {completed.returncode}.",
        ]

        if rejected_targets:
            note_parts.append("Rejected targets: " + ", ".join(rejected_targets) + ".")

        if stdout:
            note_parts.append("pytest stdout:\n" + stdout)

        if stderr:
            note_parts.append("pytest stderr:\n" + stderr)

        if completed.returncode == 0:
            note_parts.append("Scoped pytest run passed.")
        else:
            note_parts.append("Scoped pytest run failed.")

        return {
            "execution_notes": "\n\n".join(note_parts),
        }

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

        note_parts = [
            f"Scoped pytest run timed out after 60 seconds for: {', '.join(valid_targets)}."
        ]

        if rejected_targets:
            note_parts.append("Rejected targets: " + ", ".join(rejected_targets) + ".")

        if stdout.strip():
            note_parts.append("pytest stdout before timeout:\n" + stdout.strip())

        if stderr.strip():
            note_parts.append("pytest stderr before timeout:\n" + stderr.strip())

        return {
            "execution_notes": "\n\n".join(note_parts),
            "error": "Scoped pytest execution timed out.",
        }

    except Exception as exc:
        note_parts = [
            f"Scoped pytest execution failed unexpectedly for: {', '.join(valid_targets)}.",
            f"Error: {exc}",
        ]
        if rejected_targets:
            note_parts.append("Rejected targets: " + ", ".join(rejected_targets) + ".")

        return {
            "execution_notes": "\n\n".join(note_parts),
            "error": str(exc),
        }


def save_memory(state: FaustState, *, store) -> dict:
    """Persist explicit or implicit durable memory requests."""
    config = state["config"]
    user_input = state.get("user_input", "").strip()

    if not config.memory.enabled or store is None or not user_input:
        return {}

    slot, extracted_value = _extract_memory_candidate(user_input)
    if slot is None or not extracted_value:
        return {}

    namespace = _memory_namespace(state)
    now = datetime.now(timezone.utc)

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
        normalized_text, normalized_category = _normalize_memory_text(user_input)
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
        m
        for m in state.get("recalled_memories", [])
        if not (slot is not None and m.slot == slot)
    ]
    recalled.append(record)
    recalled = _dedupe_memories(recalled)

    if slot == "preference.favorite_editor":
        response = f"Okay — I'll remember that your favorite editor is {extracted_value}."
    elif slot == "profile.name":
        response = f"Okay — I'll remember that your name is {extracted_value}."
    elif slot == "profile.birthdate":
        response = f"Okay — I'll remember that your birthdate is {extracted_value}."
    else:
        response = "Okay — I'll remember that."

    return {
        "recalled_memories": recalled,
        "response": response,
        "error": None,
        "intent": "memory_write",
        "active_agent": "memory_write",
        "task_type": "memory",
        "execution_notes": "Persisted durable memory record.",
        "requested_tests": [],
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
    workflow.add_node("classify_task", classify_task)
    workflow.add_node("route_memory", route_memory)
    workflow.add_node("memory_answer", memory_answer_node)
    workflow.add_node("role_router", determine_role)
    workflow.add_node("build_prompt", build_prompt)
    workflow.add_node("assistant", partial(assistant_node, adapter=adapter))
    workflow.add_node("reasoner", partial(reasoner_node, adapter=adapter))
    workflow.add_node("coder", partial(coder_node, adapter=adapter))
    workflow.add_node("run_requested_tests", run_requested_tests)
    workflow.add_node("save_memory", partial(save_memory, store=store))

    workflow.set_entry_point("retrieve_memories")
    workflow.add_edge("retrieve_memories", "classify_task")
    workflow.add_edge("classify_task", "route_memory")
    workflow.add_conditional_edges(
        "route_memory",
        should_route_after_memory,
        {
            "save_memory": "save_memory",
            "memory_answer": "memory_answer",
            "role_router": "role_router",
        },
    )
    workflow.add_edge("memory_answer", END)
    workflow.add_edge("role_router", "build_prompt")
    workflow.add_conditional_edges(
        "build_prompt",
        route_role,
        {
            "assistant": "assistant",
            "reasoner": "reasoner",
            "coder": "coder",
        },
    )
    workflow.add_edge("assistant", "save_memory")
    workflow.add_edge("reasoner", "save_memory")
    workflow.add_conditional_edges(
        "coder",
        should_run_requested_tests,
        {
            "run_requested_tests": "run_requested_tests",
            "save_memory": "save_memory",
        },
    )
    workflow.add_edge("run_requested_tests", "save_memory")
    workflow.add_edge("save_memory", END)

    checkpointer = make_checkpointer(config)

    return workflow.compile(checkpointer=checkpointer, store=store)