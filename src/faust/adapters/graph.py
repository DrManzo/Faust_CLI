"""LangGraph state graph for Faust."""


from __future__ import annotations


import re
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------
# Report path — always faust/reports/ regardless of CWD.
# src/faust/adapters/graph.py  ->  three parents up  ->  repo root (faust/)
# ---------------------------------------------------------------------------
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_REPORTS_DIR: Path = _REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# JSON schemas for structured extraction
# ---------------------------------------------------------------------------

_CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": ["general", "coding", "reasoning", "test_draft", "test_run", "memory"],
            "description": "The primary task type for this user turn.",
        },
        "requested_role": {
            "type": ["string", "null"],
            "enum": ["assistant", "coder", "reasoner", "test_proposer", None],
            "description": "The best agent role for this turn, or null to let task_type decide.",
        },
    },
    "required": ["task_type", "requested_role"],
    "additionalProperties": False,
}

_EXTRACT_TESTS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "targets": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of pytest target paths in the form "
                "'tests/path/to/test_file.py::test_function_name'. "
                "Return an empty list if no targets are mentioned."
            ),
        },
    },
    "required": ["targets"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SlotSpec:
    slot: str
    category: str
    write_patterns: tuple[tuple[str, str | None], ...]
    recall_phrases: tuple[str, ...]
    stored_text_template: str
    response_template: str
    confirm_template: str
    score_phrases: tuple[str, ...] = ()
    priority: int = 200



SLOT_SPECS: tuple[SlotSpec, ...] = (
    SlotSpec(
        slot="preference.favorite_editor",
        category="preference",
        write_patterns=(
            (r"my favorite editor is\s+(.+)", "my favorite editor is "),
            (r"i prefer\s+(.+)", None),
        ),
        recall_phrases=(
            "favorite editor",
            "preferred editor",
            "what editor do i prefer",
            "which editor do i prefer",
            "what editor do i use",
        ),
        stored_text_template="The user's favorite editor is {value}.",
        response_template="Your favorite editor is {value}.",
        confirm_template="Okay — I'll remember that your favorite editor is {value}.",
        score_phrases=("favorite editor", "preferred editor"),
    ),
    SlotSpec(
        slot="preference.favorite_shell",
        category="preference",
        write_patterns=(
            (r"my favorite shell is\s+(.+)", "my favorite shell is "),
            (r"i prefer\s+(.+)", None),
        ),
        recall_phrases=(
            "favorite shell",
            "preferred shell",
            "what shell do i prefer",
            "which shell do i prefer",
            "what shell do i use",
        ),
        stored_text_template="The user's favorite shell is {value}.",
        response_template="Your favorite shell is {value}.",
        confirm_template="Okay — I'll remember that your favorite shell is {value}.",
        score_phrases=("favorite shell", "preferred shell"),
    ),
    SlotSpec(
        slot="profile.name",
        category="profile",
        write_patterns=(
            (r"my name is\s+(.+)", "my name is "),
            (r"i am\s+(.+)", "i am "),
        ),
        recall_phrases=(
            "who am i",
            "what is my name",
            "what's my name",
            "do you know my name",
        ),
        stored_text_template="The user's name is {value}.",
        response_template="Your name is {value}.",
        confirm_template="Okay — I'll remember that your name is {value}.",
        score_phrases=("my name", "who am i"),
    ),
    SlotSpec(
        slot="profile.birthdate",
        category="profile",
        write_patterns=(
            (r"i was born on\s+(.+)", "i was born on "),
            (r"my birthdate is\s+(.+)", "my birthdate is "),
            (r"my birthday is\s+(.+)", "my birthday is "),
        ),
        recall_phrases=(
            "when was i born",
            "what is my birthdate",
            "what's my birthdate",
            "when is my birthday",
            "what is my birthday",
            "what's my birthday",
        ),
        stored_text_template="The user's birthdate is {value}.",
        response_template="Your birthdate is {value}.",
        confirm_template="Okay — I'll remember that your birthdate is {value}.",
        score_phrases=("birthdate", "birthday", "born"),
        priority=220,
    ),
    SlotSpec(
        slot="profile.location",
        category="profile",
        write_patterns=(
            (r"i live in\s+(.+)", "i live in "),
            (r"my location is\s+(.+)", "my location is "),
            (r"i am from\s+(.+)", "i am from "),
        ),
        recall_phrases=(
            "where do i live",
            "what is my location",
            "what's my location",
            "where am i from",
            "where am i located",
        ),
        stored_text_template="The user's location is {value}.",
        response_template="Your location is {value}.",
        confirm_template="Okay — I'll remember that your location is {value}.",
        score_phrases=("location", "live", "from"),
    ),
)



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



def _slot_specs() -> tuple[SlotSpec, ...]:
    return SLOT_SPECS



def _find_slot_spec(slot: str) -> SlotSpec | None:
    for spec in _slot_specs():
        if spec.slot == slot:
            return spec
    return None



def _extract_slot_value(spec: SlotSpec, original: str, normalized: str) -> str | None:
    cleaned_original = re.sub(
        r"^(actually|no[, ]+|nope[, ]+|it's|it is)[,\s]+",
        "",
        original,
        flags=re.IGNORECASE,
    ).strip()
    cleaned_normalized = _normalize_text(cleaned_original)


    for pattern, required_prefix in spec.write_patterns:
        match = re.match(pattern, cleaned_original, flags=re.IGNORECASE)
        if not match:
            continue
        if required_prefix and not cleaned_normalized.startswith(required_prefix):
            continue


        candidate = _strip_correction_prefixes(match.group(1))


        if spec.slot == "preference.favorite_editor" and "editor" not in cleaned_normalized:
            if pattern.startswith(r"i prefer"):
                continue


        if spec.slot == "preference.favorite_shell" and "shell" not in cleaned_normalized:
            if pattern.startswith(r"i prefer"):
                continue


        return candidate
    return None



def _detect_slot(memory_text: str) -> tuple[str | None, str]:
    normalized = _normalize_text(memory_text)
    original = memory_text.strip().rstrip(".")


    for spec in _slot_specs():
        value = _extract_slot_value(spec, original, normalized)
        if value:
            return spec.slot, value


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


    spec = _find_slot_spec(memory.slot) if memory.slot else None
    if spec and any(phrase in normalized_query for phrase in spec.recall_phrases):
        score += spec.priority


    if spec:
        for phrase in spec.score_phrases:
            if phrase in normalized_query and phrase in normalized_text:
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



def _filter_relevant_memories(
    query: str,
    memories: list[MemoryRecord],
    *,
    max_results: int,
) -> list[MemoryRecord]:
    """Return only the most relevant memories for the current turn."""
    if not memories:
        return []


    if not query.strip():
        return memories[:max_results]


    scored = [
        (_score_memory(query, memory), memory)
        for memory in memories
    ]
    scored.sort(key=lambda item: item[0], reverse=True)


    relevant: list[MemoryRecord] = []
    for (score, _timestamp), memory in scored:
        if score <= 0:
            continue
        relevant.append(memory)
        if len(relevant) >= max_results:
            break


    return relevant



def retrieve_memories(state: FaustState, *, store) -> dict:
    """Load durable memories for the current user and attach relevant hits."""
    config = state["config"]


    if not config.memory.enabled or store is None:
        return {
            "memory_query": state.get("user_input", "").strip(),
            "memory_hits": [],
            "recalled_memories": [],
        }


    namespace = _memory_namespace(state)
    query = state.get("user_input", "").strip()
    candidates: list[MemoryRecord] = []


    slot_keys = tuple(spec.slot for spec in _slot_specs())


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
        return {
            "memory_query": query,
            "memory_hits": [],
            "recalled_memories": [],
        }


    deduped = _dedupe_memories(candidates)
    relevant = _filter_relevant_memories(
        query,
        deduped,
        max_results=config.memory.max_results,
    )


    return {
        "memory_query": query,
        "memory_hits": relevant,
        "recalled_memories": relevant,
    }



# ---------------------------------------------------------------------------
# Structured classification helpers
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """You are a task classifier. Given a user message, return ONLY a JSON object.

Choose task_type from:
- "general"     : casual chat, questions, explanations
- "coding"      : implement code, refactor, debug, fix a bug
- "reasoning"   : plan, design, architecture, break down a problem
- "test_draft"  : draft / propose / write / suggest a new test
- "test_run"    : run / execute an existing pytest test
- "memory"      : save or recall a personal fact

Choose requested_role from:
- "assistant"      : general help
- "coder"          : code implementation or test execution
- "reasoner"       : planning and design
- "test_proposer"  : test drafting
- null             : let task_type decide
"""

_EXTRACT_TESTS_SYSTEM_PROMPT = """You are a pytest target extractor.
Given a user message, extract all pytest targets mentioned.
Targets look like: tests/path/to/test_file.py or tests/path/to/test_file.py::test_function_name
Return ONLY a JSON object with a \"targets\" array. Return an empty array if none are mentioned.
"""

# ---------------------------------------------------------------------------
# Approval detection
# ---------------------------------------------------------------------------

_APPROVAL_PHRASES: tuple[str, ...] = (
    "approved",
    "yes run",
    "go ahead",
    "run the tests",
    "run tests",
    "execute the tests",
    "execute tests",
    "run scoped",
    "execute scoped",
    "confirmed",
    "proceed",
    "yes proceed",
    "do it",
    "test is approved",
    "tests are approved",
    "this is approved",
    "faust this test is approved",
    "faust run",
)


def _is_approval(query: str) -> bool:
    """Return True if the user message is an explicit test-run approval."""
    q = _normalize_text(query)
    return any(phrase in q for phrase in _APPROVAL_PHRASES)


def _classify_with_adapter(query: str, adapter) -> dict | None:
    """Call generate_structured() for task classification. Returns None on failure."""
    if adapter is None or not hasattr(adapter, "generate_structured"):
        return None
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        result = adapter.generate_structured(messages, _CLASSIFY_SCHEMA)
        if result and "task_type" in result:
            return result
    except Exception:
        pass
    return None


def _extract_tests_with_adapter(query: str, adapter) -> list[str] | None:
    """Call generate_structured() to extract pytest targets. Returns None on failure."""
    if adapter is None or not hasattr(adapter, "generate_structured"):
        return None
    messages = [
        {"role": "system", "content": _EXTRACT_TESTS_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        result = adapter.generate_structured(messages, _EXTRACT_TESTS_SCHEMA)
        if result and isinstance(result.get("targets"), list):
            return [t for t in result["targets"] if isinstance(t, str) and t.strip()]
    except Exception:
        pass
    return None



def classify_task(state: FaustState, adapter=None) -> dict:
    """Classify the turn into a narrow task type before role routing.

    Uses generate_structured() with a JSON schema for deterministic extraction
    when an adapter is available. Falls back to regex markers when the adapter
    is absent (unit tests / CI) or returns an empty result.

    Approval detection runs FIRST: if the user sends an approval phrase and
    requested_tests are already in state, we open the gate (test_approved=True)
    and route to test_run without calling the LLM classifier.
    """
    query = _normalize_text(state.get("user_input", ""))


    if not query:
        return {"task_type": "general"}

    # --- Approval gate: detect explicit approval phrases FIRST ---
    # If the user is approving a pending test proposal, open the gate immediately.
    # This must run before the LLM classifier so the coder node sees test_approved=True.
    if _is_approval(query) and state.get("requested_tests"):
        return {
            "task_type": "test_run",
            "requested_role": "coder",
            "test_approved": True,
        }

    # If already armed from a previous turn, preserve the test_run route.
    if state.get("test_approved") and state.get("requested_tests"):
        return {"task_type": "test_run"}

    # --- Structured classification (primary path) ---
    structured = _classify_with_adapter(query, adapter)
    if structured:
        task_type = structured.get("task_type", "general")
        requested_role = structured.get("requested_role")  # may be None
        result = {"task_type": task_type}
        if requested_role:
            result["requested_role"] = requested_role
        return result

    # --- Regex fallback (unit tests / adapter unavailable) ---
    test_run_markers = (
        "run scoped pytest",
        "execute scoped pytest",
        "run pytest",
        "execute pytest",
    )
    if any(marker in query for marker in test_run_markers):
        return {"task_type": "test_run"}

    test_draft_markers = (
        "draft a test",
        "draft test",
        "propose a test",
        "propose a scoped",
        "write a test for",
        "write tests for",
        "suggest a test",
        "generate a test",
        "test coverage",
        "test plan",
    )
    if any(marker in query for marker in test_draft_markers):
        return {"task_type": "test_draft"}


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


    if _is_memory_write(query) or _detect_recall_slot(query):
        return {"task_type": "memory"}


    return {"task_type": "general"}



def route_memory(state: FaustState) -> dict:
    """Decide whether this turn is a memory write, memory recall, or normal flow."""
    query = state.get("user_input", "")
    recalled = state.get("recalled_memories", [])

    # Never divert an armed approval turn into the memory path.
    if state.get("test_approved") and state.get("requested_tests"):
        return {
            "memory_route": "role_router",
            "execution_notes": "Approval gate armed — bypassing memory route check.",
        }

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


    if requested_role in {"assistant", "reasoner", "coder", "test_proposer"}:
        role = requested_role
    elif task_type == "test_draft":
        role = "test_proposer"
    elif task_type in {"coding", "test_run"}:
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
    if role == "test_proposer":
        return "test_proposer"
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

    # Inject real model names so the assistant can answer 'what models are active?'
    if config and hasattr(config, "models"):
        models = config.models
        system_parts.append(
            f"Active model routing:\n"
            f"  assistant (default): {models.default}\n"
            f"  coder: {models.coder}\n"
            f"  reasoner/planner: {models.planner}"
        )

    if requested_role:
        system_parts.append(
            f"Active role: {requested_role}. Task type: {task_type or 'general'}."
        )


    if requested_role == "reasoner":
        system_parts.append(
            "Focus on planning, decomposition, tradeoffs, and clear execution steps."
        )
    elif requested_role == "coder":
        if task_type == "test_run":
            requested_tests = state.get("requested_tests") or []
            targets_str = (
                "\n".join(f"  - {t}" for t in requested_tests)
                if requested_tests
                else "  (none extracted — check your target format)"
            )
            system_parts.append(
                "You are in test-run mode (Step 9 approval gate).\n"
                "Scoped pytest targets extracted from this request:\n"
                f"{targets_str}\n\n"
                "Rules you must follow without exception:\n"
                "1. Do NOT run any tests yet. Human approval is required first.\n"
                "2. Do NOT invent alternate commands, flags, or output paths.\n"
                "3. Do NOT modify or suggest changes to production code.\n"
                "4. Acknowledge the exact targets listed above.\n"
                "5. State clearly that approval is required before execution.\n"
                "6. Ask the human to reply with explicit approval to proceed.\n"
                "7. If no valid targets were extracted, say so and stop."
            )
        else:
            system_parts.append(
                "You are the coder role. You MUST ground all code and test suggestions "
                "in the actual repository files provided in context. "
                "Do NOT invent classes, imports, or test helpers that are not already "
                "present in the repo. When drafting a test, quote the existing test "
                "function names and fixtures from the target file before proposing "
                "anything new. If you cannot verify a class or function exists in the "
                "repo, say so explicitly and stop. "
                "Focus on implementation details, code changes, and relevant tests."
            )
    elif requested_role == "test_proposer":
        system_parts.append(
            "You are in test-draft mode. Propose a scoped pytest test file or test "
            "function for the described behavior. Output only the proposed test code "
            "and a brief explanation. Do NOT implement production code. Do NOT run "
            "any tests. The human will review and approve before anything executes. "
            "IMPORTANT: Do NOT invent classes or imports that do not exist in the repo. "
            "Use only the testing patterns and helpers that are already present in the "
            "test file you are targeting. If unsure, state what you cannot verify."
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


    for spec in _slot_specs():
        if any(phrase in q for phrase in spec.recall_phrases):
            return spec.slot


    return None



def _has_recalled_slot(recalled: list[MemoryRecord], slot: str) -> bool:
    return any(memory.slot == slot for memory in recalled)



def _extract_recall_answer(query: str, recalled: list[MemoryRecord]) -> str | None:
    """Return a deterministic answer for simple user-fact recall."""
    slot = _detect_recall_slot(query)
    if not slot or not recalled:
        return None


    best_by_slot = {memory.slot: memory for memory in recalled if memory.slot}
    memory = best_by_slot.get(slot)
    if memory is None:
        return None


    spec = _find_slot_spec(slot)
    if spec is None:
        return None


    prefix = spec.stored_text_template.format(value="").rstrip(".")
    text = memory.text.strip()
    value = text.removeprefix(prefix).strip().rstrip(".")
    if value.startswith("is "):
        value = value[3:].strip()


    return spec.response_template.format(value=value)



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



def _extract_requested_tests(query: str, adapter=None) -> list[str]:
    """Extract narrow pytest targets from plain-text requests.

    Uses generate_structured() with a JSON schema when an adapter is available.
    Falls back to regex extraction when adapter is absent or returns empty.
    """
    # --- Structured extraction (primary path) ---
    structured = _extract_tests_with_adapter(query, adapter)
    if structured is not None:
        return structured

    # --- Regex fallback ---
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



def _write_test_report(
    *,
    targets: list[str],
    rejected: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    """Write a human-readable markdown report to faust/reports/ and return the path."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)


    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = _REPORTS_DIR / f"test-report-{timestamp}.md"


    status = "PASSED" if returncode == 0 else "FAILED"
    lines = [
        f"# Faust Test Report — {timestamp}",
        f"",
        f"**Status:** {status}  ",
        f"**Exit code:** {returncode}  ",
        f"**Targets:** {', '.join(targets)}  ",
    ]
    if rejected:
        lines.append(f"**Rejected targets (outside tests/):** {', '.join(rejected)}  ")
    if stdout:
        lines += ["", "## pytest stdout", "", f"```\n{stdout}\n```"]
    if stderr:
        lines += ["", "## pytest stderr", "", f"```\n{stderr}\n```"]
    lines += ["", "---", "*Generated by Faust Step 9 tests-only workflow.*"]


    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)



def test_proposal_node(state: FaustState, adapter) -> dict:
    """Generate a proposed scoped test. Does NOT run it. Requires human approval.

    Extracts pytest targets from the user input and writes them into
    requested_tests so they survive into the next (approval) turn.
    Does NOT reset test_approved — that field stays False until the user
    explicitly approves via classify_task's approval detection.
    """
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""

    # Extract targets NOW so they are preserved in state for the approval turn.
    targets = _extract_requested_tests(state.get("user_input", ""), adapter=adapter)
    # Merge with any targets already in state (e.g. from a prior draft turn).
    existing = state.get("requested_tests") or []
    merged = existing + [t for t in targets if t not in existing]

    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        return {
            "response": full_response,
            "test_proposal": full_response,
            "test_approved": False,
            "error": None,
            "intent": "test_draft",
            "active_agent": "test_proposer",
            "requested_tests": merged,
            "test_report_path": None,
            "execution_notes": (
                "Test proposal generated. "
                "Review the proposed test above. "
                "Approve explicitly before any test run is triggered."
            ),
        }
    except Exception as exc:
        return {
            "response": "",
            "test_proposal": None,
            "test_approved": False,
            "error": str(exc),
            "intent": "test_draft",
            "active_agent": "test_proposer",
            "requested_tests": merged,
            "test_report_path": None,
            "execution_notes": "Test proposal generation failed.",
        }



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
    """Planning and decomposition role — uses deepseek-r1:8b."""
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
    """Code-focused implementation role — uses qwen2.5-coder:14b."""
    message_dicts = [m.to_dict() for m in state["messages"]]
    full_response = ""


    try:
        for chunk in adapter.generate(message_dicts, stream=True):
            full_response += chunk
        requested_tests = _extract_requested_tests(state.get("user_input", ""), adapter=adapter)
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
    """Gate: only run tests when test_approved=True AND scoped targets exist."""
    if not state.get("test_approved", False):
        return "save_memory"

    requested_tests = state.get("requested_tests", [])
    if requested_tests:
        return "run_requested_tests"

    return "save_memory"



def run_requested_tests(state: FaustState) -> dict:
    """Run scoped pytest targets. Safety rules enforced; full output to report file."""
    requested_tests = state.get("requested_tests", [])
    if not requested_tests:
        return {
            "execution_notes": "No scoped tests requested.",
            "requested_tests": requested_tests,
            "test_report_path": None,
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
            "requested_tests": requested_tests,
            "test_report_path": None,
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


        report_path = _write_test_report(
            targets=valid_targets,
            rejected=rejected_targets,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )


        status_word = "passed" if completed.returncode == 0 else "FAILED"
        note_parts = [
            f"Scoped pytest run {status_word}: {', '.join(valid_targets)}.",
            f"Exit code: {completed.returncode}.",
            f"Full output written to: {report_path}",
        ]
        if rejected_targets:
            note_parts.append("Rejected targets (outside tests/): " + ", ".join(rejected_targets) + ".")


        return {
            "execution_notes": "\n".join(note_parts),
            "requested_tests": requested_tests,
            "test_report_path": report_path,
        }


    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")


        report_path = _write_test_report(
            targets=valid_targets,
            rejected=rejected_targets,
            returncode=-1,
            stdout=stdout.strip(),
            stderr=stderr.strip() + "\n[TIMEOUT after 60 seconds]",
        )


        note_parts = [
            f"Scoped pytest run timed out after 60 seconds: {', '.join(valid_targets)}.",
            f"Full output written to: {report_path}",
        ]
        if rejected_targets:
            note_parts.append("Rejected targets: " + ", ".join(rejected_targets) + ".")


        return {
            "execution_notes": "\n".join(note_parts),
            "requested_tests": requested_tests,
            "test_report_path": report_path,
            "error": "Scoped pytest execution timed out.",
        }


    except Exception as exc:
        note_parts = [
            f"Scoped pytest execution failed unexpectedly: {', '.join(valid_targets)}.",
            f"Error: {exc}",
        ]
        if rejected_targets:
            note_parts.append("Rejected targets: " + ", ".join(rejected_targets) + ".")


        return {
            "execution_notes": "\n".join(note_parts),
            "requested_tests": requested_tests,
            "test_report_path": None,
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
    spec = _find_slot_spec(slot) if slot else None


    if spec is not None:
        stored_text = spec.stored_text_template.format(value=extracted_value)
        key = slot
        category = spec.category
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


    if spec is not None:
        response = spec.confirm_template.format(value=extracted_value)
    else:
        response = "Okay — I'll remember that."


    return {
        "recalled_memories": recalled,
        "memory_hits": recalled,
        "memory_query": user_input,
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
        allowed_msgpack_modules=allowed_msgpack_modules,
    )


    if config.checkpointer_backend == "sqlite":
        db_path = Path(config.sqlite.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(db_path), serde=serde)


    return InMemorySaver(serde=serde)



def build_graph(config: AppConfig, adapter=None) -> CompiledStateGraph:
    """Build and compile the LangGraph state graph with per-role model routing.

    Three OllamaAdapter instances are created from the models routing config:
      - adapter_default  -> config.models.default  (llama3:8b)         -- assistant, memory
      - adapter_coder    -> config.models.coder     (qwen2.5-coder:14b) -- coder, test_proposer
      - adapter_reasoner -> config.models.planner   (deepseek-r1:8b)   -- reasoner

    classify_task and _extract_requested_tests receive adapter_default so they
    can use generate_structured() for schema-constrained JSON extraction. The
    regex fallback is preserved for unit tests and CI where no live model is present.
    """
    from faust.adapters.ollama import OllamaAdapter
    from faust.adapters.openai_compat import OpenAICompatAdapter

    # Backward-compat: old call was build_graph(adapter, config).
    if not isinstance(config, AppConfig):
        config, adapter = adapter, config  # type: ignore[assignment]

    def _make_adapter(model_name: str):
        if config.backend == "openai_compat":
            return OpenAICompatAdapter(config)
        return OllamaAdapter(config, model_override=model_name)

    adapter_default  = _make_adapter(config.models.default)
    adapter_coder    = _make_adapter(config.models.coder)
    adapter_reasoner = _make_adapter(config.models.planner)

    workflow = StateGraph(FaustState)
    store = make_memory_store(config)

    workflow.add_node("retrieve_memories", partial(retrieve_memories, store=store))
    workflow.add_node("classify_task", partial(classify_task, adapter=adapter_default))
    workflow.add_node("route_memory", route_memory)
    workflow.add_node("memory_answer", memory_answer_node)
    workflow.add_node("role_router", determine_role)
    workflow.add_node("build_prompt", build_prompt)
    workflow.add_node("assistant",      partial(assistant_node,      adapter=adapter_default))
    workflow.add_node("reasoner",       partial(reasoner_node,       adapter=adapter_reasoner))
    workflow.add_node("coder",          partial(coder_node,          adapter=adapter_coder))
    workflow.add_node("test_proposer",  partial(test_proposal_node,  adapter=adapter_coder))
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
            "test_proposer": "test_proposer",
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
    workflow.add_edge("test_proposer", END)
    workflow.add_edge("save_memory", END)

    checkpointer = make_checkpointer(config)
    return workflow.compile(checkpointer=checkpointer, store=store)
