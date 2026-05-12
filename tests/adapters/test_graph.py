"""Tests for LangGraph wiring in faust.adapters.graph."""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.store.memory import InMemoryStore

from faust.adapters.graph import (
    _detect_recall_slot,
    _has_recalled_slot,
    assistant_node,
    build_graph,
    build_prompt,
    classify_task,
    coder_node,
    determine_role,
    memory_answer_node,
    reasoner_node,
    retrieve_memories,
    route_memory,
    save_memory,
)
from faust.core.models import AppConfig, MemoryRecord, Message, Role, Session


class FakeAdapter:
    """Small fake adapter that yields deterministic chunks."""

    def __init__(self, chunks=None, should_fail: bool = False):
        self.chunks = chunks or ["Hello ", "world"]
        self.should_fail = should_fail
        self.last_messages = None

    def generate(self, messages, stream: bool = True):
        self.last_messages = messages
        if self.should_fail:
            raise RuntimeError("fake adapter failure")
        for chunk in self.chunks:
            yield chunk


def make_state(
    messages=None,
    user_input: str = "Hello",
    user_id: str = "test-user",
    config: AppConfig | None = None,
):
    """Minimal valid FaustState payload for graph tests."""
    config = config or AppConfig()
    session = Session(id="test-session", model=config.model)

    return {
        "session": session,
        "config": config,
        "user_id": user_id,
        "user_input": user_input,
        "intent": None,
        "memory_route": None,
        "active_agent": None,
        "messages": messages or [],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
    }


def test_build_prompt_injects_system_message_when_missing():
    """build_prompt should prepend a system message when one is absent."""
    state = make_state(
        messages=[
            Message(role=Role.USER, content="Hi"),
        ]
    )

    result = build_prompt(state)
    messages = result["messages"]

    assert len(messages) == 2
    assert messages[0].role == Role.SYSTEM
    assert state["config"].system_prompt in messages[0].content
    assert messages[1].role == Role.USER
    assert messages[1].content == "Hi"


def test_build_prompt_includes_recalled_memories():
    """build_prompt should inject recalled memories into the system prompt."""
    now = datetime.now(timezone.utc)
    state = make_state(
        messages=[
            Message(role=Role.USER, content="What do I like?"),
        ]
    )
    state["recalled_memories"] = [
        MemoryRecord(
            key="mem-1",
            text="User prefers concise answers.",
            category="preference",
            source="explicit",
            created_at=now,
            updated_at=now,
        )
    ]

    result = build_prompt(state)
    system_message = result["messages"][0]

    assert system_message.role == Role.SYSTEM
    assert "Relevant long-term memory about the user" in system_message.content
    assert "User prefers concise answers." in system_message.content


def test_build_prompt_includes_role_guidance_when_requested_role_set():
    """build_prompt should inject role-specific guidance for reasoner and coder."""
    state = make_state(
        messages=[Message(role=Role.USER, content="Plan this feature")],
    )
    state["requested_role"] = "reasoner"
    state["task_type"] = "reasoning"

    result = build_prompt(state)
    system_message = result["messages"][0]

    assert system_message.role == Role.SYSTEM
    assert "Active role: reasoner" in system_message.content
    assert "planning, decomposition" in system_message.content


def test_assistant_node_concatenates_streamed_chunks():
    """assistant_node should join adapter chunks into one response string."""
    adapter = FakeAdapter(chunks=["Fa", "ust"])
    state = make_state(
        messages=[
            Message(role=Role.SYSTEM, content="You are Faust."),
            Message(role=Role.USER, content="Say your name"),
        ]
    )

    result = assistant_node(state, adapter=adapter)

    assert result["response"] == "Faust"
    assert result["error"] is None
    assert result["active_agent"] == "assistant"
    assert result["requested_tests"] == []
    assert "Assistant role completed" in result["execution_notes"]


def test_assistant_node_returns_error_on_adapter_failure():
    """assistant_node should return an error string instead of raising."""
    adapter = FakeAdapter(should_fail=True)
    state = make_state(
        messages=[
            Message(role=Role.USER, content="Hi"),
        ]
    )

    result = assistant_node(state, adapter=adapter)

    assert result["response"] == ""
    assert "fake adapter failure" in result["error"]
    assert result["active_agent"] == "assistant"


def test_reasoner_node_sets_active_agent_and_execution_notes():
    """reasoner_node should set active_agent to 'reasoner' and log execution."""
    adapter = FakeAdapter(chunks=["Plan: step 1, step 2"])
    state = make_state(
        messages=[Message(role=Role.USER, content="Plan this feature")],
    )

    result = reasoner_node(state, adapter=adapter)

    assert result["response"] == "Plan: step 1, step 2"
    assert result["error"] is None
    assert result["active_agent"] == "reasoner"
    assert result["intent"] == "reasoning"
    assert "Reasoner role completed" in result["execution_notes"]


def test_coder_node_sets_active_agent_and_execution_notes():
    """coder_node should set active_agent to 'coder' and log execution."""
    adapter = FakeAdapter(chunks=["def foo(): pass"])
    state = make_state(
        messages=[Message(role=Role.USER, content="Write code for this")],
    )

    result = coder_node(state, adapter=adapter)

    assert result["response"] == "def foo(): pass"
    assert result["error"] is None
    assert result["active_agent"] == "coder"
    assert result["intent"] == "coding"
    assert "Coder role completed" in result["execution_notes"]


def test_classify_task_detects_memory_task():
    """classify_task should identify memory write/recall queries as memory tasks."""
    state = make_state(user_input="remember that my favorite editor is Vim")
    result = classify_task(state)
    assert result["task_type"] == "memory"

    state = make_state(user_input="What is my favorite editor?")
    result = classify_task(state)
    assert result["task_type"] == "memory"


def test_classify_task_detects_coding_task():
    """classify_task should identify coding-related queries."""
    state = make_state(user_input="write code for a fizzbuzz function")
    result = classify_task(state)
    assert result["task_type"] == "coding"

    state = make_state(user_input="add tests for this module")
    result = classify_task(state)
    assert result["task_type"] == "coding"


def test_classify_task_detects_reasoning_task():
    """classify_task should identify planning and decomposition queries."""
    state = make_state(user_input="plan this feature step by step")
    result = classify_task(state)
    assert result["task_type"] == "reasoning"

    state = make_state(user_input="think through the architecture")
    result = classify_task(state)
    assert result["task_type"] == "reasoning"


def test_classify_task_defaults_to_general():
    """classify_task should default to general for normal queries."""
    state = make_state(user_input="Hello, how are you?")
    result = classify_task(state)
    assert result["task_type"] == "general"


def test_determine_role_selects_coder_for_coding_task():
    """determine_role should map coding tasks to coder role."""
    state = make_state(user_input="implement fizzbuzz")
    state["task_type"] = "coding"

    result = determine_role(state)

    assert result["requested_role"] == "coder"
    assert "coder" in result["execution_notes"]


def test_determine_role_selects_reasoner_for_reasoning_task():
    """determine_role should map reasoning tasks to reasoner role."""
    state = make_state(user_input="plan this feature")
    state["task_type"] = "reasoning"

    result = determine_role(state)

    assert result["requested_role"] == "reasoner"
    assert "reasoner" in result["execution_notes"]


def test_determine_role_selects_assistant_for_general_task():
    """determine_role should map general tasks to assistant role."""
    state = make_state(user_input="Hello")
    state["task_type"] = "general"

    result = determine_role(state)

    assert result["requested_role"] == "assistant"
    assert "assistant" in result["execution_notes"]


def test_determine_role_respects_explicit_requested_role():
    """determine_role should honor an explicit requested_role if already set."""
    state = make_state(user_input="Hello")
    state["task_type"] = "general"
    state["requested_role"] = "coder"

    result = determine_role(state)

    assert result["requested_role"] == "coder"


def test_save_memory_ignores_non_memory_requests():
    """save_memory should not store normal prompts."""
    store = InMemoryStore()
    state = make_state(user_input="What is Python?")

    result = save_memory(state, store=store)

    assert result == {}


def test_retrieve_memories_returns_empty_when_none_exist():
    """retrieve_memories should return an empty list when nothing is stored."""
    store = InMemoryStore()
    state = make_state(user_input="Hello")

    result = retrieve_memories(state, store=store)

    assert result["recalled_memories"] == []


def test_save_memory_returns_recalled_record():
    """Explicit memory requests should append a recalled memory record."""
    store = InMemoryStore()
    state = make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.text == "The user's favorite editor is Neovim."
    assert memory.slot == "preference.favorite_editor"
    assert memory.category == "preference"
    assert result["task_type"] == "memory"
    assert result["active_agent"] == "memory_write"
    assert "Persisted durable memory" in result["execution_notes"]


def test_retrieve_memories_returns_saved_memory_for_same_user():
    """Saved memories should be retrievable for the same user namespace."""
    store = InMemoryStore()
    namespace = ("memories", "javier")
    now = datetime.now(timezone.utc)

    store.put(
        namespace,
        "mem-1",
        {
            "key": "mem-1",
            "text": "The user's favorite editor is Neovim.",
            "slot": "preference.favorite_editor",
            "category": "preference",
            "source": "explicit",
            "created_at": now,
            "updated_at": now,
        },
    )

    state = make_state(
        user_input="What is my favorite editor?",
        user_id="javier",
    )

    result = retrieve_memories(state, store=store)

    assert len(result["recalled_memories"]) >= 1
    assert any(
        memory.text == "The user's favorite editor is Neovim."
        and memory.slot == "preference.favorite_editor"
        and memory.category == "preference"
        for memory in result["recalled_memories"]
    )


def test_build_graph_runs_end_to_end():
    """Compiled graph should inject prompt and produce a full response."""
    adapter = FakeAdapter(chunks=["Hello ", "from graph"])
    state = make_state(
        messages=[
            Message(role=Role.USER, content="Test graph"),
        ]
    )

    graph = build_graph(adapter, state["config"])

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "thread_id": "test-thread",
                "user_id": "test-user",
            }
        },
    )

    assert result["response"] == "Hello from graph"
    assert result["error"] is None
    assert result["messages"][0].role == Role.SYSTEM
    assert result["messages"][1].role == Role.USER
    assert result["active_agent"] == "assistant"


def test_sqlite_memory_persists_across_graph_instances(tmp_path):
    """SQLite-backed long-term memory should persist across graph rebuilds."""
    db_path = tmp_path / "faust.db"
    config = AppConfig()
    config.memory.backend = "sqlite"
    config.sqlite.path = str(db_path)

    remember_adapter = FakeAdapter(chunks=["Stored."])
    graph1 = build_graph(remember_adapter, config)

    remember_state = make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
        config=config,
    )

    graph1.invoke(
        remember_state,
        config={
            "configurable": {
                "thread_id": "thread-a",
                "user_id": "javier",
            }
        },
    )

    recall_adapter = FakeAdapter(chunks=["Recall."])
    graph2 = build_graph(recall_adapter, config)

    recall_state = make_state(
        messages=[Message(role=Role.USER, content="What is my favorite editor?")],
        user_input="What is my favorite editor?",
        user_id="javier",
        config=config,
    )

    result = graph2.invoke(
        recall_state,
        config={
            "configurable": {
                "thread_id": "thread-b",
                "user_id": "javier",
            }
        },
    )

    recalled = result.get("recalled_memories", [])
    assert any("favorite editor is Neovim" in memory.text for memory in recalled)

    # Step 6: Verify deterministic memory answer instead of LLM fallback
    assert result["response"] == "Your favorite editor is Neovim."
    assert result["intent"] == "memory_recall"
    assert result["active_agent"] == "memory_answer"


def test_save_memory_overwrites_slot_based_memories():
    """Saving a slot-backed memory should overwrite older records for that slot."""
    store = InMemoryStore()
    config = AppConfig()

    # First memory
    state1 = make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
        config=config,
    )
    result1 = save_memory(state1, store=store)
    assert len(result1["recalled_memories"]) == 1
    first = result1["recalled_memories"][0]
    assert first.text == "The user's favorite editor is Neovim."
    assert first.slot == "preference.favorite_editor"

    # Second memory for the same slot
    state2 = make_state(
        user_input="remember that my favorite editor is Emacs",
        user_id="javier",
        config=config,
    )
    result2 = save_memory(state2, store=store)
    recalled = result2["recalled_memories"]

    # Only one slot-backed preference should remain, with the new value
    assert len([m for m in recalled if m.slot == "preference.favorite_editor"]) == 1
    latest = [m for m in recalled if m.slot == "preference.favorite_editor"][0]
    assert latest.text == "The user's favorite editor is Emacs."


def test_memory_answer_node_sets_intent_and_active_agent():
    """memory_answer_node should set intent and active_agent when answering."""
    now = datetime.now(timezone.utc)
    recalled = [
        MemoryRecord(
            key="preference.favorite_editor",
            slot="preference.favorite_editor",
            text="The user's favorite editor is Neovim.",
            category="preference",
            source="explicit",
            created_at=now,
            updated_at=now,
        )
    ]
    state = make_state(
        user_input="What is my favorite editor?",
        user_id="javier",
    )
    state["recalled_memories"] = recalled

    result = memory_answer_node(state)

    assert result["response"] == "Your favorite editor is Neovim."
    assert result["error"] is None
    assert result.get("intent") == "memory_recall"
    assert result.get("active_agent") == "memory_answer"
    assert result.get("task_type") == "memory"
    assert "deterministic long-term memory recall" in result["execution_notes"]


def test_graph_preserves_artifacts_list():
    """Graph execution should not drop existing artifacts."""
    adapter = FakeAdapter(chunks=["Hello ", "from graph"])
    state = make_state(
        messages=[Message(role=Role.USER, content="Test graph")],
    )
    state["artifacts"] = ["initial-note"]

    graph = build_graph(adapter, state["config"])

    result = graph.invoke(
        state,
        config={
            "configurable": {
                "thread_id": "test-thread",
                "user_id": "test-user",
            }
        },
    )

    # The graph doesn't write artifacts yet, but it should not delete them
    assert "artifacts" in result
    assert "initial-note" in result["artifacts"]


# ========== Step 6: Memory Router Tests ==========


def test_detect_recall_slot_favorite_editor_variants():
    """_detect_recall_slot should recognize favorite editor phrasings."""
    assert (
        _detect_recall_slot("What is my favorite editor?")
        == "preference.favorite_editor"
    )
    assert (
        _detect_recall_slot("which editor do i prefer") == "preference.favorite_editor"
    )
    assert _detect_recall_slot("what editor do i use") == "preference.favorite_editor"
    assert (
        _detect_recall_slot("what's my preferred editor")
        == "preference.favorite_editor"
    )


def test_detect_recall_slot_name_variants():
    """_detect_recall_slot should recognize name recall phrasings."""
    assert _detect_recall_slot("Who am I?") == "profile.name"
    assert _detect_recall_slot("What is my name?") == "profile.name"
    assert _detect_recall_slot("Do you know my name?") == "profile.name"
    assert _detect_recall_slot("what's my name") == "profile.name"


def test_detect_recall_slot_birthdate_variants():
    """_detect_recall_slot should recognize birthdate recall phrasings."""
    assert _detect_recall_slot("When was I born?") == "profile.birthdate"
    assert _detect_recall_slot("What is my birthdate?") == "profile.birthdate"
    assert _detect_recall_slot("When is my birthday?") == "profile.birthdate"
    assert _detect_recall_slot("what's my birthday") == "profile.birthdate"


def test_detect_recall_slot_returns_none_for_non_recall():
    """_detect_recall_slot should return None for non-recall queries."""
    assert _detect_recall_slot("Hello!") is None
    assert _detect_recall_slot("How are you?") is None
    assert _detect_recall_slot("Tell me a joke") is None


def test_has_recalled_slot_returns_true_when_slot_exists():
    """_has_recalled_slot should detect an available recalled slot."""
    now = datetime.now(timezone.utc)
    recalled = [
        MemoryRecord(
            key="preference.favorite_editor",
            slot="preference.favorite_editor",
            text="The user's favorite editor is Neovim.",
            category="preference",
            source="explicit",
            created_at=now,
            updated_at=now,
        )
    ]

    assert _has_recalled_slot(recalled, "preference.favorite_editor") is True


def test_has_recalled_slot_returns_false_when_slot_missing():
    """_has_recalled_slot should return False when slot doesn't exist."""
    now = datetime.now(timezone.utc)
    recalled = [
        MemoryRecord(
            key="profile.name",
            slot="profile.name",
            text="The user's name is Javier.",
            category="profile",
            source="explicit",
            created_at=now,
            updated_at=now,
        )
    ]

    assert _has_recalled_slot(recalled, "preference.favorite_editor") is False


def test_route_memory_returns_memory_write_for_remember_input():
    """route_memory should classify explicit remember requests as memory writes."""
    state = make_state(user_input="remember that my favorite editor is Neovim")

    result = route_memory(state)

    assert result["memory_route"] == "memory_write"
    assert "memory write" in result["execution_notes"]


def test_route_memory_returns_memory_recall_when_slot_exists():
    """route_memory should choose deterministic recall when slot memory exists."""
    now = datetime.now(timezone.utc)
    state = make_state(
        user_input="What is my favorite editor?",
        user_id="javier",
    )
    state["recalled_memories"] = [
        MemoryRecord(
            key="preference.favorite_editor",
            slot="preference.favorite_editor",
            text="The user's favorite editor is Neovim.",
            category="preference",
            source="explicit",
            created_at=now,
            updated_at=now,
        )
    ]

    result = route_memory(state)

    assert result["memory_route"] == "memory_recall"
    assert "deterministic memory recall" in result["execution_notes"]


def test_route_memory_returns_role_router_when_slot_missing():
    """route_memory should fall back to role_router when the slot is not present."""
    state = make_state(
        user_input="What is my favorite editor?",
        user_id="javier",
    )

    result = route_memory(state)

    assert result["memory_route"] == "role_router"


def test_route_memory_returns_role_router_for_normal_query():
    """route_memory should route normal queries to role_router."""
    state = make_state(user_input="How are you today?")

    result = route_memory(state)

    assert result["memory_route"] == "role_router"


def test_graph_returns_deterministic_memory_answer_without_llm_fallback():
    """Graph should answer direct recall from memory instead of using LLM output."""
    adapter = FakeAdapter(chunks=["This should not be used"])
    config = AppConfig()
    graph = build_graph(adapter, config)

    remember_state = make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
        config=config,
    )
    graph.invoke(
        remember_state,
        config={"configurable": {"thread_id": "thread-1", "user_id": "javier"}},
    )

    recall_state = make_state(
        messages=[Message(role=Role.USER, content="What is my favorite editor?")],
        user_input="What is my favorite editor?",
        user_id="javier",
        config=config,
    )
    result = graph.invoke(
        recall_state,
        config={"configurable": {"thread_id": "thread-2", "user_id": "javier"}},
    )

    assert result["response"] == "Your favorite editor is Neovim."
    assert result["intent"] == "memory_recall"
    assert result["active_agent"] == "memory_answer"


def test_save_memory_accepts_implicit_name_statement():
    """save_memory should persist a natural self-fact name statement."""
    store = InMemoryStore()
    state = make_state(
        user_input="I am Javier",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.slot == "profile.name"
    assert memory.text == "The user's name is Javier."
    assert result["response"] == "Okay — I'll remember that your name is Javier."


def test_save_memory_accepts_implicit_birthday_statement():
    """save_memory should persist a natural birthday statement."""
    store = InMemoryStore()
    state = make_state(
        user_input="my birthday is April 4th 1994",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.slot == "profile.birthdate"
    assert memory.text == "The user's birthdate is April 4th 1994."
    assert (
        result["response"]
        == "Okay — I'll remember that your birthdate is April 4th 1994."
    )


def test_save_memory_accepts_implicit_favorite_editor_statement():
    """save_memory should persist a natural favorite editor statement."""
    store = InMemoryStore()
    state = make_state(
        user_input="my favorite editor is Vim",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.slot == "preference.favorite_editor"
    assert memory.text == "The user's favorite editor is Vim."
    assert result["response"] == "Okay — I'll remember that your favorite editor is Vim."


def test_save_memory_strips_actually_from_editor_correction():
    """save_memory should normalize 'actually' in favorite editor corrections."""
    store = InMemoryStore()
    state = make_state(
        user_input="my favorite editor is actually Vim",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.slot == "preference.favorite_editor"
    assert memory.text == "The user's favorite editor is Vim."
    assert result["response"] == "Okay — I'll remember that your favorite editor is Vim."


def test_route_memory_returns_memory_write_for_implicit_name_statement():
    """route_memory should classify 'I am ...' as a memory write."""
    state = make_state(user_input="I am Javier")

    result = route_memory(state)

    assert result["memory_route"] == "memory_write"


def test_route_memory_returns_memory_write_for_implicit_birthday_statement():
    """route_memory should classify birthday self-facts as memory writes."""
    state = make_state(user_input="my birthday is April 4th 1994")

    result = route_memory(state)

    assert result["memory_route"] == "memory_write"


def test_route_memory_returns_memory_write_for_editor_correction():
    """route_memory should classify direct editor corrections as memory writes."""
    state = make_state(user_input="my favorite editor is actually Vim")

    result = route_memory(state)

    assert result["memory_route"] == "memory_write"


def test_save_memory_overwrites_slot_for_implicit_editor_correction():
    """Implicit editor correction should overwrite the existing slot value."""
    store = InMemoryStore()
    config = AppConfig()

    first_state = make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
        config=config,
    )
    save_memory(first_state, store=store)

    second_state = make_state(
        user_input="my favorite editor is actually Vim",
        user_id="javier",
        config=config,
    )
    result = save_memory(second_state, store=store)

    recalled = result["recalled_memories"]
    editor_memories = [m for m in recalled if m.slot == "preference.favorite_editor"]

    assert len(editor_memories) == 1
    assert editor_memories[0].text == "The user's favorite editor is Vim."


def test_graph_returns_deterministic_name_answer_from_implicit_write():
    """Graph should recall a name saved from a natural self-fact statement."""
    adapter = FakeAdapter(chunks=["This should not be used"])
    config = AppConfig()
    graph = build_graph(adapter, config)

    remember_state = make_state(
        user_input="I am Javier",
        user_id="javier",
        config=config,
    )
    graph.invoke(
        remember_state,
        config={"configurable": {"thread_id": "thread-name-1", "user_id": "javier"}},
    )

    recall_state = make_state(
        messages=[Message(role=Role.USER, content="Who am I?")],
        user_input="Who am I?",
        user_id="javier",
        config=config,
    )
    result = graph.invoke(
        recall_state,
        config={"configurable": {"thread_id": "thread-name-2", "user_id": "javier"}},
    )

    assert result["response"] == "Your name is Javier."
    assert result["intent"] == "memory_recall"
    assert result["active_agent"] == "memory_answer"


def test_graph_returns_deterministic_birthdate_answer_from_implicit_write():
    """Graph should recall a birthdate saved from a natural self-fact statement."""
    adapter = FakeAdapter(chunks=["This should not be used"])
    config = AppConfig()
    graph = build_graph(adapter, config)

    remember_state = make_state(
        user_input="my birthday is April 4th 1994",
        user_id="javier",
        config=config,
    )
    graph.invoke(
        remember_state,
        config={"configurable": {"thread_id": "thread-bday-1", "user_id": "javier"}},
    )

    recall_state = make_state(
        messages=[Message(role=Role.USER, content="When is my birthday?")],
        user_input="When is my birthday?",
        user_id="javier",
        config=config,
    )
    result = graph.invoke(
        recall_state,
        config={"configurable": {"thread_id": "thread-bday-2", "user_id": "javier"}},
    )

    assert result["response"] == "Your birthdate is April 4th 1994."
    assert result["intent"] == "memory_recall"
    assert result["active_agent"] == "memory_answer"


# ========== Step 7.1: Role Routing Tests ==========


def test_graph_routes_coding_query_to_coder_node():
    """Graph should route code-related queries to the coder role."""
    adapter = FakeAdapter(chunks=["def fizzbuzz(): ..."])
    config = AppConfig()
    graph = build_graph(adapter, config)

    state = make_state(
        user_input="write code for fizzbuzz",
        config=config,
    )

    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": "coding-test", "user_id": "test-user"}},
    )

    assert result["active_agent"] == "coder"
    assert result["intent"] == "coding"
    assert "def fizzbuzz" in result["response"]


def test_graph_routes_reasoning_query_to_reasoner_node():
    """Graph should route planning queries to the reasoner role."""
    adapter = FakeAdapter(chunks=["Plan: step 1, step 2"])
    config = AppConfig()
    graph = build_graph(adapter, config)

    state = make_state(
        user_input="plan this feature step by step",
        config=config,
    )

    result = graph.invoke(
        state,
        config={
            "configurable": {"thread_id": "reasoning-test", "user_id": "test-user"}
        },
    )

    assert result["active_agent"] == "reasoner"
    assert result["intent"] == "reasoning"
    assert "Plan" in result["response"]


def test_graph_routes_general_query_to_assistant_node():
    """Graph should route general queries to the assistant role."""
    adapter = FakeAdapter(chunks=["Hello, I'm here to help."])
    config = AppConfig()
    graph = build_graph(adapter, config)

    state = make_state(
        user_input="Hello, how are you?",
        config=config,
    )

    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": "general-test", "user_id": "test-user"}},
    )

    assert result["active_agent"] == "assistant"
    assert "Hello" in result["response"]

    # ========== Step 7.2: Scoped Test Execution Tests ==========

import subprocess

from faust.adapters.graph import run_requested_tests, should_run_requested_tests


def test_should_run_requested_tests_routes_coder_with_requested_tests():
    """Coder flows with requested tests should route into run_requested_tests."""
    state = make_state(user_input="write code")
    state["active_agent"] = "coder"
    state["requested_tests"] = ["tests/adapters/test_graph.py::test_build_graph_runs_end_to_end"]

    result = should_run_requested_tests(state)

    assert result == "run_requested_tests"


def test_should_run_requested_tests_skips_when_not_coder():
    """Non-coder flows should skip scoped test execution."""
    state = make_state(user_input="plan this")
    state["active_agent"] = "reasoner"
    state["requested_tests"] = ["tests/adapters/test_graph.py::test_build_graph_runs_end_to_end"]

    result = should_run_requested_tests(state)

    assert result == "save_memory"


def test_should_run_requested_tests_skips_when_no_requested_tests():
    """Coder flows without requested tests should skip scoped test execution."""
    state = make_state(user_input="write code")
    state["active_agent"] = "coder"
    state["requested_tests"] = []

    result = should_run_requested_tests(state)

    assert result == "save_memory"


def test_run_requested_tests_returns_noop_when_empty():
    """run_requested_tests should no-op when no tests were requested."""
    state = make_state(user_input="write code")
    state["requested_tests"] = []

    result = run_requested_tests(state)

    assert result["execution_notes"] == "No scoped tests requested."


def test_run_requested_tests_rejects_invalid_targets():
    """run_requested_tests should reject non-tests paths and unsafe targets."""
    state = make_state(user_input="write code")
    state["requested_tests"] = [
        "src/faust/adapters/graph.py",
        "tests/adapters/test_graph.py; rm -rf /",
        "tests/does_not_exist.py",
    ]

    result = run_requested_tests(state)

    assert "no valid pytest targets" in result["execution_notes"].lower()
    assert "Rejected targets:" in result["execution_notes"]


def test_run_requested_tests_executes_valid_pytest_targets(monkeypatch):
    """run_requested_tests should execute valid scoped pytest targets."""
    state = make_state(user_input="write code")
    state["requested_tests"] = [
        "tests/adapters/test_graph.py::test_build_graph_runs_end_to_end"
    ]

    captured = {}

    class FakeCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = "1 passed in 0.05s"
            self.stderr = ""

    def fake_run(command, capture_output, text, timeout, check):
        captured["command"] = command
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        captured["check"] = check
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_requested_tests(state)

    assert captured["command"] == [
        "pytest",
        "tests/adapters/test_graph.py::test_build_graph_runs_end_to_end",
        "-q",
    ]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 60
    assert captured["check"] is False
    assert "Ran scoped tests:" in result["execution_notes"]
    assert "Exit code: 0." in result["execution_notes"]
    assert "Scoped pytest run passed." in result["execution_notes"]
    assert "1 passed in 0.05s" in result["execution_notes"]


def test_run_requested_tests_records_failed_pytest_run(monkeypatch):
    """run_requested_tests should record failing pytest output."""
    state = make_state(user_input="write code")
    state["requested_tests"] = [
        "tests/adapters/test_graph.py::test_build_graph_runs_end_to_end"
    ]

    class FakeCompletedProcess:
        def __init__(self):
            self.returncode = 1
            self.stdout = "1 failed in 0.04s"
            self.stderr = "AssertionError: boom"

    def fake_run(command, capture_output, text, timeout, check):
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_requested_tests(state)

    assert "Exit code: 1." in result["execution_notes"]
    assert "Scoped pytest run failed." in result["execution_notes"]
    assert "1 failed in 0.04s" in result["execution_notes"]
    assert "AssertionError: boom" in result["execution_notes"]


def test_run_requested_tests_handles_timeout(monkeypatch):
    """run_requested_tests should normalize timeout output and set error."""
    state = make_state(user_input="write code")
    state["requested_tests"] = [
        "tests/adapters/test_graph.py::test_build_graph_runs_end_to_end"
    ]

    def fake_run(command, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=60,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_requested_tests(state)

    assert result["error"] == "Scoped pytest execution timed out."
    assert "timed out after 60 seconds" in result["execution_notes"]
    assert "partial stdout" in result["execution_notes"]
    assert "partial stderr" in result["execution_notes"]


def test_run_requested_tests_handles_unexpected_exception(monkeypatch):
    """run_requested_tests should normalize unexpected subprocess errors."""
    state = make_state(user_input="write code")
    state["requested_tests"] = [
        "tests/adapters/test_graph.py::test_build_graph_runs_end_to_end"
    ]

    def fake_run(command, capture_output, text, timeout, check):
        raise RuntimeError("subprocess exploded")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_requested_tests(state)

    assert result["error"] == "subprocess exploded"
    assert "failed unexpectedly" in result["execution_notes"]
    assert "subprocess exploded" in result["execution_notes"]

    # ========== Step 7: Favorite Shell Slot Tests ==========


def test_detect_recall_slot_favorite_shell_variants():
    """_detect_recall_slot should recognize favorite shell phrasings."""
    assert (
        _detect_recall_slot("What is my favorite shell?")
        == "preference.favorite_shell"
    )
    assert (
        _detect_recall_slot("which shell do i prefer")
        == "preference.favorite_shell"
    )
    assert (
        _detect_recall_slot("what shell do i use")
        == "preference.favorite_shell"
    )
    assert (
        _detect_recall_slot("what's my preferred shell")
        == "preference.favorite_shell"
    )


def test_save_memory_accepts_implicit_favorite_shell_statement():
    """save_memory should persist a natural favorite shell statement."""
    store = InMemoryStore()
    state = make_state(
        user_input="my favorite shell is zsh",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.slot == "preference.favorite_shell"
    assert memory.text == "The user's favorite shell is zsh."
    assert result["response"] == "Okay — I'll remember that your favorite shell is zsh."


def test_route_memory_returns_memory_write_for_implicit_shell_statement():
    """route_memory should classify favorite shell self-facts as memory writes."""
    state = make_state(user_input="my favorite shell is zsh")

    result = route_memory(state)

    assert result["memory_route"] == "memory_write"


def test_memory_answer_node_returns_shell_answer():
    """memory_answer_node should answer favorite shell recall deterministically."""
    now = datetime.now(timezone.utc)
    recalled = [
        MemoryRecord(
            key="preference.favorite_shell",
            slot="preference.favorite_shell",
            text="The user's favorite shell is zsh.",
            category="preference",
            source="explicit",
            created_at=now,
            updated_at=now,
        )
    ]
    state = make_state(
        user_input="What is my favorite shell?",
        user_id="javier",
    )
    state["recalled_memories"] = recalled

    result = memory_answer_node(state)

    assert result["response"] == "Your favorite shell is zsh."
    assert result["error"] is None
    assert result["intent"] == "memory_recall"
    assert result["active_agent"] == "memory_answer"


def test_save_memory_overwrites_slot_for_implicit_shell_correction():
    """Implicit shell correction should overwrite the existing slot value."""
    store = InMemoryStore()
    config = AppConfig()

    first_state = make_state(
        user_input="remember that my favorite shell is bash",
        user_id="javier",
        config=config,
    )
    save_memory(first_state, store=store)

    second_state = make_state(
        user_input="my favorite shell is zsh",
        user_id="javier",
        config=config,
    )
    result = save_memory(second_state, store=store)

    recalled = result["recalled_memories"]
    shell_memories = [m for m in recalled if m.slot == "preference.favorite_shell"]

    assert len(shell_memories) == 1
    assert shell_memories[0].text == "The user's favorite shell is zsh."


def test_graph_returns_deterministic_shell_answer_from_implicit_write():
    """Graph should recall a shell saved from a natural self-fact statement."""
    adapter = FakeAdapter(chunks=["This should not be used"])
    config = AppConfig()
    graph = build_graph(adapter, config)

    remember_state = make_state(
        user_input="my favorite shell is zsh",
        user_id="javier",
        config=config,
    )
    graph.invoke(
        remember_state,
        config={"configurable": {"thread_id": "thread-shell-1", "user_id": "javier"}},
    )

    recall_state = make_state(
        messages=[Message(role=Role.USER, content="What is my favorite shell?")],
        user_input="What is my favorite shell?",
        user_id="javier",
        config=config,
    )
    result = graph.invoke(
        recall_state,
        config={"configurable": {"thread_id": "thread-shell-2", "user_id": "javier"}},
    )

    assert result["response"] == "Your favorite shell is zsh."
    assert result["intent"] == "memory_recall"
    assert result["active_agent"] == "memory_answer"


def test_location_correction_actually():
    """Correction with 'actually' should overwrite location slot."""
    store = InMemoryStore()
    config = AppConfig()

    state1 = make_state(
        user_input="I live in San Bernardino",
        user_id="javier",
        config=config,
    )
    result1 = save_memory(state1, store=store)
    assert "San Bernardino" in result1["response"]

    stored = store.get((config.memory.namespace, "javier"), "profile.location")
    assert stored is not None
    assert stored.value["text"] == "The user's location is San Bernardino."

    state2 = make_state(
        user_input="Actually, I live in Los Angeles",
        user_id="javier",
        config=config,
    )
    result2 = save_memory(state2, store=store)
    assert "Los Angeles" in result2["response"]

    updated = store.get((config.memory.namespace, "javier"), "profile.location")
    assert updated is not None
    assert updated.value["text"] == "The user's location is Los Angeles."

    
