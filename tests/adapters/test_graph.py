"""Tests for LangGraph wiring in faust.adapters.graph."""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.store.memory import InMemoryStore

from faust.adapters.graph import (
    _detect_recall_slot,
    _has_recalled_slot,
    build_graph,
    build_prompt,
    llm_node,
    memory_answer_node,
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


def test_llm_node_concatenates_streamed_chunks():
    """llm_node should join adapter chunks into one response string."""
    adapter = FakeAdapter(chunks=["Fa", "ust"])
    state = make_state(
        messages=[
            Message(role=Role.SYSTEM, content="You are Faust."),
            Message(role=Role.USER, content="Say your name"),
        ]
    )

    result = llm_node(state, adapter=adapter)

    assert result["response"] == "Faust"
    assert result["error"] is None
    assert adapter.last_messages == [
        {"role": "system", "content": "You are Faust."},
        {"role": "user", "content": "Say your name"},
    ]


def test_llm_node_returns_error_on_adapter_failure():
    """llm_node should return an error string instead of raising."""
    adapter = FakeAdapter(should_fail=True)
    state = make_state(
        messages=[
            Message(role=Role.USER, content="Hi"),
        ]
    )

    result = llm_node(state, adapter=adapter)

    assert result["response"] == ""
    assert "fake adapter failure" in result["error"]


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


def test_llm_node_sets_active_agent():
    """llm_node should set active_agent to 'assistant' when it runs."""
    adapter = FakeAdapter(chunks=["Hi"])
    state = make_state(
        messages=[Message(role=Role.USER, content="Hello")],
    )

    result = llm_node(state, adapter=adapter)

    assert result["response"] == "Hi"
    assert result["error"] is None
    assert result.get("active_agent") == "assistant"


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
    assert _detect_recall_slot("What is my favorite editor?") == "preference.favorite_editor"
    assert _detect_recall_slot("which editor do i prefer") == "preference.favorite_editor"
    assert _detect_recall_slot("what editor do i use") == "preference.favorite_editor"
    assert _detect_recall_slot("what's my preferred editor") == "preference.favorite_editor"


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


def test_route_memory_returns_llm_when_slot_missing():
    """route_memory should fall back to llm when the slot is not present."""
    state = make_state(
        user_input="What is my favorite editor?",
        user_id="javier",
    )

    result = route_memory(state)

    assert result["memory_route"] == "llm"


def test_route_memory_returns_llm_for_normal_query():
    """route_memory should route normal queries to llm."""
    state = make_state(user_input="How are you today?")

    result = route_memory(state)

    assert result["memory_route"] == "llm"


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