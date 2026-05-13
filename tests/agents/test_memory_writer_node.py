"""Isolated unit tests for the memory_writer (save_memory) node.

Every test uses an in-memory store and asserts on the state fields written by
the node — slot key, category, active_agent — not just on the response string.
No shared state between tests — each test builds its own store and state.
"""
from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from faust.adapters.graph import save_memory
from faust.core.models import AppConfig, Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(
    user_input: str,
    user_id: str = "test-user",
):
    """Return a minimal, fully isolated FaustState payload."""
    config = AppConfig()
    session = Session(id="test-session", model=config.model)
    return {
        "session": session,
        "config": config,
        "user_id": user_id,
        "user_input": user_input,
        "intent": None,
        "memory_route": None,
        "active_agent": None,
        "messages": [],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
        "test_proposal": None,
        "test_approved": False,
        "test_report_path": None,
        "requested_tests": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_memory_writer_persists_favorite_editor_slot():
    """save_memory must write a MemoryRecord with the correct slot key for
    an explicit 'remember my favorite editor' request."""
    store = InMemoryStore()
    state = _make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert len(result["recalled_memories"]) == 1
    memory = result["recalled_memories"][0]
    assert memory.slot == "preference.favorite_editor"
    assert "Neovim" in memory.text


def test_memory_writer_sets_correct_category():
    """save_memory must tag the persisted record with the correct category."""
    store = InMemoryStore()
    state = _make_state(
        user_input="remember that my favorite editor is Vim",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    memory = result["recalled_memories"][0]
    assert memory.category == "preference"


def test_memory_writer_sets_active_agent_to_memory_write():
    """save_memory must set active_agent == 'memory_write' in the returned
    state slice so the graph can confirm which node handled the turn."""
    store = InMemoryStore()
    state = _make_state(
        user_input="remember that my favorite shell is zsh",
        user_id="javier",
    )

    result = save_memory(state, store=store)

    assert result.get("active_agent") == "memory_write"


def test_memory_writer_overwrites_existing_slot():
    """A second save_memory call for the same slot must overwrite the previous
    value — no duplicate records should exist for the same slot key."""
    store = InMemoryStore()
    config = AppConfig()

    state1 = _make_state(
        user_input="remember that my favorite editor is Neovim",
        user_id="javier",
    )
    state1["config"] = config
    save_memory(state1, store=store)

    state2 = _make_state(
        user_input="remember that my favorite editor is Emacs",
        user_id="javier",
    )
    state2["config"] = config
    result2 = save_memory(state2, store=store)

    editor_records = [
        m for m in result2["recalled_memories"]
        if m.slot == "preference.favorite_editor"
    ]
    assert len(editor_records) == 1
    assert "Emacs" in editor_records[0].text
