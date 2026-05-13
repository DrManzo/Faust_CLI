"""Isolated unit tests for the assistant agent node.

Every test stubs the LLM adapter (no real model call) and asserts on the
state fields written by the node, not just the response string.
No shared state between tests — each test builds its own payload.
"""
from __future__ import annotations

from faust.adapters.graph import assistant_node
from faust.core.models import AppConfig, Message, Role, Session


# ---------------------------------------------------------------------------
# Shared helpers (module-private — not pytest fixtures)
# ---------------------------------------------------------------------------

class FakeAdapter:
    """Deterministic LLM stub — yields fixed chunks or raises on demand."""

    def __init__(self, chunks=None, should_fail: bool = False):
        self.chunks = chunks or ["Hello ", "world"]
        self.should_fail = should_fail
        self.last_messages = None

    def generate(self, messages, stream: bool = True):
        self.last_messages = messages
        if self.should_fail:
            raise RuntimeError("fake adapter failure")
        yield from self.chunks


def _make_state(
    messages=None,
    user_input: str = "Hello",
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
        "messages": messages or [],
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


def test_assistant_node_concatenates_chunks():
    """assistant_node must join all streamed chunks into a single response string."""
    adapter = FakeAdapter(chunks=["Fa", "ust", "!"])
    state = _make_state(
        messages=[
            Message(role=Role.SYSTEM, content="You are Faust."),
            Message(role=Role.USER, content="Say your name"),
        ]
    )

    result = assistant_node(state, adapter=adapter)

    assert result["response"] == "Faust!"


def test_assistant_node_sets_active_agent_field():
    """assistant_node must set active_agent == 'assistant' regardless of input."""
    adapter = FakeAdapter(chunks=["ok"])
    state = _make_state(
        messages=[Message(role=Role.USER, content="Hey")]
    )

    result = assistant_node(state, adapter=adapter)

    assert result["active_agent"] == "assistant"


def test_assistant_node_populates_execution_notes():
    """assistant_node must write a non-empty execution_notes string."""
    adapter = FakeAdapter(chunks=["note"])
    state = _make_state(
        messages=[Message(role=Role.USER, content="Hi")]
    )

    result = assistant_node(state, adapter=adapter)

    assert result.get("execution_notes")
    assert "Assistant role completed" in result["execution_notes"]


def test_assistant_node_captures_adapter_failure_without_raising():
    """On adapter failure assistant_node must return an error string, not raise."""
    adapter = FakeAdapter(should_fail=True)
    state = _make_state(
        messages=[Message(role=Role.USER, content="Hi")]
    )

    result = assistant_node(state, adapter=adapter)

    assert result["response"] == ""
    assert result["error"] is not None
    assert "fake adapter failure" in result["error"]
    assert result["active_agent"] == "assistant"
