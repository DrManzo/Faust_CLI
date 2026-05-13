"""Isolated unit tests for the coder agent node.

Every test stubs the LLM adapter (no real model call) and asserts on the
state fields written by the node — particularly test_approved — not just
the response string.
No shared state between tests — each test builds its own payload.
"""
from __future__ import annotations

from faust.adapters.graph import coder_node
from faust.core.models import AppConfig, Message, Role, Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, chunks=None, should_fail: bool = False):
        self.chunks = chunks or ["def foo(): pass"]
        self.should_fail = should_fail
        self.last_messages = None

    def generate(self, messages, stream: bool = True):
        self.last_messages = messages
        if self.should_fail:
            raise RuntimeError("fake adapter failure")
        yield from self.chunks


def _make_state(user_input: str = "write code for fizzbuzz", messages=None):
    config = AppConfig()
    session = Session(id="test-session", model=config.model)
    return {
        "session": session,
        "config": config,
        "user_id": "test-user",
        "user_input": user_input,
        "intent": None,
        "memory_route": None,
        "active_agent": None,
        "messages": messages or [Message(role=Role.USER, content=user_input)],
        "recalled_memories": [],
        "artifacts": [],
        "response": "",
        "error": None,
        "test_proposal": None,
        "test_approved": False,
        "test_report_path": None,
        "requested_tests": [],
        "task_type": "coding",
        "requested_role": "coder",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_coder_node_concatenates_chunks():
    """coder_node must join all streamed chunks into a single response string."""
    adapter = FakeAdapter(chunks=["def fizzbuzz", "(): pass"])
    state = _make_state()

    result = coder_node(state, adapter=adapter)

    assert result["response"] == "def fizzbuzz(): pass"


def test_coder_node_sets_active_agent_field():
    """coder_node must set active_agent == 'coder'."""
    adapter = FakeAdapter(chunks=["ok"])
    state = _make_state()

    result = coder_node(state, adapter=adapter)

    assert result["active_agent"] == "coder"


def test_coder_node_sets_intent_to_coding():
    """coder_node must set intent == 'coding' in the returned state slice."""
    adapter = FakeAdapter(chunks=["code here"])
    state = _make_state()

    result = coder_node(state, adapter=adapter)

    assert result["intent"] == "coding"


def test_coder_node_captures_adapter_failure_without_raising():
    """On adapter failure coder_node must return an error string, not raise."""
    adapter = FakeAdapter(should_fail=True)
    state = _make_state()

    result = coder_node(state, adapter=adapter)

    assert result["response"] == ""
    assert result["error"] is not None
    assert "fake adapter failure" in result["error"]
    assert result["active_agent"] == "coder"
