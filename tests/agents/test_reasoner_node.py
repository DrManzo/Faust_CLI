"""Isolated unit tests for the reasoner agent node.

Every test stubs the LLM adapter (no real model call) and asserts on the
state fields written by the node, not just the response string.
No shared state between tests — each test builds its own payload.
"""
from __future__ import annotations

from faust.adapters.graph import reasoner_node
from faust.core.models import AppConfig, Message, Role, Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, chunks=None, should_fail: bool = False):
        self.chunks = chunks or ["Step 1", ", Step 2"]
        self.should_fail = should_fail
        self.last_messages = None

    def generate(self, messages, stream: bool = True):
        self.last_messages = messages
        if self.should_fail:
            raise RuntimeError("fake adapter failure")
        yield from self.chunks


def _make_state(user_input: str = "plan this feature", messages=None):
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
        "task_type": "reasoning",
        "requested_role": "reasoner",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reasoner_node_concatenates_chunks():
    """reasoner_node must join all streamed chunks into a single response string."""
    adapter = FakeAdapter(chunks=["Plan: ", "Step 1", ", Step 2"])
    state = _make_state()

    result = reasoner_node(state, adapter=adapter)

    assert result["response"] == "Plan: Step 1, Step 2"


def test_reasoner_node_sets_active_agent_field():
    """reasoner_node must set active_agent == 'reasoner'."""
    adapter = FakeAdapter(chunks=["ok"])
    state = _make_state()

    result = reasoner_node(state, adapter=adapter)

    assert result["active_agent"] == "reasoner"


def test_reasoner_node_sets_intent_to_reasoning():
    """reasoner_node must set intent == 'reasoning' in the returned state slice."""
    adapter = FakeAdapter(chunks=["analysis"])
    state = _make_state()

    result = reasoner_node(state, adapter=adapter)

    assert result["intent"] == "reasoning"


def test_reasoner_node_captures_adapter_failure_without_raising():
    """On adapter failure reasoner_node must return an error string, not raise."""
    adapter = FakeAdapter(should_fail=True)
    state = _make_state()

    result = reasoner_node(state, adapter=adapter)

    assert result["response"] == ""
    assert result["error"] is not None
    assert "fake adapter failure" in result["error"]
    assert result["active_agent"] == "reasoner"
