"""Isolated unit tests for the test_proposer agent node.

Every test stubs the LLM adapter (no real model call) and asserts on the
state fields written by the node — particularly test_approved and the absence
of subprocess.run calls.
No shared state between tests — each test builds its own payload.
"""
from __future__ import annotations
import subprocess

# Production function aliased so pytest does not collect it as a test.
from faust.adapters.graph import test_proposal_node as graph_proposal_node
from faust.core.models import AppConfig, Message, Role, Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, chunks=None, should_fail: bool = False):
        self.chunks = chunks or ["def test_foo():\n    assert True"]
        self.should_fail = should_fail
        self.last_messages = None

    def generate(self, messages, stream: bool = True):
        self.last_messages = messages
        if self.should_fail:
            raise RuntimeError("fake adapter failure")
        yield from self.chunks


def _make_state(user_input: str = "draft a test for foo", messages=None):
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
        "task_type": "test_draft",
        "requested_role": "test_proposer",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_proposal_node_returns_proposal_string():
    """graph_proposal_node must place the adapter output in both response and
    test_proposal state fields."""
    adapter = FakeAdapter(chunks=["def test_foo():\n    assert True"])
    state = _make_state()

    result = graph_proposal_node(state, adapter=adapter)

    assert "def test_foo" in result["response"]
    assert result["test_proposal"] == result["response"]


def test_proposal_node_test_approved_is_false():
    """graph_proposal_node must set test_approved == False — proposals are never
    auto-approved, the operator must confirm explicitly."""
    adapter = FakeAdapter(chunks=["def test_bar(): pass"])
    state = _make_state()

    result = graph_proposal_node(state, adapter=adapter)

    assert result["test_approved"] is False
    assert result["requested_tests"] == []
    assert result["test_report_path"] is None


def test_proposal_node_does_not_call_subprocess_run(monkeypatch):
    """graph_proposal_node must never invoke subprocess.run at any point."""
    called = {}

    def fake_run(*args, **kwargs):
        called["invoked"] = True

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = FakeAdapter(chunks=["def test_baz(): pass"])
    state = _make_state()

    graph_proposal_node(state, adapter=adapter)

    assert "invoked" not in called, (
        "graph_proposal_node must not call subprocess.run — proposals are drafts only"
    )


def test_proposal_node_captures_adapter_failure_without_raising():
    """On adapter failure graph_proposal_node must return error state, not raise."""
    adapter = FakeAdapter(should_fail=True)
    state = _make_state()

    result = graph_proposal_node(state, adapter=adapter)

    assert result["response"] == ""
    assert result["test_proposal"] is None
    assert result["test_approved"] is False
    assert "fake adapter failure" in result["error"]
    assert result["active_agent"] == "test_proposer"
