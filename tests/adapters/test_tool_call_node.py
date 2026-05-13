"""Isolated unit tests for tool_call_node and its routing helpers.

Phase 3 of Step 11 — verifies that:
  1. classify_task recognises 'tool_call' intent via regex fallback.
  2. determine_role maps task_type 'tool_call' -> role 'tool_call'.
  3. route_role returns 'tool_call' for role 'tool_call'.
  4. tool_call_node dispatches read_file (no approval needed) correctly.
  5. tool_call_node blocks run_pytest (approval required) when gate is closed.
  6. tool_call_node passes run_pytest through when gate is open.
  7. tool_call_node handles missing tool_name gracefully.
  8. tool_call_node captures unexpected exceptions into state['error'].

All tests use a real (but controlled) PluginRegistry — no subprocess calls
are made; run_pytest is stubbed at the registry dispatch layer via monkeypatch.
"""

from __future__ import annotations

import pytest

from faust.adapters.graph import (
    classify_task,
    determine_role,
    route_role,
    tool_call_node,
)
from faust.core.models import AppConfig, FaustState, Message, Role, Session


# ---------------------------------------------------------------------------
# Minimal state factory
# ---------------------------------------------------------------------------

def _make_state(
    user_input: str = "",
    task_type: str | None = None,
    requested_role: str | None = None,
    tool_name: str | None = None,
    tool_inputs: dict | None = None,
    test_approved: bool = False,
) -> FaustState:
    config = AppConfig()
    session = Session(id="test-session", model="llama3.2")
    state: FaustState = {
        "session": session,
        "config": config,
        "user_id": "test-user",
        "user_input": user_input,
        "intent": None,
        "active_agent": None,
        "response": "",
        "error": None,
        "messages": [
            Message(role=Role.USER, content=user_input)
        ],
        "recalled_memories": [],
        "artifacts": [],
        "memory_route": None,
        "requested_tests": [],
        "test_approved": test_approved,
        "tool_result": None,
    }
    if task_type is not None:
        state["task_type"] = task_type
    if requested_role is not None:
        state["requested_role"] = requested_role
    if tool_name is not None:
        state["tool_name"] = tool_name
    if tool_inputs is not None:
        state["tool_inputs"] = tool_inputs
    return state


# ---------------------------------------------------------------------------
# classify_task — regex fallback recognises tool_call intent
# ---------------------------------------------------------------------------

class TestClassifyTaskToolCall:
    """classify_task must return task_type='tool_call' for file-read intents."""

    def test_read_file_marker(self):
        state = _make_state(user_input="read file src/faust/cli/constants.py")
        result = classify_task(state)  # no adapter -> regex fallback
        assert result["task_type"] == "tool_call"

    def test_read_the_file_marker(self):
        state = _make_state(user_input="read the file src/faust/cli/constants.py")
        result = classify_task(state)
        assert result["task_type"] == "tool_call"

    def test_call_tool_marker(self):
        state = _make_state(user_input="call tool read_file with path=src/faust/cli/constants.py")
        result = classify_task(state)
        assert result["task_type"] == "tool_call"

    def test_use_tool_marker(self):
        state = _make_state(user_input="use tool read_file")
        result = classify_task(state)
        assert result["task_type"] == "tool_call"


# ---------------------------------------------------------------------------
# determine_role — maps tool_call task_type to tool_call role
# ---------------------------------------------------------------------------

class TestDetermineRoleToolCall:
    """determine_role must select role='tool_call' for task_type='tool_call'."""

    def test_task_type_maps_to_role(self):
        state = _make_state(task_type="tool_call")
        result = determine_role(state)
        assert result["requested_role"] == "tool_call"

    def test_explicit_requested_role_preserved(self):
        """If requested_role is already 'tool_call', determine_role must not override it."""
        state = _make_state(task_type="general", requested_role="tool_call")
        result = determine_role(state)
        assert result["requested_role"] == "tool_call"


# ---------------------------------------------------------------------------
# route_role — returns 'tool_call' for role 'tool_call'
# ---------------------------------------------------------------------------

class TestRouteRoleToolCall:
    """route_role must return the string 'tool_call' so build_graph can route it."""

    def test_routes_to_tool_call(self):
        state = _make_state(requested_role="tool_call")
        assert route_role(state) == "tool_call"

    def test_other_roles_unaffected(self):
        """Adding tool_call routing must not disturb existing role routing."""
        assert route_role(_make_state(requested_role="assistant")) == "assistant"
        assert route_role(_make_state(requested_role="coder"))     == "coder"
        assert route_role(_make_state(requested_role="reasoner"))  == "reasoner"


# ---------------------------------------------------------------------------
# tool_call_node — dispatch behaviour
# ---------------------------------------------------------------------------

class TestToolCallNodeDispatch:
    """tool_call_node must dispatch correctly, respect approval gates, and
    capture exceptions without crashing the graph."""

    def test_no_tool_name_returns_neutral_message(self):
        """State with no tool_name must produce a safe neutral response."""
        state = _make_state()
        result = tool_call_node(state)
        assert result["active_agent"] == "tool_call"
        assert result["intent"] == "tool_call"
        assert result["error"] is None
        assert "no tool" in result["response"].lower()
        assert result["tool_result"] is None

    def test_read_file_dispatches_without_approval(self, tmp_path):
        """read_file does not require approval — must succeed gate-closed."""
        target = tmp_path / "sample.py"
        target.write_text("# hello\ndef foo(): pass\n")

        state = _make_state(
            tool_name="read_file",
            tool_inputs={"path": str(target)},
            test_approved=False,
        )
        result = tool_call_node(state)

        assert result["error"] is None
        assert result["active_agent"] == "tool_call"
        assert result["tool_result"] is not None
        # response must contain some content from the file
        assert "foo" in result["response"] or "hello" in result["response"]

    def test_run_pytest_blocked_without_approval(self, monkeypatch):
        """run_pytest requires approval — must return PermissionError message when
        gate is closed, NOT raise, NOT execute any subprocess."""
        import subprocess as _sp
        monkeypatch.setattr(_sp, "run", lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("subprocess.run must not be called without approval")
        ))

        state = _make_state(
            tool_name="run_pytest",
            tool_inputs={"targets": ["tests/core/test_config.py"]},
            test_approved=False,
        )
        result = tool_call_node(state)

        assert result["error"] is None
        assert result["active_agent"] == "tool_call"
        assert result["tool_result"] is None
        assert "approval" in result["response"].lower()
        # Gate must NOT have been opened by this call
        assert result.get("test_approved") is not True

    def test_run_pytest_passes_with_approval(self, monkeypatch):
        """run_pytest must dispatch when test_approved=True."""
        from faust.core import plugins as _plugins

        # Stub the actual pytest subprocess inside the registry tool.
        def _fake_dispatch(tool_name, inputs, *, approval_override=False):
            if tool_name == "run_pytest":
                if not approval_override:
                    raise PermissionError("run_pytest requires approval")
                return "1 passed in 0.01s"
            raise KeyError(f"Unknown tool: {tool_name}")

        registry = _plugins.get_registry()
        monkeypatch.setattr(registry, "dispatch", _fake_dispatch)

        state = _make_state(
            tool_name="run_pytest",
            tool_inputs={"targets": ["tests/core/test_config.py"]},
            test_approved=True,
        )
        result = tool_call_node(state)

        assert result["error"] is None
        assert result["active_agent"] == "tool_call"
        assert "passed" in result["response"]
        # Gate must be reset after successful dispatch
        assert result["test_approved"] is False

    def test_unexpected_exception_captured_in_error(self, monkeypatch):
        """Any non-PermissionError from dispatch must land in state['error'],
        not propagate as an unhandled exception."""
        from faust.core import plugins as _plugins

        def _boom(tool_name, inputs, *, approval_override=False):
            raise RuntimeError("disk full")

        registry = _plugins.get_registry()
        monkeypatch.setattr(registry, "dispatch", _boom)

        state = _make_state(
            tool_name="read_file",
            tool_inputs={"path": "src/faust/cli/constants.py"},
        )
        result = tool_call_node(state)

        assert result["active_agent"] == "tool_call"
        assert result["tool_result"] is None
        assert result["error"] is not None
        assert "disk full" in result["error"]
