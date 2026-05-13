"""Targeted exit-behavior tests for Step 10 Phase 1 CLI cleanup.

Covers:
- Interactive exit / quit / /exit terminates the loop without a model turn.
- Piped stdin exhaustion exits cleanly (no 'Aborted.' on stderr).
- loop command exit tokens work.
- _is_safe_target safety gate.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from faust.cli.app import app
from faust.cli.commands.loop import _is_safe_target
from faust.core.models import AppConfig

runner = CliRunner(mix_stderr=False)


class FakeGraph:
    """Minimal graph stub — records invocations; never calls a real model."""

    def __init__(self, response: str = "stub response"):
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, state, config=None):
        self.calls.append({"state": state, "config": config})
        return {
            "response": self.response,
            "error": None,
            "messages": state.get("messages", []),
            "test_proposal": None,
            "requested_tests": [],
            "execution_notes": None,
        }


def _ctx(response: str = "stub response"):
    return {"config": AppConfig(), "graph": FakeGraph(response=response)}


# ---------------------------------------------------------------------------
# chat exit tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", ["exit", "quit", "q", "/exit"])
def test_chat_exit_tokens_terminate_without_model_turn(token):
    """Typing an exit token must not trigger a graph invocation."""
    ctx = _ctx()
    result = runner.invoke(app, ["chat"], input=f"{token}\n", obj=ctx)
    # Graph must never have been called.
    assert ctx["graph"].calls == [], (
        f"Graph was unexpectedly invoked after '{token}'"
    )
    assert result.exit_code == 0
    assert "Goodbye" in result.output or "Session ended" in result.output


def test_chat_piped_stdin_exhaustion_exits_cleanly():
    """When stdin is exhausted (empty pipe), no 'Aborted.' should leak."""
    ctx = _ctx()
    # Empty input simulates a pipe that closes immediately.
    result = runner.invoke(app, ["chat"], input="", obj=ctx)
    assert result.exit_code == 0
    assert "Aborted" not in result.output
    assert "Aborted" not in (result.stderr or "")


def test_chat_exit_after_valid_turn():
    """A real turn followed by /exit must exit cleanly with one graph call."""
    ctx = _ctx(response="hello back")
    result = runner.invoke(app, ["chat"], input="hello\n/exit\n", obj=ctx)
    assert result.exit_code == 0
    assert len(ctx["graph"].calls) == 1, "Expected exactly one graph invocation"
    assert "hello back" in result.output
    assert "Goodbye" in result.output


# ---------------------------------------------------------------------------
# loop exit tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", ["exit", "quit", "/exit"])
def test_loop_exit_tokens_terminate_without_model_turn(token):
    """Loop exit tokens must not trigger a graph invocation."""
    ctx = _ctx()
    result = runner.invoke(app, ["loop"], input=f"{token}\n", obj=ctx)
    assert ctx["graph"].calls == []
    assert result.exit_code == 0
    assert "Goodbye" in result.output or "Loop ended" in result.output


# ---------------------------------------------------------------------------
# _is_safe_target safety gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,expected", [
    ("tests/adapters/test_graph.py", True),
    ("tests/cli/test_commands.py", True),
    ("tests/", True),
    ("src/faust/cli/chat.py", False),        # outside tests/
    ("/etc/passwd", False),                  # absolute path
    ("../tests/evil.py", False),             # traversal
    ("tests/evil; rm -rf /", False),         # shell injection
    ("tests/evil`whoami`", False),           # backtick injection
    ("", False),                             # empty string
])
def test_is_safe_target(target, expected):
    assert _is_safe_target(target) is expected, (
        f"_is_safe_target({target!r}) expected {expected}"
    )
