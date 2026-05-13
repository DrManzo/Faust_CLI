"""Targeted exit-behavior tests for Step 10 CLI cleanup (Phase 1 + Phase 3).

Covers:
- Interactive exit / quit / /exit terminates the loop without a model turn.
- Piped stdin exhaustion exits cleanly (no 'Aborted.' on stderr).
- loop command exit tokens work.
- _is_safe_target safety gate including full pytest node-id forms.
- _extract_node_ids normalisation: valid pass-through, sloppy extraction,
  and hard rejection of unsafe targets.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from faust.cli.app import app
from faust.cli.commands.loop import _extract_node_ids, _is_safe_target
from faust.core.models import AppConfig

# Typer's CliRunner does not expose mix_stderr — instantiate with no kwargs.
runner = CliRunner()


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
    assert ctx["graph"].calls == [], (
        f"Graph was unexpectedly invoked after '{token}'"
    )
    assert result.exit_code == 0
    assert "Goodbye" in result.output or "Session ended" in result.output


def test_chat_piped_stdin_exhaustion_exits_cleanly():
    """When stdin is exhausted (empty pipe), no 'Aborted.' should leak."""
    ctx = _ctx()
    result = runner.invoke(app, ["chat"], input="", obj=ctx)
    assert result.exit_code == 0
    assert "Aborted" not in result.output


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
    # Plain file paths
    ("tests/adapters/test_graph.py", True),
    ("tests/cli/test_commands.py", True),
    ("tests/", True),
    # Full pytest node-ids (Phase 3 addition)
    ("tests/cli/test_exit_behavior.py::test_chat_exit_after_valid_turn", True),
    ("tests/adapters/test_graph.py::TestClass::test_method", True),
    # Unsafe targets
    ("src/faust/cli/chat.py", False),        # outside tests/
    ("/etc/passwd", False),                  # absolute path
    ("../tests/evil.py", False),             # traversal
    ("tests/evil; rm -rf /", False),         # shell injection
    ("tests/evil`whoami`", False),           # backtick injection
    ("", False),                             # empty string
    # Malformed node-ids that look right but aren't
    ("test_exit_behavior::test_chat_exit_after_valid_turn", False),   # missing tests/ prefix
    ("tests/cli/test_exit_behavior::test_chat_exit_after_valid_turn", False),  # missing .py
])
def test_is_safe_target(target, expected):
    assert _is_safe_target(target) is expected, (
        f"_is_safe_target({target!r}) expected {expected}"
    )


# ---------------------------------------------------------------------------
# _extract_node_ids normalisation (Phase 3)
# ---------------------------------------------------------------------------

def test_extract_node_ids_passes_valid_targets_unchanged():
    """Already-valid node-ids must be returned as-is without modification."""
    valid = [
        "tests/cli/test_exit_behavior.py::test_chat_exit_after_valid_turn",
        "tests/adapters/test_graph.py",
    ]
    result = _extract_node_ids(valid)
    assert result == valid


def test_extract_node_ids_rejects_missing_tests_prefix():
    """A node-id without the tests/ prefix must be dropped entirely."""
    result = _extract_node_ids(
        ["test_exit_behavior::test_chat_exit_after_valid_turn"]
    )
    assert result == [], "Expected empty list — no valid id extractable"


def test_extract_node_ids_extracts_embedded_node_id():
    """A sloppy string embedding a valid node-id must have the id extracted."""
    sloppy = "Run tests/cli/test_exit_behavior.py::test_chat_exit_after_valid_turn please"
    result = _extract_node_ids([sloppy])
    assert result == [
        "tests/cli/test_exit_behavior.py::test_chat_exit_after_valid_turn"
    ]


def test_extract_node_ids_rejects_pure_garbage():
    """Pure garbage strings that contain no valid node-id must be dropped."""
    result = _extract_node_ids(["not a test at all", "definitely::not::valid"])
    assert result == []


def test_extract_node_ids_empty_and_blanks():
    """Empty list and blank strings must produce an empty list."""
    assert _extract_node_ids([]) == []
    assert _extract_node_ids(["", "   "]) == []
