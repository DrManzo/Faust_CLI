"""Tests for Faust CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from faust.cli.app import app
from faust.core.models import AppConfig


runner = CliRunner()


class FakeGraph:
    """Small fake graph object that mimics graph.invoke()."""

    def __init__(self, response: str = "fake response", error: str | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append({"state": state, "config": config})
        return {
            "response": self.response,
            "error": self.error,
            "messages": state.get("messages", []),
        }


def make_ctx_obj(response: str = "fake response", error: str | None = None):
    """Build the ctx.obj payload expected by the CLI commands."""
    return {
        "config": AppConfig(),
        "graph": FakeGraph(response=response, error=error),
    }


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "faust" in result.output.lower()


def test_run_command_prints_response():
    ctx_obj = make_ctx_obj(response="hello from run")

    result = runner.invoke(app, ["run", "hello"], obj=ctx_obj)

    assert result.exit_code == 0
    assert "hello from run" in result.output


def test_run_command_exits_with_error_when_graph_returns_error():
    ctx_obj = make_ctx_obj(error="graph failed")

    result = runner.invoke(app, ["run", "hello"], obj=ctx_obj)

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "graph failed" in result.output


def test_chat_command_single_turn_and_quit():
    ctx_obj = make_ctx_obj(response="hello from chat")

    result = runner.invoke(
        app,
        ["chat"],
        input="hello\nquit\n",
        obj=ctx_obj,
    )

    assert result.exit_code == 0
    assert "hello from chat" in result.output


def test_default_callback_starts_chat_when_no_subcommand():
    ctx_obj = make_ctx_obj(response="hello from default chat")

    result = runner.invoke(
        app,
        [],
        input="hello\nquit\n",
        obj=ctx_obj,
    )

    assert result.exit_code == 0
    assert "hello from default chat" in result.output


