"""Tests for faust.cli.renderer.

All four public functions are covered. Rich output is captured via
StringIO so nothing leaks to stdout during the test run.
"""

from __future__ import annotations

import io

from rich.console import Console

from faust.cli import renderer as _renderer_module
from faust.cli.renderer import print_banner, print_error, print_response, print_session_info
from faust.core.models import Session


def _capture(fn, *args, **kwargs) -> str:
    """Run *fn* with a captured Rich Console and return the text output."""
    buf = io.StringIO()
    cap = Console(file=buf, highlight=False, markup=True)
    original = _renderer_module.console
    _renderer_module.console = cap
    try:
        fn(*args, **kwargs)
    finally:
        _renderer_module.console = original
    return buf.getvalue()


def test_print_banner_renders_without_error():
    """print_banner must complete without raising and produce non-empty output."""
    output = _capture(print_banner)
    assert output.strip(), "Expected non-empty banner output"
    assert "FAUST" in output


def test_print_response_renders_text():
    """print_response must render the supplied text without raising."""
    output = _capture(print_response, "Hello **world**")
    assert "Hello" in output
    assert "world" in output


def test_print_error_includes_message():
    """print_error must include the error message in its output."""
    output = _capture(print_error, "something went wrong")
    assert "something went wrong" in output


def test_print_session_info_includes_id_model_and_turns():
    """print_session_info must surface session id prefix, model, and turn count."""
    session = Session(id="abcdef1234567890", model="gpt-4o")
    output = _capture(print_session_info, session)
    assert "abcdef12" in output, "Expected first 8 chars of session id"
    assert "gpt-4o" in output
    assert "0" in output  # turns
