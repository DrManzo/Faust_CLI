"""Isolated unit tests for the PluginRegistry and built-in tools.

Step 11 / Phase 3 validation target:
  pytest tests/core/ -v --tb=short

Design:
  - All filesystem operations are patched via monkeypatch / tmp_path so
    no real project files are read or written.
  - No subprocess calls are made; _builtin_run_pytest is patched where tested.
  - Each test is fully self-contained with no shared mutable state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from faust.core.plugins import (
    PluginEntry,
    PluginRegistry,
    _builtin_read_file,
    _safe_test_target,
    get_registry,
)


# ---------------------------------------------------------------------------
# PluginRegistry — registration and lookup
# ---------------------------------------------------------------------------

def test_register_and_lookup_returns_entry():
    registry = PluginRegistry()
    entry = PluginEntry(name="echo", fn=lambda text: text, input_schema={"text": "str"})
    registry.register(entry)
    assert registry.get("echo") is entry


def test_register_duplicate_raises_value_error():
    registry = PluginRegistry()
    entry = PluginEntry(name="echo", fn=lambda text: text)
    registry.register(entry)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(PluginEntry(name="echo", fn=lambda text: text))


def test_get_unknown_returns_none():
    registry = PluginRegistry()
    assert registry.get("nonexistent") is None


def test_list_names_sorted():
    registry = PluginRegistry()
    registry.register(PluginEntry(name="beta", fn=lambda: None))
    registry.register(PluginEntry(name="alpha", fn=lambda: None))
    assert registry.list_names() == ["alpha", "beta"]


def test_unregister_removes_entry():
    registry = PluginRegistry()
    registry.register(PluginEntry(name="temp", fn=lambda: None))
    registry.unregister("temp")
    assert registry.get("temp") is None


# ---------------------------------------------------------------------------
# PluginRegistry — dispatch
# ---------------------------------------------------------------------------

def test_dispatch_calls_fn_and_returns_result():
    registry = PluginRegistry()
    registry.register(PluginEntry(
        name="add",
        fn=lambda a, b: a + b,
        input_schema={"a": "int", "b": "int"},
        requires_approval=False,
    ))
    result = registry.dispatch("add", {"a": 2, "b": 3}, approval_override=False)
    assert result == 5


def test_dispatch_approval_required_without_flag_raises_permission_error():
    registry = PluginRegistry()
    registry.register(PluginEntry(
        name="guarded",
        fn=lambda: "secret",
        requires_approval=True,
    ))
    with pytest.raises(PermissionError, match="requires explicit operator approval"):
        registry.dispatch("guarded", {}, approval_override=False)


def test_dispatch_approval_required_with_flag_succeeds():
    registry = PluginRegistry()
    registry.register(PluginEntry(
        name="guarded2",
        fn=lambda: "ok",
        requires_approval=True,
    ))
    result = registry.dispatch("guarded2", {}, approval_override=True)
    assert result == "ok"


def test_dispatch_unknown_name_raises_key_error():
    registry = PluginRegistry()
    with pytest.raises(KeyError, match="No plugin registered"):
        registry.dispatch("ghost", {}, approval_override=True)


# ---------------------------------------------------------------------------
# _builtin_read_file — boundary guard
# ---------------------------------------------------------------------------

def test_read_file_rejects_path_outside_src(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    evil = tmp_path / "evil.txt"
    evil.write_text("I am evil")
    with pytest.raises(ValueError, match="outside the allowed src/ boundary"):
        _builtin_read_file("../evil.txt")


def test_read_file_reads_valid_src_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src" / "faust"
    src.mkdir(parents=True)
    target = src / "hello.py"
    target.write_text("# hello")
    content = _builtin_read_file("src/faust/hello.py")
    assert content == "# hello"


# ---------------------------------------------------------------------------
# _safe_test_target guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,expected", [
    ("tests/core/test_plugins.py", True),
    ("tests/", True),
    ("tests/core/test_plugins.py::test_register_and_lookup_returns_entry", True),
    ("src/faust/core/plugins.py", False),
    ("/etc/passwd", False),
    ("tests/evil; rm -rf /", False),
    ("", False),
])
def test_safe_test_target_gate(target, expected):
    assert _safe_test_target(target) is expected


# ---------------------------------------------------------------------------
# get_registry — singleton with built-in tools
# ---------------------------------------------------------------------------

def test_get_registry_contains_built_in_tools():
    registry = get_registry()
    assert registry.get("read_file") is not None
    assert registry.get("run_pytest") is not None


def test_run_pytest_requires_approval():
    registry = get_registry()
    entry = registry.get("run_pytest")
    assert entry is not None
    assert entry.requires_approval is True


def test_read_file_does_not_require_approval():
    registry = get_registry()
    entry = registry.get("read_file")
    assert entry is not None
    assert entry.requires_approval is False
