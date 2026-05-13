"""faust plugin/tool registry — Phase 3 of Step 11.

This module defines ``PluginRegistry``, a typed, introspectable store of
named callable tools that Faust can invoke as first-class graph operations.

Design rules:
  - Every registered tool declares its input_schema and requires_approval.
  - Tools with ``requires_approval=True`` MUST pass through the human gate
    before execution.  ``dispatch`` enforces this via ``PermissionError``.
  - Only two tools are built in this step: ``read_file`` and ``run_pytest``.
    No speculative tools are added.

Safety invariants:
  - ``read_file`` enforces that the path is inside the project ``src/``
    directory.  Traversal is rejected.
  - ``run_pytest`` runs only targets that pass ``_is_safe_target`` from
    the loop module.  It does not accept arbitrary shell commands.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PluginEntry:
    """A single registered tool.

    Attributes:
        name:             Unique tool identifier.
        fn:               The callable to invoke.
        input_schema:     Describes expected input keys and their types.
        requires_approval: If True, the operator must explicitly approve
                          before ``dispatch`` will call ``fn``.
        description:      Human-readable summary shown in proposals.
    """
    name: str
    fn: Callable[..., Any]
    input_schema: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    description: str = ""


class PluginRegistry:
    """Dict-backed registry of named, typed tool callables.

    Usage::

        registry = PluginRegistry()
        registry.register(PluginEntry(name="my_tool", fn=my_fn, ...))
        result = registry.dispatch("my_tool", {"arg": "value"}, approval_override=True)
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, PluginEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, entry: PluginEntry) -> None:
        """Register a plugin entry.  Raises ValueError on duplicate name."""
        if entry.name in self._plugins:
            raise ValueError(
                f"A plugin named '{entry.name}' is already registered. "
                "Unregister it first or use a different name."
            )
        self._plugins[entry.name] = entry

    def unregister(self, name: str) -> None:
        """Remove a plugin by name (no-op if not found)."""
        self._plugins.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[PluginEntry]:
        """Return the PluginEntry for *name*, or None if not registered."""
        return self._plugins.get(name)

    def list_names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._plugins.keys())

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        name: str,
        inputs: Dict[str, Any],
        *,
        approval_override: bool = False,
    ) -> Any:
        """Look up *name* and call its function with *inputs*.

        Args:
            name:              The registered tool name.
            inputs:            Keyword arguments forwarded to the tool function.
            approval_override: Must be True for tools with
                               ``requires_approval=True``.  The caller is
                               responsible for obtaining approval through the
                               human gate before setting this flag.

        Returns:
            The return value of the tool function.

        Raises:
            KeyError:        If *name* is not registered.
            PermissionError: If the tool requires approval and
                             ``approval_override`` is False.
        """
        entry = self._plugins.get(name)
        if entry is None:
            raise KeyError(f"No plugin registered under name '{name}'.")

        if entry.requires_approval and not approval_override:
            raise PermissionError(
                f"Tool '{name}' requires explicit operator approval before "
                "execution.  Route through the approval gate first."
            )

        return entry.fn(**inputs)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_SAFE_TARGET_RE = re.compile(
    r"^tests/[a-zA-Z0-9_/\-]*(?:\.py(?:::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)?)?)?/?$"
)


def _safe_test_target(target: str) -> bool:
    t = target.strip()
    if not t or not t.startswith("tests/"):
        return False
    if re.search(r"[;&|`$<>\\!]", t):
        return False
    return bool(_SAFE_TARGET_RE.match(t))


def _builtin_read_file(path: str) -> str:
    """Read a file and return its text content.

    The file must reside inside the project ``src/`` directory.  Path
    traversal attempts are rejected with ValueError.
    """
    target = Path(path)
    resolved = (Path.cwd() / target).resolve()
    src_root = (Path.cwd() / "src").resolve()

    try:
        resolved.relative_to(src_root)
    except ValueError:
        raise ValueError(
            f"read_file rejected: '{path}' is outside the allowed src/ boundary."
        )

    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")

    return resolved.read_text(encoding="utf-8")


def _builtin_run_pytest(targets: list[str]) -> dict:
    """Run pytest on the given *targets* (must pass _safe_test_target).

    Returns a dict with keys:
      returncode (int), stdout (str), stderr (str)
    """
    safe = [t for t in targets if _safe_test_target(t)]
    if not safe:
        raise ValueError(
            "run_pytest: no safe test targets provided. "
            "Targets must be rooted at tests/ with no shell metacharacters."
        )

    cmd = [sys.executable, "-m", "pytest"] + safe + ["-v", "--tb=short"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Global singleton registry
# ---------------------------------------------------------------------------

_REGISTRY: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """Return (and lazily initialise) the global plugin registry.

    The registry is populated with the two built-in tools on first access.
    Callers should use this function rather than instantiating PluginRegistry
    directly so there is a single shared registry in the process.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PluginRegistry()
        _REGISTRY.register(PluginEntry(
            name="read_file",
            fn=_builtin_read_file,
            input_schema={"path": "str"},
            requires_approval=False,
            description="Read a file inside src/ and return its text content.",
        ))
        _REGISTRY.register(PluginEntry(
            name="run_pytest",
            fn=_builtin_run_pytest,
            input_schema={"targets": "list[str]"},
            requires_approval=True,
            description="Run pytest on approved, scoped test targets.",
        ))
    return _REGISTRY
