"""tool_call graph node — dispatches to the PluginRegistry.

This node is invoked when the router classifies the user intent as
``tool_call``.  It reads ``state["tool_name"]`` and ``state["tool_inputs"]``
and dispatches to the registered plugin.

Approval gate:
  Tools with ``requires_approval=True`` (e.g. ``run_pytest``) must be
  pre-approved via the human gate before this node will execute them.
  If ``state["test_approved"]`` is False for such a tool, the node writes
  a ``test_proposal`` describing the pending call and returns early without
  executing.

Safety invariants:
  - No tool executes without approval when ``requires_approval=True``.
  - Errors from the tool are captured into ``state["error"]``; they do not
    propagate as exceptions out of the node.
  - This node does NOT write to disk directly; file writes remain the
    exclusive domain of ``loop_editor.write_approved_change``.
"""

from __future__ import annotations

from typing import Any

from faust.core.plugins import get_registry


def tool_call_node(state: dict) -> dict:
    """Dispatch a registered tool call, gated by approval for protected tools.

    State fields read:
      tool_name (str):        The registered plugin name to invoke.
      tool_inputs (dict):     Keyword arguments forwarded to the plugin.
      test_approved (bool):   Whether the operator has approved execution.

    State fields written:
      tool_output (Any):      The return value from the plugin (on success).
      test_proposal (dict):   Populated when approval is pending; describes
                              the call so it can be displayed to the operator.
      error (str | None):     Populated when the tool raises an exception.
      active_agent (str):     Always set to 'tool_call'.
    """
    state = dict(state)
    state["active_agent"] = "tool_call"
    state["tool_output"] = None
    state["error"] = None

    tool_name: str = state.get("tool_name", "")
    tool_inputs: dict = state.get("tool_inputs") or {}
    approved: bool = bool(state.get("test_approved", False))

    if not tool_name:
        state["error"] = "tool_call_node: no tool_name in state."
        return state

    registry = get_registry()
    entry = registry.get(tool_name)

    if entry is None:
        state["error"] = f"tool_call_node: no plugin registered under name '{tool_name}'."
        return state

    # Approval gate for protected tools
    if entry.requires_approval and not approved:
        state["test_proposal"] = {
            "description": (
                f"Tool call pending approval: [bold]{tool_name}[/bold]\n"
                f"Inputs: {tool_inputs}\n"
                f"{entry.description}"
            ),
            "tool_name": tool_name,
            "tool_inputs": tool_inputs,
            "test_targets": tool_inputs.get("targets", []),
        }
        return state

    try:
        result = registry.dispatch(tool_name, tool_inputs, approved=approved)
        state["tool_output"] = result
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"tool_call_node: {type(exc).__name__}: {exc}"

    return state
