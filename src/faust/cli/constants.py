"""Shared CLI constants for faust chat and faust loop.

Centralises the approval regex, exit tokens, and the Typer option
resolver so that chat.py and loop.py stay in sync without duplication.
"""

from __future__ import annotations

import re

from typer.models import OptionInfo

# ---------------------------------------------------------------------------
# Approval vocabulary
# ---------------------------------------------------------------------------

# Matches explicit approval at the start of a user turn.
# Covers: "approve", "approved", "yes", "yes run", "yes execute",
#         "yes go ahead", "APPROVE", "confirm" (added for loop UX).
_APPROVE_RE = re.compile(
    r"^\s*(approved?|yes[,.]?\s*(run|execute|go\s+ahead)?|APPROVE|confirm)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Exit vocabulary
# ---------------------------------------------------------------------------

# All tokens that trigger a clean exit BEFORE a model turn is attempted.
# Checked against the lowercased, stripped user input.
_EXIT_TOKENS: frozenset[str] = frozenset({"exit", "quit", "q", "/exit"})

# ---------------------------------------------------------------------------
# Typer option resolver
# ---------------------------------------------------------------------------

def _resolve_option(value: object, fallback: str) -> str:
    """Return *fallback* when *value* is a raw Typer OptionInfo sentinel.

    Typer sometimes hands the ``OptionInfo`` object through ``ctx.obj``
    instead of the resolved default string.  This helper normalises that
    so callers always receive a plain ``str``.
    """
    if isinstance(value, OptionInfo):
        return fallback
    return str(value)
