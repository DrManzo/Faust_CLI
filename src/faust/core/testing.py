"""Step 9 tests-only helper workflow for Faust.

This module provides the narrow, human-controlled test-assistant workflow:
- detect_test_request  — classify whether a user message is asking for test help
- format_test_proposal — wrap raw LLM test output into a reviewable proposal dict
- validate_pytest_targets — reject any target outside tests/ before execution
- build_approval_prompt — build the terminal string asking the human to approve
- record_execution_note — normalise execution notes for state injection

Design constraints (Step 9):
- This module NEVER runs pytest itself.
- This module NEVER modifies production code.
- All pytest execution is gated in adapters/graph.py::run_requested_tests, which
  requires test_approved=True in state before it fires.
- This module has no side-effects: every function is pure or writes only to
  the reports/ directory via _write_test_report (imported from graph).
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_TEST_DRAFT_MARKERS: tuple[str, ...] = (
    "draft a test",
    "draft test",
    "propose a test",
    "write a test for",
    "write tests for",
    "suggest a test",
    "generate a test",
)

_RUN_TEST_MARKERS: tuple[str, ...] = (
    "run the test",
    "run tests",
    "run scoped",
    "execute the test",
    "execute tests",
    "pytest ",
    "run pytest",
)


def detect_test_request(user_input: str) -> str:
    """Return 'test_draft', 'test_run', or 'none'.

    'test_draft' — user wants Faust to propose a test.
    'test_run'   — user wants to run already-approved scoped tests.
    'none'       — not a test-workflow request.

    Priority: test_draft is checked before test_run so that
    'draft and then run' phrasing maps to draft first.
    """
    normalised = re.sub(r"\s+", " ", user_input.strip().lower())
    if not normalised:
        return "none"

    if any(marker in normalised for marker in _TEST_DRAFT_MARKERS):
        return "test_draft"

    if any(marker in normalised for marker in _RUN_TEST_MARKERS):
        return "test_run"

    return "none"


# ---------------------------------------------------------------------------
# Proposal formatting
# ---------------------------------------------------------------------------


def format_test_proposal(raw_llm_output: str, user_input: str) -> dict:
    """Wrap raw LLM test output into a structured proposal dict.

    Returns a dict with keys:
        proposal_text  — the raw proposed test code/content
        user_input     — the original request that triggered this proposal
        approved       — always False; human must explicitly approve
        targets        — empty list; filled by human on approval
        status         — 'pending_review'

    This function has no side-effects.
    """
    return {
        "proposal_text": raw_llm_output.strip(),
        "user_input": user_input,
        "approved": False,
        "targets": [],
        "status": "pending_review",
    }


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

_UNSAFE_TOKENS: tuple[str, ...] = (";", "&&", "||", "|", "`", "$(", "..\\")


def validate_pytest_targets(targets: list[str]) -> tuple[list[str], list[str]]:
    """Split targets into (valid, rejected).

    Valid targets must:
    - start with 'tests/'
    - not contain unsafe shell tokens
    - resolve to an existing file on disk

    Returns (valid_targets, rejected_targets).  Does NOT run anything.
    """
    valid: list[str] = []
    rejected: list[str] = []

    for raw in targets:
        target = raw.strip()
        if not target:
            continue

        if not target.startswith("tests/"):
            rejected.append(target)
            continue

        if any(tok in target for tok in _UNSAFE_TOKENS):
            rejected.append(target)
            continue

        file_part = target.split("::", 1)[0]
        if not Path(file_part).exists():
            rejected.append(target)
            continue

        valid.append(target)

    return valid, rejected


# ---------------------------------------------------------------------------
# Approval prompt builder
# ---------------------------------------------------------------------------


def build_approval_prompt(proposal: dict) -> str:
    """Return a concise terminal string presenting the proposal for human review.

    The string is informational only — it is shown to the human before any
    execution is considered.  It does not trigger any action itself.
    """
    lines = [
        "\n─── Faust test proposal (Step 9) ───────────────────────────────────",
        "",
        proposal.get("proposal_text", "").strip(),
        "",
        "────────────────────────────────────────────────────────────────────",
        "Review the proposed test above.",
        "To approve and run scoped targets, reply with the pytest node IDs",
        "  (e.g. 'run tests/adapters/test_graph.py::test_foo').",
        "To discard, just continue the conversation.",
        "Faust will NOT run anything until you explicitly approve.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution note normaliser
# ---------------------------------------------------------------------------


def record_execution_note(
    *,
    targets: list[str],
    rejected: list[str],
    returncode: int,
    report_path: str | None,
) -> str:
    """Return a concise single-line execution note suitable for state injection.

    Full pytest output lives in the report file; the terminal note is compact.
    """
    status = "passed" if returncode == 0 else "FAILED"
    parts = [f"Scoped pytest run {status}: {', '.join(targets) or 'none'}."]
    parts.append(f"Exit code: {returncode}.")
    if report_path:
        parts.append(f"Report: {report_path}")
    if rejected:
        parts.append("Rejected (outside tests/): " + ", ".join(rejected) + ".")
    return "  ".join(parts)
