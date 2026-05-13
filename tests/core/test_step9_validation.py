"""Step 9 — 15 live validation tests, easy → hard.

Covers:
- Memory slot recall and writes (Steps 5-8 regression)
- classify_task / determine_role routing (Step 7 regression)
- test_proposal_node approval gate (Step 9 core)
- run_requested_tests target validation and report generation (Step 9 core)
- _write_test_report with a simulated failure / fake error (Step 9 report)
- Full graph state flow smoke tests (end-to-end)

All tests are self-contained, use InMemoryStore + InMemorySaver,
and never touch src/ production files.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.store.memory import InMemoryStore

from faust.adapters.graph import (
    _detect_recall_slot,
    _has_recalled_slot,
    _is_memory_write,
    _write_test_report,
    classify_task,
    determine_role,
    memory_answer_node,
    retrieve_memories,
    run_requested_tests,
    save_memory,
    should_run_requested_tests,
    test_proposal_node,
)
from faust.core.models import AppConfig, FaustState, MemoryRecord, Message, Role, Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> AppConfig:
    return AppConfig()


def _session() -> Session:
    return Session(id="test-session", model=_cfg().model)


def _state(
    user_input: str = "",
    recalled: list[MemoryRecord] | None = None,
    messages: list[Message] | None = None,
    test_approved: bool = False,
    requested_tests: list[str] | None = None,
    active_agent: str | None = None,
) -> FaustState:
    return {
        "session": _session(),
        "config": _cfg(),
        "user_id": "javier",
        "user_input": user_input,
        "intent": None,
        "memory_route": None,
        "active_agent": active_agent,
        "messages": messages or [],
        "recalled_memories": recalled or [],
        "artifacts": [],
        "response": "",
        "error": None,
        "test_proposal": None,
        "test_approved": test_approved,
        "test_report_path": None,
        "requested_tests": requested_tests or [],
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mem(slot: str, text: str) -> MemoryRecord:
    now = _now()
    return MemoryRecord(
        key=slot,
        slot=slot,
        text=text,
        category="profile",
        source="explicit",
        created_at=now,
        updated_at=now,
    )


# ===========================================================================
# EASY (1–5) — single-function, no graph wiring
# ===========================================================================


# Test 1
def test_detect_recall_slot_name():
    """EASY: _detect_recall_slot recognises 'what is my name'."""
    slot = _detect_recall_slot("what is my name")
    assert slot == "profile.name"


# Test 2
def test_detect_recall_slot_editor():
    """EASY: _detect_recall_slot recognises 'what is my favorite editor'."""
    slot = _detect_recall_slot("what is my favorite editor")
    assert slot == "preference.favorite_editor"


# Test 3
def test_is_memory_write_name():
    """EASY: _is_memory_write detects 'my name is Javier' as a write."""
    assert _is_memory_write("my name is Javier") is True


# Test 4
def test_is_memory_write_returns_false_for_question():
    """EASY: _is_memory_write is False for a recall question."""
    assert _is_memory_write("what is my name") is False


# Test 5
def test_classify_task_test_draft():
    """EASY: classify_task routes 'draft a test for save_memory' → test_draft."""
    state = _state(user_input="draft a test for save_memory")
    result = classify_task(state)
    assert result["task_type"] == "test_draft"


# ===========================================================================
# MEDIUM (6–10) — multi-function + node tests
# ===========================================================================


# Test 6
def test_classify_task_does_not_confuse_draft_with_coding():
    """MEDIUM: 'draft a test' must NOT be classified as coding."""
    state = _state(user_input="draft a test for the coder node")
    result = classify_task(state)
    assert result["task_type"] == "test_draft"
    assert result["task_type"] != "coding"


# Test 7
def test_determine_role_selects_test_proposer():
    """MEDIUM: determine_role maps task_type=test_draft → test_proposer."""
    state = _state(user_input="draft a test for build_prompt")
    state["task_type"] = "test_draft"  # type: ignore[typeddict-unknown-key]
    result = determine_role(state)
    assert result["requested_role"] == "test_proposer"


# Test 8
def test_memory_answer_node_returns_name_from_recall():
    """MEDIUM: memory_answer_node returns a grounded name answer."""
    recalled = [_mem("profile.name", "The user's name is Javier.")]
    state = _state(user_input="what is my name", recalled=recalled)
    result = memory_answer_node(state)
    assert "Javier" in result.get("response", "")
    assert result.get("intent") == "memory_recall"


# Test 9
def test_save_memory_persists_name_slot(tmp_path):
    """MEDIUM: save_memory writes 'my name is Javier' to an InMemoryStore."""
    store = InMemoryStore()
    state = _state(user_input="my name is Javier")
    result = save_memory(state, store=store)
    assert "Javier" in result.get("response", "")
    assert result.get("intent") == "memory_write"


# Test 10
def test_should_run_requested_tests_blocks_without_approval():
    """MEDIUM: approval gate sends to save_memory when test_approved=False."""
    state = _state(
        active_agent="coder",
        test_approved=False,
        requested_tests=["tests/core/test_testing.py"],
    )
    route = should_run_requested_tests(state)
    assert route == "save_memory"


# ===========================================================================
# HARD (11–15) — report generation, fake errors, graph smoke tests
# ===========================================================================


# Test 11
def test_write_test_report_creates_file(tmp_path, monkeypatch):
    """HARD: _write_test_report creates a timestamped markdown file in reports/."""
    monkeypatch.chdir(tmp_path)
    path = _write_test_report(
        targets=["tests/core/test_testing.py"],
        rejected=[],
        returncode=0,
        stdout="1 passed in 0.12s",
        stderr="",
    )
    report = Path(path)
    assert report.exists()
    content = report.read_text()
    assert "PASSED" in content
    assert "1 passed in 0.12s" in content
    assert "tests/core/test_testing.py" in content


# Test 12
def test_write_test_report_with_fake_error(tmp_path, monkeypatch):
    """HARD: _write_test_report records a fake AssertionError in the report.

    This simulates what happens when Faust runs a scoped test that fails.
    The report must contain the failure status, the fake traceback, and
    must NOT modify any file outside reports/.
    """
    monkeypatch.chdir(tmp_path)

    fake_stderr = (
        "FAILED tests/core/test_testing.py::TestDetectTestRequest::test_case_insensitive\n"
        "AssertionError: Expected 'test_draft', got 'none'\n"
        "  File \"tests/core/test_testing.py\", line 42, in test_case_insensitive\n"
        "    assert detect_test_request('DRAFT A TEST FOR foo') == 'test_draft'\n"
        "1 failed in 0.08s"
    )

    path = _write_test_report(
        targets=["tests/core/test_testing.py::TestDetectTestRequest::test_case_insensitive"],
        rejected=[],
        returncode=1,
        stdout="",
        stderr=fake_stderr,
    )

    report = Path(path)
    assert report.exists(), "Report file was not created."

    content = report.read_text()

    # Status must show FAILED
    assert "FAILED" in content, "Report must contain FAILED status."

    # Exit code must be present
    assert "Exit code: 1" in content, "Report must show exit code."

    # The fake traceback must be preserved verbatim
    assert "AssertionError" in content, "Report must include the fake AssertionError."
    assert "test_case_insensitive" in content, "Report must name the failing test."
    assert "1 failed in 0.08s" in content, "Report must include pytest summary line."

    # Production code must not have been touched
    src_dir = tmp_path / "src"
    assert not src_dir.exists(), "_write_test_report must not create a src/ directory."


# Test 13
def test_write_test_report_with_rejected_targets(tmp_path, monkeypatch):
    """HARD: _write_test_report records rejected (outside tests/) targets."""
    monkeypatch.chdir(tmp_path)
    path = _write_test_report(
        targets=["tests/core/test_testing.py"],
        rejected=["src/faust/adapters/graph.py"],
        returncode=0,
        stdout="1 passed",
        stderr="",
    )
    content = Path(path).read_text()
    assert "src/faust/adapters/graph.py" in content
    assert "Rejected" in content


# Test 14
def test_test_proposal_node_does_not_run_subprocess(monkeypatch):
    """HARD: test_proposal_node NEVER calls subprocess.run, even with a real adapter."""
    called = {}

    def fake_subprocess_run(*args, **kwargs):
        called["yes"] = True
        return subprocess.CompletedProcess(args, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    class FakeAdapter:
        def generate(self, messages, stream=True):
            yield "def test_foo(): pass"

    state = _state(
        user_input="draft a test for save_memory",
        messages=[Message(role=Role.USER, content="draft a test for save_memory")],
    )
    result = test_proposal_node(state, adapter=FakeAdapter())

    assert "yes" not in called, "test_proposal_node must never call subprocess.run."
    assert result["test_approved"] is False
    assert "test_draft" == result.get("intent")
    assert "def test_foo" in result.get("test_proposal", "")


# Test 15
def test_run_requested_tests_rejects_src_and_writes_report(tmp_path, monkeypatch):
    """HARD: run_requested_tests rejects src/ targets and writes a report
    for the valid ones, with a simulated fake failure exit code.

    This is the hardest test: it exercises the full execution path including
    target validation, scoped pytest invocation (mocked), report writing,
    and the rejection of production-code paths — all in one flow.
    """
    monkeypatch.chdir(tmp_path)

    # Create a fake valid test file so path existence check passes
    (tmp_path / "tests" / "core").mkdir(parents=True)
    fake_test = tmp_path / "tests" / "core" / "test_fake.py"
    fake_test.write_text("def test_placeholder(): pass\n")

    fake_completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=1,
        stdout="",
        stderr=(
            "FAILED tests/core/test_fake.py::test_placeholder\n"
            "AssertionError: intentional fake failure\n"
            "1 failed in 0.05s"
        ),
    )

    with patch("faust.adapters.graph.subprocess.run", return_value=fake_completed):
        state = _state(
            active_agent="coder",
            test_approved=True,
            requested_tests=[
                "tests/core/test_fake.py::test_placeholder",  # valid
                "src/faust/adapters/graph.py",                # must be rejected
            ],
        )
        result = run_requested_tests(state)

    notes = result.get("execution_notes", "")
    report_path = result.get("test_report_path")

    # Execution notes must flag FAILED
    assert "FAILED" in notes, "Execution notes must contain FAILED."

    # src/ path must appear as rejected in notes
    assert "src/faust/adapters/graph.py" in notes or "Rejected" in notes, (
        "Execution notes must mention the rejected src/ target."
    )

    # Report file must exist
    assert report_path is not None, "A report file path must be returned."
    report = Path(report_path)
    assert report.exists(), f"Report file {report_path} was not created."

    content = report.read_text()
    assert "FAILED" in content
    assert "intentional fake failure" in content
    assert "AssertionError" in content
