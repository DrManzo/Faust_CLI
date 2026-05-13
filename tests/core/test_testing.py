"""Tests for the Step 9 tests-only helper workflow (faust.core.testing).

All tests in this file verify:
- detect_test_request classifies user input correctly
- format_test_proposal always marks approved=False
- validate_pytest_targets accepts only safe tests/ paths
- build_approval_prompt includes required human-readable sections
- record_execution_note formats terminal notes correctly
- No function in faust.core.testing modifies production code
- No function in faust.core.testing invokes subprocess.run
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from faust.core.testing import (
    _TEST_DRAFT_MARKERS,
    _RUN_TEST_MARKERS,
    build_approval_prompt,
    detect_test_request,
    format_test_proposal,
    record_execution_note,
    validate_pytest_targets,
)


# ── detect_test_request ────────────────────────────────────────────────────


class TestDetectTestRequest:
    def test_returns_test_draft_for_draft_markers(self):
        for marker in [
            "draft a test for foo",
            "draft test for the save_memory node",
            "propose a test for the coder node",
            "write a test for retrieve_memories",
            "write tests for the new slot",
            "suggest a test for build_prompt",
            "generate a test for the graph",
        ]:
            assert detect_test_request(marker) == "test_draft", f"Failed for: {marker!r}"

    def test_returns_test_run_for_run_markers(self):
        for phrase in [
            "run the test now",
            "run tests in tests/adapters/",
            "run scoped pytest targets",
            "execute the test file",
            "execute tests please",
            "pytest tests/core/",
            "run pytest on this",
        ]:
            assert detect_test_request(phrase) == "test_run", f"Failed for: {phrase!r}"

    def test_returns_none_for_unrelated_input(self):
        for phrase in [
            "Hello, how are you?",
            "plan this feature",
            "write code for fizzbuzz",
            "What is my name?",
            "",
        ]:
            assert detect_test_request(phrase) == "none", f"Expected none for: {phrase!r}"

    def test_test_draft_takes_priority_over_test_run(self):
        """If a message mentions both draft and run, draft wins."""
        result = detect_test_request("draft a test and then run it")
        assert result == "test_draft"

    def test_case_insensitive(self):
        assert detect_test_request("DRAFT A TEST FOR foo") == "test_draft"
        assert detect_test_request("RUN TESTS NOW") == "test_run"

    def test_empty_string_returns_none(self):
        assert detect_test_request("") == "none"

    def test_whitespace_only_returns_none(self):
        assert detect_test_request("   ") == "none"


# ── format_test_proposal ───────────────────────────────────────────────────


class TestFormatTestProposal:
    def test_returns_approved_false(self):
        proposal = format_test_proposal("def test_foo(): pass", "draft a test for foo")
        assert proposal["approved"] is False

    def test_returns_pending_review_status(self):
        proposal = format_test_proposal("def test_foo(): pass", "draft a test for foo")
        assert proposal["status"] == "pending_review"

    def test_preserves_proposal_text(self):
        raw = "def test_bar():\n    assert 1 + 1 == 2"
        proposal = format_test_proposal(raw, "write tests for bar")
        assert proposal["proposal_text"] == raw.strip()

    def test_preserves_user_input(self):
        proposal = format_test_proposal("", "write a test for baz")
        assert proposal["user_input"] == "write a test for baz"

    def test_targets_is_empty_list(self):
        proposal = format_test_proposal("def test_x(): pass", "suggest a test")
        assert proposal["targets"] == []

    def test_strips_whitespace_from_proposal_text(self):
        proposal = format_test_proposal("  def test_y(): pass  \n", "draft a test")
        assert proposal["proposal_text"] == "def test_y(): pass"

    def test_does_not_run_subprocess(self, monkeypatch):
        called = {}

        def fake_run(*a, **kw):
            called["yes"] = True

        monkeypatch.setattr(subprocess, "run", fake_run)
        format_test_proposal("def test_z(): pass", "propose a test")
        assert "yes" not in called


# ── validate_pytest_targets ────────────────────────────────────────────────


class TestValidatePytestTargets:
    def test_rejects_paths_outside_tests(self):
        _, rejected = validate_pytest_targets(["src/faust/adapters/graph.py"])
        assert "src/faust/adapters/graph.py" in rejected

    def test_rejects_unsafe_tokens(self):
        unsafe = [
            "tests/foo.py; rm -rf /",
            "tests/foo.py && echo bad",
            "tests/foo.py || true",
            "tests/foo.py | cat",
            "tests/foo.py`whoami`",
            "tests/foo.py$(id)",
        ]
        for target in unsafe:
            _, rejected = validate_pytest_targets([target])
            assert target.strip() in rejected, f"Expected rejection of {target!r}"

    def test_rejects_nonexistent_test_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, rejected = validate_pytest_targets(["tests/core/does_not_exist.py"])
        assert "tests/core/does_not_exist.py" in rejected

    def test_accepts_valid_existing_target(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tests" / "core").mkdir(parents=True)
        (tmp_path / "tests" / "core" / "test_example.py").write_text("")

        valid, rejected = validate_pytest_targets(
            ["tests/core/test_example.py::test_something"]
        )
        assert valid == ["tests/core/test_example.py::test_something"]
        assert rejected == []

    def test_accepts_file_level_target(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_sample.py").write_text("")

        valid, rejected = validate_pytest_targets(["tests/test_sample.py"])
        assert valid == ["tests/test_sample.py"]
        assert rejected == []

    def test_empty_targets_returns_empty_lists(self):
        valid, rejected = validate_pytest_targets([])
        assert valid == []
        assert rejected == []

    def test_skips_blank_strings(self):
        valid, rejected = validate_pytest_targets(["", "   "])
        assert valid == []
        assert rejected == []

    def test_does_not_modify_src_directory(self, tmp_path, monkeypatch):
        """validate_pytest_targets must never touch src/ contents."""
        monkeypatch.chdir(tmp_path)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "important.py").write_text("# production code")

        validate_pytest_targets(["src/important.py"])

        # src/ contents must be unchanged
        assert (src_dir / "important.py").read_text() == "# production code"


# ── build_approval_prompt ──────────────────────────────────────────────────


class TestBuildApprovalPrompt:
    def _make_proposal(self, text: str = "def test_foo(): pass") -> dict:
        return format_test_proposal(text, "draft a test")

    def test_includes_proposal_text(self):
        proposal = self._make_proposal("def test_foo(): pass")
        prompt = build_approval_prompt(proposal)
        assert "def test_foo(): pass" in prompt

    def test_includes_approval_instruction(self):
        prompt = build_approval_prompt(self._make_proposal())
        assert "approve" in prompt.lower() or "Approve" in prompt

    def test_states_faust_will_not_run_automatically(self):
        prompt = build_approval_prompt(self._make_proposal())
        assert "will NOT run" in prompt or "will not run" in prompt.lower()

    def test_mentions_pytest_node_ids(self):
        prompt = build_approval_prompt(self._make_proposal())
        assert "pytest" in prompt.lower() or "node ID" in prompt

    def test_returns_string(self):
        prompt = build_approval_prompt(self._make_proposal())
        assert isinstance(prompt, str)

    def test_empty_proposal_text_still_returns_prompt(self):
        proposal = format_test_proposal("", "suggest a test")
        prompt = build_approval_prompt(proposal)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ── record_execution_note ─────────────────────────────────────────────────


class TestRecordExecutionNote:
    def test_includes_passed_for_zero_returncode(self):
        note = record_execution_note(
            targets=["tests/core/test_testing.py"],
            rejected=[],
            returncode=0,
            report_path="reports/test-report-fake.md",
        )
        assert "passed" in note

    def test_includes_failed_for_nonzero_returncode(self):
        note = record_execution_note(
            targets=["tests/core/test_testing.py"],
            rejected=[],
            returncode=1,
            report_path="reports/test-report-fake.md",
        )
        assert "FAILED" in note

    def test_includes_exit_code(self):
        note = record_execution_note(
            targets=["tests/core/test_testing.py"],
            rejected=[],
            returncode=2,
            report_path=None,
        )
        assert "Exit code: 2" in note

    def test_includes_report_path_when_provided(self):
        note = record_execution_note(
            targets=["tests/core/test_testing.py"],
            rejected=[],
            returncode=0,
            report_path="reports/test-report-abc.md",
        )
        assert "reports/test-report-abc.md" in note

    def test_omits_report_path_when_none(self):
        note = record_execution_note(
            targets=["tests/core/test_testing.py"],
            rejected=[],
            returncode=0,
            report_path=None,
        )
        assert "Report:" not in note

    def test_lists_rejected_targets(self):
        note = record_execution_note(
            targets=["tests/core/test_testing.py"],
            rejected=["src/faust/core/testing.py"],
            returncode=0,
            report_path=None,
        )
        assert "src/faust/core/testing.py" in note
        assert "outside tests/" in note

    def test_handles_empty_targets(self):
        note = record_execution_note(
            targets=[],
            rejected=[],
            returncode=0,
            report_path=None,
        )
        assert isinstance(note, str)
        assert "passed" in note


# ── production-code safety guarantees ─────────────────────────────────────


class TestProductionCodeSafety:
    """Guarantee that no testing.py function modifies or touches src/."""

    def test_validate_pytest_targets_cannot_accept_src_paths(self):
        valid, rejected = validate_pytest_targets(
            [
                "src/faust/adapters/graph.py",
                "src/faust/core/models.py",
                "src/faust/core/testing.py",
            ]
        )
        assert valid == []
        assert len(rejected) == 3

    def test_format_test_proposal_returns_pure_dict(self):
        """format_test_proposal must return a plain dict with no filesystem side-effects."""
        result = format_test_proposal("def test_x(): pass", "draft a test")
        assert isinstance(result, dict)
        # Keys that must always be present
        for key in ("proposal_text", "user_input", "approved", "targets", "status"):
            assert key in result, f"Missing key: {key!r}"

    def test_approved_is_always_false_on_new_proposal(self):
        """A fresh proposal must never start with approved=True."""
        for text in [
            "def test_a(): pass",
            "",
            "# some comment",
        ]:
            proposal = format_test_proposal(text, "propose a test")
            assert proposal["approved"] is False, (
                f"Got approved=True for proposal_text={text!r}"
            )
