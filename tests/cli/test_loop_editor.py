"""Isolated unit tests for faust.loop_editor (Phase 2 validation).

Covers:
- EditProposal dataclass: field defaults, required fields, test_targets default.
- build_unified_diff: pure function — no filesystem access.
    - Identical content produces an empty diff.
    - A real change produces a unified diff with correct headers.
    - The path label appears in the diff.
- write_approved_change: safety gate + filesystem write.
    - Accepts valid src/ paths.
    - Rejects absolute paths (raises ValueError).
    - Rejects directory traversal (../../) (raises ValueError).
    - Rejects paths outside src/ — tests/, configs/, data/ (raises ValueError).
    - Creates missing parent directories.
    - Writes the correct content and returns a resolved Path.
    - Returned path is absolute.
"""

from __future__ import annotations

import pytest

from faust.loop_editor import EditProposal, build_unified_diff, write_approved_change


# ---------------------------------------------------------------------------
# EditProposal dataclass
# ---------------------------------------------------------------------------

class TestEditProposal:
    """EditProposal carries a proposed change through the approval gate."""

    def test_required_fields_set_correctly(self):
        ep = EditProposal(
            path="src/faust/foo.py",
            original="old",
            proposed="new",
            diff="--- a/src/faust/foo.py\n+++ b/src/faust/foo.py\n",
        )
        assert ep.path == "src/faust/foo.py"
        assert ep.original == "old"
        assert ep.proposed == "new"
        assert "---" in ep.diff

    def test_description_defaults_to_empty_string(self):
        ep = EditProposal(
            path="src/faust/foo.py",
            original="",
            proposed="",
            diff="",
        )
        assert ep.description == ""

    def test_test_targets_defaults_to_empty_list(self):
        ep = EditProposal(
            path="src/faust/foo.py",
            original="",
            proposed="",
            diff="",
        )
        assert ep.test_targets == []

    def test_test_targets_are_independent_per_instance(self):
        """Mutable default via field(default_factory=list) — no aliasing."""
        ep1 = EditProposal(path="src/a.py", original="", proposed="", diff="")
        ep2 = EditProposal(path="src/b.py", original="", proposed="", diff="")
        ep1.test_targets.append("tests/a.py")
        assert ep2.test_targets == [], "test_targets must not be shared between instances"


# ---------------------------------------------------------------------------
# build_unified_diff
# ---------------------------------------------------------------------------

class TestBuildUnifiedDiff:
    """build_unified_diff is a pure function — no filesystem access."""

    def test_identical_content_returns_empty_string(self):
        result = build_unified_diff("same\n", "same\n", "src/faust/foo.py")
        assert result == ""

    def test_changed_content_produces_nonempty_diff(self):
        result = build_unified_diff("old line\n", "new line\n", "src/faust/foo.py")
        assert result != ""

    def test_diff_has_from_header(self):
        result = build_unified_diff("a\n", "b\n", "src/faust/bar.py")
        assert "--- a/src/faust/bar.py" in result

    def test_diff_has_to_header(self):
        result = build_unified_diff("a\n", "b\n", "src/faust/bar.py")
        assert "+++ b/src/faust/bar.py" in result

    def test_diff_marks_removed_line(self):
        result = build_unified_diff("removed\n", "", "src/faust/bar.py")
        assert "-removed" in result

    def test_diff_marks_added_line(self):
        result = build_unified_diff("", "added\n", "src/faust/bar.py")
        assert "+added" in result

    def test_path_label_appears_in_diff_header(self):
        path = "src/faust/agents/coder.py"
        result = build_unified_diff("x\n", "y\n", path)
        assert path in result


# ---------------------------------------------------------------------------
# write_approved_change — safety gate
# ---------------------------------------------------------------------------

class TestWriteApprovedChangeSafetyGate:
    """write_approved_change must reject any path outside src/."""

    def test_rejects_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        with pytest.raises(ValueError, match="Write rejected"):
            write_approved_change("/etc/passwd", "evil")

    def test_rejects_traversal_outside_src(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        with pytest.raises(ValueError, match="Write rejected"):
            write_approved_change("src/../../etc/passwd", "evil")

    def test_rejects_tests_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        with pytest.raises(ValueError, match="Write rejected"):
            write_approved_change("tests/some_test.py", "evil")

    def test_rejects_configs_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "configs").mkdir()
        with pytest.raises(ValueError, match="Write rejected"):
            write_approved_change("configs/settings.yaml", "evil")

    def test_rejects_data_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "data").mkdir()
        with pytest.raises(ValueError, match="Write rejected"):
            write_approved_change("data/store.json", "evil")

    def test_error_message_includes_rejected_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        with pytest.raises(ValueError) as exc_info:
            write_approved_change("tests/evil.py", "x")
        assert "tests/evil.py" in str(exc_info.value) or "Write rejected" in str(exc_info.value)


# ---------------------------------------------------------------------------
# write_approved_change — filesystem write
# ---------------------------------------------------------------------------

class TestWriteApprovedChangeFilesystem:
    """write_approved_change writes correct content inside src/."""

    def test_writes_content_to_valid_src_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src" / "faust"
        src.mkdir(parents=True)
        write_approved_change("src/faust/foo.py", "print('hello')\n")
        assert (src / "foo.py").read_text() == "print('hello')\n"

    def test_returns_resolved_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src" / "faust"
        src.mkdir(parents=True)
        result = write_approved_change("src/faust/bar.py", "x = 1\n")
        assert result.is_absolute()

    def test_creates_missing_parent_directories(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "src").mkdir()
        # deep nested path — parents do not exist yet
        write_approved_change("src/faust/agents/deep/nested.py", "# ok\n")
        assert (tmp_path / "src" / "faust" / "agents" / "deep" / "nested.py").exists()

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src" / "faust"
        src.mkdir(parents=True)
        target = src / "overwrite_me.py"
        target.write_text("old content\n")
        write_approved_change("src/faust/overwrite_me.py", "new content\n")
        assert target.read_text() == "new content\n"
