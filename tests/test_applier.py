"""
Tests for file change applier.
"""

import pytest
from pathlib import Path

from pocketcoder.core.models import Edit
from pocketcoder.core.applier import (
    apply_edit,
    apply_edits_to_content,
    group_edits_by_file,
    generate_diff,
    find_search_in_content,
    find_similar_lines,
    ChangeTracker,
)


class TestApplyEdit:
    """Tests for applying edits to files."""

    def test_simple_edit(self, tmp_path):
        file = tmp_path / "test.py"
        file.write_text("def hello():\n    pass\n")

        edit = Edit(
            filename=str(file),
            search="def hello():\n    pass",
            replace="def hello():\n    print('hi')",
        )

        success, error = apply_edit(file, edit)

        assert success is True
        assert error is None
        assert file.read_text() == "def hello():\n    print('hi')\n"

    def test_edit_not_found(self, tmp_path):
        file = tmp_path / "test.py"
        file.write_text("def other():\n    pass\n")

        edit = Edit(
            filename=str(file),
            search="def hello():",
            replace="def hello(): pass",
        )

        success, error = apply_edit(file, edit)

        assert success is False
        assert "not found" in error.lower()

    def test_new_file(self, tmp_path):
        file = tmp_path / "new_file.py"

        edit = Edit(
            filename=str(file),
            search="",
            replace="def new_func():\n    pass",
        )

        success, error = apply_edit(file, edit)

        assert success is True
        assert file.exists()
        assert "new_func" in file.read_text()

    def test_new_file_already_exists(self, tmp_path):
        file = tmp_path / "existing.py"
        file.write_text("# existing content")

        edit = Edit(
            filename=str(file),
            search="",
            replace="# new content",
        )

        success, error = apply_edit(file, edit)

        assert success is False
        assert "exists" in error.lower()

    def test_multiple_matches(self, tmp_path):
        file = tmp_path / "test.py"
        file.write_text("pass\npass\npass\n")

        edit = Edit(
            filename=str(file),
            search="pass",
            replace="continue",
        )

        success, error = apply_edit(file, edit)

        assert success is False
        assert "3 times" in error or "multiple" in error.lower()


class TestApplyEditsToContent:
    """Tests for preview mode editing."""

    def test_single_edit(self):
        content = "old code here"
        edits = [Edit(filename="test.py", search="old", replace="new")]

        result = apply_edits_to_content(content, edits)

        assert result == "new code here"

    def test_multiple_edits(self):
        content = "first\nsecond\nthird"
        edits = [
            Edit(filename="test.py", search="first", replace="ONE"),
            Edit(filename="test.py", search="third", replace="THREE"),
        ]

        result = apply_edits_to_content(content, edits)

        assert "ONE" in result
        assert "THREE" in result
        assert "second" in result


class TestGroupEditsByFile:
    """Tests for edit grouping."""

    def test_group_edits(self):
        edits = [
            Edit(filename="a.py", search="1", replace="1"),
            Edit(filename="b.py", search="2", replace="2"),
            Edit(filename="a.py", search="3", replace="3"),
        ]

        grouped = group_edits_by_file(edits)

        assert len(grouped) == 2
        assert len(grouped["a.py"]) == 2
        assert len(grouped["b.py"]) == 1


class TestGenerateDiff:
    """Tests for diff generation."""

    def test_simple_diff(self):
        old = "line1\nold line\nline3"
        new = "line1\nnew line\nline3"

        diff = generate_diff("test.py", old, new)

        assert "-old line" in diff
        assert "+new line" in diff

    def test_no_changes(self):
        content = "same content"
        diff = generate_diff("test.py", content, content)

        # Empty or minimal diff
        assert "-" not in diff or "++" in diff  # Only headers


class TestFindSearchInContent:
    """Tests for search matching."""

    def test_exact_match(self):
        content = "def hello():\n    pass"
        search = "def hello():\n    pass"

        pos, match_type, confidence = find_search_in_content(search, content)

        assert pos == 0
        assert match_type == "exact"
        assert confidence == 1.0

    def test_not_found(self):
        content = "def hello():\n    pass"
        search = "def goodbye():\n    pass"

        pos, match_type, confidence = find_search_in_content(search, content)

        assert pos == -1
        assert match_type == "not_found"
        assert confidence == 0.0

    def test_fuzzy_match(self):
        content = "def hello():\n    pass"
        search = "def hello():\n   pass"  # Slightly different whitespace

        pos, match_type, confidence = find_search_in_content(search, content)

        # Should find with whitespace or fuzzy
        assert match_type in ("whitespace", "fuzzy", "exact")


class TestFindSimilarLines:
    """Tests for similar line finding."""

    def test_find_similar(self):
        content = "def hello():\n    pass\ndef world():\n    pass"
        search = "def hello():\n    return"

        similar = find_similar_lines(search, content, n=2)

        assert len(similar) > 0
        # First result should be "def hello():" line
        assert "hello" in similar[0][1]


class TestChangeTracker:
    """Tests for undo functionality."""

    def test_record_and_undo(self, tmp_path):
        file = tmp_path / "test.py"
        file.write_text("original")

        tracker = ChangeTracker()
        tracker.record(file, "original", "modified")

        file.write_text("modified")

        # Undo
        success, msg = tracker.undo_last()

        assert success is True
        assert file.read_text() == "original"

    def test_undo_nothing(self):
        tracker = ChangeTracker()
        success, msg = tracker.undo_last()

        assert success is False
        assert "nothing" in msg.lower()

    def test_undo_all(self, tmp_path):
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"

        file1.write_text("original1")
        file2.write_text("original2")

        tracker = ChangeTracker()
        tracker.record(file1, "original1", "modified1")
        tracker.record(file2, "original2", "modified2")

        file1.write_text("modified1")
        file2.write_text("modified2")

        count = tracker.undo_all()

        assert count == 2
        assert file1.read_text() == "original1"
        assert file2.read_text() == "original2"
