"""
File change applier for PocketCoder.

Handles applying SEARCH/REPLACE edits to files with safety checks.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import unified_diff, SequenceMatcher
from pathlib import Path

from pocketcoder.core.models import Edit, Change


def apply_edit(path: Path, edit: Edit) -> tuple[bool, str | None]:
    """
    Apply a single edit to a file.

    Args:
        path: Path to file
        edit: Edit to apply

    Returns:
        Tuple of (success, error_message)
    """
    # Handle new file
    if edit.is_new_file:
        if path.exists():
            return False, f"File already exists: {path}"

        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(edit.replace)
        return True, None

    # Read current content
    try:
        current = path.read_text()
    except FileNotFoundError:
        return False, f"File not found: {path}"
    except PermissionError:
        return False, f"Permission denied: {path}"

    # Handle delete file
    if edit.is_delete_file:
        if edit.search == current.strip():
            path.unlink()
            return True, None
        else:
            return False, "SEARCH doesn't match entire file content"

    # Find and replace
    pos = current.find(edit.search)
    if pos == -1:
        return False, "SEARCH block not found in file"

    # Check for multiple matches
    if current.count(edit.search) > 1:
        return False, f"SEARCH block found {current.count(edit.search)} times - please provide more context to make it unique"

    # Apply replacement
    new_content = current[:pos] + edit.replace + current[pos + len(edit.search):]

    # Write back
    try:
        path.write_text(new_content)
    except PermissionError:
        return False, f"Permission denied: {path}"

    return True, None


def apply_edits_to_content(content: str, edits: list[Edit]) -> str:
    """
    Apply multiple edits to content string (preview mode).

    Args:
        content: Original file content
        edits: List of edits to apply

    Returns:
        Modified content
    """
    result = content

    # Find positions (from end to start to preserve indices)
    positions = []
    for edit in edits:
        pos = result.find(edit.search)
        if pos != -1:
            positions.append((pos, edit))

    # Sort by position descending
    positions.sort(key=lambda x: x[0], reverse=True)

    # Apply edits
    for pos, edit in positions:
        result = result[:pos] + edit.replace + result[pos + len(edit.search):]

    return result


def group_edits_by_file(edits: list[Edit]) -> dict[str, list[Edit]]:
    """
    Group edits by filename.

    Args:
        edits: List of edits

    Returns:
        Dict of filename -> list of edits
    """
    by_file: dict[str, list[Edit]] = defaultdict(list)
    for edit in edits:
        by_file[edit.filename].append(edit)
    return dict(by_file)


def generate_diff(filename: str, old: str, new: str) -> str:
    """
    Generate unified diff between old and new content.

    Args:
        filename: File name for diff header
        old: Original content
        new: Modified content

    Returns:
        Unified diff string
    """
    diff_lines = list(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )
    return "".join(diff_lines)


def find_search_in_content(
    search: str, content: str
) -> tuple[int, str, float]:
    """
    Find SEARCH block in content with fuzzy matching.

    Args:
        search: Text to find
        content: File content

    Returns:
        Tuple of (position, match_type, confidence)
        - position: -1 if not found
        - match_type: "exact" | "whitespace" | "fuzzy" | "not_found"
        - confidence: 0.0 - 1.0
    """
    # Strategy 1: Exact match
    pos = content.find(search)
    if pos != -1:
        return pos, "exact", 1.0

    # Strategy 2: Whitespace normalized
    def normalize(s: str) -> str:
        return " ".join(s.split())

    norm_search = normalize(search)
    norm_content = normalize(content)

    if norm_search in norm_content:
        # Find actual position (approximate)
        # This is tricky because whitespace was normalized
        search_lines = search.splitlines()
        content_lines = content.splitlines()

        for i in range(len(content_lines) - len(search_lines) + 1):
            candidate = "\n".join(content_lines[i : i + len(search_lines)])
            if normalize(candidate) == norm_search:
                # Found position
                pos = content.find(content_lines[i])
                return pos, "whitespace", 0.95

    # Strategy 3: Fuzzy match
    search_lines = search.splitlines()
    content_lines = content.splitlines()

    best_ratio = 0.0
    best_start = -1

    for i in range(len(content_lines) - len(search_lines) + 1):
        candidate = "\n".join(content_lines[i : i + len(search_lines)])
        ratio = SequenceMatcher(None, search, candidate).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio >= 0.8:  # 80% threshold
        pos = sum(len(line) + 1 for line in content_lines[:best_start])
        return pos, "fuzzy", best_ratio

    return -1, "not_found", 0.0


def find_similar_lines(search: str, content: str, n: int = 3) -> list[tuple[int, str, float]]:
    """
    Find lines in content similar to search text.

    Args:
        search: Text to find
        content: File content
        n: Number of results to return

    Returns:
        List of (line_number, line_text, similarity_score)
    """
    search_first_line = search.splitlines()[0] if search else ""
    results = []

    for i, line in enumerate(content.splitlines(), 1):
        ratio = SequenceMatcher(None, search_first_line, line).ratio()
        if ratio > 0.5:
            results.append((i, line, ratio))

    # Sort by similarity descending
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:n]


class ChangeTracker:
    """
    Tracks file changes for undo functionality.
    """

    def __init__(self):
        self.changes: list[Change] = []

    def record(self, file: Path, old: str, new: str):
        """Record a file change."""
        self.changes.append(Change(file=file, old_content=old, new_content=new))

    def undo_last(self) -> tuple[bool, str]:
        """
        Undo last change.

        Returns:
            Tuple of (success, message)
        """
        if not self.changes:
            return False, "Nothing to undo"

        change = self.changes.pop()

        try:
            change.file.write_text(change.old_content)
            return True, f"Reverted {change.file.name}"
        except Exception as e:
            # Restore to history
            self.changes.append(change)
            return False, f"Failed to undo: {e}"

    def undo_all(self) -> int:
        """
        Undo all changes.

        Returns:
            Number of changes reverted
        """
        count = 0
        while self.changes:
            ok, _ = self.undo_last()
            if ok:
                count += 1
        return count

    def clear(self):
        """Clear change history."""
        self.changes.clear()
