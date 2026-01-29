"""
Advanced edit tools for PocketCoder.

Tools for applying diffs and batch edits:
- apply_diff: Apply unified diff to file
- multi_edit: Apply multiple SEARCH/REPLACE blocks
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple


class DiffHunk(NamedTuple):
    """Represents a diff hunk."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


def apply_diff(
    path: str,
    diff: str,
    base: Path | None = None
) -> tuple[bool, str]:
    """
    Apply unified diff to file.

    Use when:
    - Large refactoring with many changes
    - Standard diff format from git or other tools
    - Atomic application (all or nothing)

    Args:
        path: File path to apply diff to
        diff: Unified diff content

    Returns:
        Tuple of (success, result_message)

    Diff format:
        --- a/file.py
        +++ b/file.py
        @@ -10,5 +10,6 @@
         context line
        -removed line
        +added line
         context line
    """
    if base is None:
        base = Path.cwd()

    target = Path(path)
    if not target.is_absolute():
        target = (base / path).resolve()

    if not target.exists():
        return False, f"File not found: {path}"

    # Parse diff
    hunks = _parse_unified_diff(diff)

    if not hunks:
        return False, "No valid diff hunks found"

    # Read current content
    try:
        original = target.read_text()
        lines = original.splitlines(keepends=True)
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Apply hunks in reverse order (to preserve line numbers)
    try:
        for hunk in reversed(hunks):
            lines = _apply_hunk(lines, hunk)
    except Exception as e:
        return False, f"Failed to apply diff: {e}"

    # Write result
    try:
        new_content = "".join(lines)
        target.write_text(new_content)

        return True, f"Applied {len(hunks)} hunk(s) to {path}"

    except Exception as e:
        return False, f"Failed to write file: {e}"


def _parse_unified_diff(diff: str) -> list[DiffHunk]:
    """Parse unified diff into hunks."""
    hunks = []

    # Pattern for hunk header: @@ -10,5 +10,6 @@
    hunk_pattern = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    lines = diff.splitlines()
    i = 0

    while i < len(lines):
        match = hunk_pattern.match(lines[i])
        if match:
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            new_start = int(match.group(3))
            new_count = int(match.group(4) or 1)

            # Collect hunk lines
            hunk_lines = []
            i += 1
            while i < len(lines):
                line = lines[i]
                if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                    break
                if line.startswith(("+", "-", " ")):
                    hunk_lines.append(line)
                elif line == "":
                    hunk_lines.append(" ")  # Empty context line
                i += 1

            hunks.append(DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=hunk_lines,
            ))
        else:
            i += 1

    return hunks


def _apply_hunk(lines: list[str], hunk: DiffHunk) -> list[str]:
    """Apply single hunk to lines."""
    # Convert to 0-indexed
    start = hunk.old_start - 1

    # Collect additions and verify removals
    additions = []
    removals = 0

    for line in hunk.lines:
        if line.startswith("+"):
            additions.append(line[1:] + "\n")
        elif line.startswith("-"):
            removals += 1
        elif line.startswith(" "):
            additions.append(line[1:] + "\n")

    # Remove old lines and insert new
    result = lines[:start] + additions + lines[start + removals:]

    return result


def multi_edit(
    edits: list[dict],
    base: Path | None = None
) -> tuple[bool, str]:
    """
    Apply multiple SEARCH/REPLACE edits to multiple files.

    Use when:
    - Renaming across multiple files
    - Batch refactoring
    - Multiple related changes

    Args:
        edits: List of edit dicts with keys:
            - path: File path
            - search: Text to find
            - replace: Text to replace with

    Returns:
        Tuple of (success, result_summary)

    Example:
        multi_edit([
            {"path": "a.py", "search": "old_name", "replace": "new_name"},
            {"path": "b.py", "search": "old_name", "replace": "new_name"},
        ])
    """
    if base is None:
        base = Path.cwd()

    if not edits:
        return False, "No edits provided"

    results = []
    success_count = 0
    fail_count = 0

    for edit in edits:
        # Validate edit
        if not all(k in edit for k in ("path", "search", "replace")):
            results.append(f"❌ Invalid edit: missing required keys")
            fail_count += 1
            continue

        path = edit["path"]
        search = edit["search"]
        replace = edit["replace"]

        target = Path(path)
        if not target.is_absolute():
            target = (base / path).resolve()

        if not target.exists():
            results.append(f"❌ {path}: file not found")
            fail_count += 1
            continue

        # Read and apply
        try:
            content = target.read_text()

            if search not in content:
                # Try fuzzy match
                match_result = _fuzzy_find(content, search)
                if match_result:
                    actual_search, similarity = match_result
                    content = content.replace(actual_search, replace, 1)
                    results.append(f"✓ {path}: applied (fuzzy {similarity:.0%})")
                    success_count += 1
                else:
                    results.append(f"⚠️ {path}: search text not found")
                    fail_count += 1
                continue

            # Exact match
            new_content = content.replace(search, replace, 1)
            target.write_text(new_content)
            results.append(f"✓ {path}: applied")
            success_count += 1

        except Exception as e:
            results.append(f"❌ {path}: {e}")
            fail_count += 1

    # Summary
    summary = f"Multi-edit: {success_count} succeeded, {fail_count} failed\n\n"
    summary += "\n".join(results)

    overall_success = fail_count == 0
    return overall_success, summary


def _fuzzy_find(content: str, search: str, threshold: float = 0.8) -> tuple[str, float] | None:
    """
    Find fuzzy match for search text in content.

    Returns (matched_text, similarity) or None if no good match.
    """
    import difflib

    # Split content into chunks of similar length to search
    search_len = len(search)
    search_lines = search.count("\n") + 1

    # Try line-based matching for multi-line search
    if "\n" in search:
        content_lines = content.splitlines(keepends=True)
        search_clean = search.strip()

        for i in range(len(content_lines) - search_lines + 1):
            chunk = "".join(content_lines[i:i + search_lines]).strip()
            ratio = difflib.SequenceMatcher(None, search_clean, chunk).ratio()
            if ratio >= threshold:
                return "".join(content_lines[i:i + search_lines]), ratio

    # Single line - slide window
    else:
        for i in range(len(content) - search_len + 1):
            chunk = content[i:i + search_len]
            ratio = difflib.SequenceMatcher(None, search, chunk).ratio()
            if ratio >= threshold:
                return chunk, ratio

    return None


def insert_at_line(
    path: str,
    line_number: int,
    content: str,
    base: Path | None = None
) -> tuple[bool, str]:
    """
    Insert content at specific line number.

    Use when:
    - Know exact line number to insert at
    - Adding imports at top
    - Adding code at specific location

    Args:
        path: File path
        line_number: Line number to insert at (1-indexed)
        content: Content to insert

    Returns:
        Tuple of (success, result_message)
    """
    if base is None:
        base = Path.cwd()

    target = Path(path)
    if not target.is_absolute():
        target = (base / path).resolve()

    if not target.exists():
        return False, f"File not found: {path}"

    try:
        lines = target.read_text().splitlines(keepends=True)

        # Validate line number
        if line_number < 1:
            line_number = 1
        if line_number > len(lines) + 1:
            line_number = len(lines) + 1

        # Ensure content ends with newline
        if not content.endswith("\n"):
            content += "\n"

        # Insert
        idx = line_number - 1
        lines.insert(idx, content)

        target.write_text("".join(lines))

        return True, f"Inserted at line {line_number} in {path}"

    except Exception as e:
        return False, f"Failed to insert: {e}"


def delete_lines(
    path: str,
    start_line: int,
    end_line: int,
    base: Path | None = None
) -> tuple[bool, str]:
    """
    Delete lines from file.

    Use when:
    - Removing specific lines
    - Cleaning up code

    Args:
        path: File path
        start_line: First line to delete (1-indexed, inclusive)
        end_line: Last line to delete (1-indexed, inclusive)

    Returns:
        Tuple of (success, result_message)
    """
    if base is None:
        base = Path.cwd()

    target = Path(path)
    if not target.is_absolute():
        target = (base / path).resolve()

    if not target.exists():
        return False, f"File not found: {path}"

    try:
        lines = target.read_text().splitlines(keepends=True)

        # Validate range
        total_lines = len(lines)
        if total_lines == 0:
            return False, f"File {path} is empty, nothing to delete"
        if start_line < 1:
            start_line = 1
        if start_line > total_lines:
            return False, f"File {path} has only {total_lines} line(s), cannot delete from line {start_line}"
        if end_line > total_lines:
            end_line = total_lines
        if start_line > end_line:
            return False, f"Invalid range: {start_line}-{end_line} (file has {total_lines} lines)"

        # Delete
        del lines[start_line - 1:end_line]

        target.write_text("".join(lines))

        count = end_line - start_line + 1
        return True, f"Deleted {count} line(s) from {path}"

    except Exception as e:
        return False, f"Failed to delete: {e}"
