"""
History search tool for SESSION_CONTEXT architecture.

v2.0.0: Provides deep search through conversation history.
Auto-grep runs on every request (top 5 results).
This tool is for when LLM needs MORE context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# Storage directory (per-project)
POCKETCODER_DIR = ".pocketcoder"


def search_history(
    keywords: list[str] = None,
    query: str = None,
    context_lines: int = 3,
    max_results: int = 5,
    session: str = "current"
) -> str:
    """
    Search through conversation history.

    WHEN: Need info from previous conversations, forgot what was done
    NOT: For current task context — check <conversation_history> first
    TIP: If truncated, increase context_lines or read chat_raw.jsonl directly

    Args:
        keywords: List of words to search (AND logic, any language)
        query: Alternative: regex pattern (if keywords not provided)
        context_lines: Lines around each match (default: 3, no limit)
        max_results: Maximum results to return (default: 5)
        session: Session ID or "current" or "all" (default: current)

    Returns:
        Formatted search results with context
    """
    # Support both keywords list and query string
    if keywords:
        # Convert keywords to regex pattern (AND logic)
        pattern_str = ".*".join(re.escape(kw) for kw in keywords)
        query = pattern_str
    elif not query:
        return "[x] Provide either 'keywords' (list) or 'query' (string)"

    limit = max_results  # Use new parameter name internally
    project_dir = Path.cwd() / POCKETCODER_DIR

    if not project_dir.exists():
        return "[x] No .pocketcoder directory found. No history available."

    # Determine which sessions to search
    sessions_dir = project_dir / "sessions"
    if not sessions_dir.exists():
        return "[x] No sessions directory found. No history available."

    paths = []

    if session == "all":
        paths = list(sessions_dir.glob("*/raw.jsonl"))
    elif session == "current":
        # Read current session ID
        current_file = project_dir / "current_session"
        if current_file.exists():
            current_id = current_file.read_text().strip()
            raw_path = sessions_dir / current_id / "raw.jsonl"
            if raw_path.exists():
                paths = [raw_path]
        # If no current, search all
        if not paths:
            paths = list(sessions_dir.glob("*/raw.jsonl"))
    else:
        # Specific session ID
        raw_path = sessions_dir / session / "raw.jsonl"
        if raw_path.exists():
            paths = [raw_path]

    if not paths:
        return "[x] No history files found."

    # Build search pattern
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        # If regex fails, treat as literal
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    # Search through files
    matches = []

    for raw_file in paths:
        session_id = raw_file.parent.name

        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                    content = entry.get("content", "")

                    # Check pattern match
                    if pattern.search(content):
                        # Get context lines before and after
                        ctx_before = []
                        ctx_after = []

                        # Context before
                        for j in range(max(0, i - context_lines), i):
                            try:
                                ctx_entry = json.loads(lines[j])
                                ctx_content = ctx_entry.get("content", "")[:200]
                                ctx_before.append(f"{ctx_entry.get('role', '?')}: {ctx_content}")
                            except:
                                pass

                        # Context after
                        for j in range(i + 1, min(len(lines), i + 1 + context_lines)):
                            try:
                                ctx_entry = json.loads(lines[j])
                                ctx_content = ctx_entry.get("content", "")[:200]
                                ctx_after.append(f"{ctx_entry.get('role', '?')}: {ctx_content}")
                            except:
                                pass

                        match_data = {
                            "session": session_id,
                            "timestamp": entry.get("timestamp", ""),
                            "role": entry.get("role", ""),
                            "content": content,
                            "line": i,
                            "context_before": ctx_before,
                            "context_after": ctx_after
                        }

                        # Get response if this is a user message
                        if entry.get("role") == "user" and i + 1 < len(lines):
                            try:
                                next_entry = json.loads(lines[i + 1])
                                if next_entry.get("role") == "assistant":
                                    match_data["response"] = next_entry.get("content", "")
                            except Exception:
                                pass

                        matches.append(match_data)

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            continue

    if not matches:
        search_term = keywords if keywords else query
        return f"[ok] No matches found for: {search_term}\n" \
               f"[TIP: Try different keywords or read_file('.pocketcoder/sessions/*/raw.jsonl')]"

    # Sort by timestamp (most recent first)
    matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Limit results
    total_matches = len(matches)
    matches = matches[:limit]

    # Format output
    search_term = keywords if keywords else query
    result_lines = [f"[ok] Found {total_matches} match(es) for: {search_term}\n"]

    for i, match in enumerate(matches, 1):
        result_lines.append(f"--- Match {i} (line {match.get('line', '?')}) ---")

        # Show context before
        if match.get("context_before"):
            for ctx_line in match["context_before"][-context_lines:]:
                result_lines.append(f"  | {ctx_line[:200]}")

        # Main match content
        content = match["content"]
        if len(content) > 500:
            content = content[:500] + "..."
        result_lines.append(f">>> {match['role']}: {content}")

        # Show context after
        if match.get("context_after"):
            for ctx_line in match["context_after"][:context_lines]:
                result_lines.append(f"  | {ctx_line[:200]}")

        # Show response if available
        if match.get("response"):
            response = match["response"]
            if len(response) > 300:
                response = response[:300] + "..."
            result_lines.append(f"  Response: {response}")

        result_lines.append("")

    # Add hint about truncation
    if total_matches > limit:
        result_lines.append(f"[Showed {limit} of {total_matches} matches. "
                           f"Use max_results={min(total_matches, 20)} for more]")

    result_lines.append(f"[Showed ±{context_lines} lines around matches. "
                       f"Increase context_lines for more, or read full log]")

    return "\n".join(result_lines)


def get_recent_history(n: int = 10) -> list[dict]:
    """
    Get N most recent messages from current session.

    Args:
        n: Number of messages to retrieve

    Returns:
        List of message dicts
    """
    project_dir = Path.cwd() / POCKETCODER_DIR

    if not project_dir.exists():
        return []

    # Get current session
    current_file = project_dir / "current_session"
    if not current_file.exists():
        return []

    current_id = current_file.read_text().strip()
    raw_path = project_dir / "sessions" / current_id / "raw.jsonl"

    if not raw_path.exists():
        return []

    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        messages = []
        for line in lines[-n:]:
            try:
                entry = json.loads(line)
                messages.append(entry)
            except json.JSONDecodeError:
                continue

        return messages

    except Exception:
        return []


def append_to_history(role: str, content: str, **kwargs) -> bool:
    """
    Append a message to current session's raw.jsonl.

    Args:
        role: Message role (user, assistant, tool_result)
        content: Message content
        **kwargs: Additional fields (tool_name, args, etc.)

    Returns:
        True if successful
    """
    from datetime import datetime

    project_dir = Path.cwd() / POCKETCODER_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    # Get or create current session
    current_file = project_dir / "current_session"
    if current_file.exists():
        session_id = current_file.read_text().strip()
    else:
        # Create new session
        import uuid
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        current_file.write_text(session_id)

    # Create session directory
    session_dir = project_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Build entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
        **kwargs
    }

    # Append to raw.jsonl
    raw_path = session_dir / "raw.jsonl"
    try:
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False
