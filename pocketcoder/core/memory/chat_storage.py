"""
Chat Storage for PocketCoder v0.7.0.

Saves raw conversation to JSONL file for:
- Persistent history across sessions
- Grep-based search for context retrieval
- Debugging and analysis

Storage format: ~/.pocketcoder/sessions/{session_id}/chat_raw.jsonl
Each line: {"ts": "ISO timestamp", "role": "user|assistant|tool_result", "content": "...", ...}
"""

from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


def get_session_dir(project_path: str = ".", per_project: bool = True) -> Path:
    """
    Get session directory for current project.

    v2.0.0: Changed from global ~/.pocketcoder/sessions/{hash}/
            to per-project .pocketcoder/sessions/

    Args:
        project_path: Path to project directory
        per_project: If True, use .pocketcoder/ in project (v2.0.0 default)
                    If False, use global ~/.pocketcoder/sessions/{hash}/

    Returns:
        Path to session directory
    """
    if per_project:
        # v2.0.0: Per-project storage in .pocketcoder/
        project_root = Path(project_path).resolve()
        session_dir = project_root / ".pocketcoder" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    else:
        # Legacy: global storage with hash
        abs_path = Path(project_path).resolve()
        project_hash = hashlib.md5(str(abs_path).encode()).hexdigest()[:12]

        session_dir = Path.home() / ".pocketcoder" / "sessions" / project_hash
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir


class ChatStorage:
    """
    Stores raw chat conversation in JSONL format.

    Provides:
    - append(): Add message to log
    - grep(): Search by regex pattern
    - get_context_for(): Get relevant context by keywords
    - get_recent(): Get recent N messages
    """

    def __init__(self, session_dir: Optional[Path] = None, project_path: str = "."):
        """
        Initialize chat storage.

        Args:
            session_dir: Override session directory
            project_path: Project path for session directory resolution
        """
        if session_dir:
            self.session_dir = session_dir
        else:
            self.session_dir = get_session_dir(project_path)

        self.raw_file = self.session_dir / "chat_raw.jsonl"
        self.index_file = self.session_dir / "chat_index.json"

        # Ensure files exist
        self.raw_file.touch(exist_ok=True)

    def append(
        self,
        role: str,
        content: str,
        **meta: Any
    ) -> None:
        """
        Append message to raw chat log.

        Args:
            role: Message role (user, assistant, tool_result, system)
            content: Message content
            **meta: Additional metadata (tool_name, args, etc.)
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "role": role,
            "content": content[:10000],  # Limit content size
            **meta
        }

        with open(self.raw_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: str
    ) -> None:
        """
        Append tool call with result.

        Args:
            tool_name: Name of tool
            args: Tool arguments
            result: Tool result
        """
        self.append(
            role="tool_result",
            content=result[:5000],  # Truncate large results
            tool_name=tool_name,
            args={k: str(v)[:500] for k, v in args.items()},  # Truncate args
        )

    def grep(
        self,
        pattern: str,
        limit: int = 10,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search messages by regex pattern.

        Args:
            pattern: Regex pattern to search
            limit: Maximum results to return
            role_filter: Filter by role (user, assistant, tool_result)

        Returns:
            List of matching entries
        """
        results = []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Invalid regex - treat as literal string
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        if not self.raw_file.exists():
            return results

        with open(self.raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Role filter
                if role_filter and entry.get("role") != role_filter:
                    continue

                # Search in content
                content = entry.get("content", "")
                if regex.search(content):
                    results.append(entry)
                    if len(results) >= limit:
                        break

                # Also search in tool args (for paths)
                if entry.get("role") == "tool_result":
                    args = entry.get("args", {})
                    args_str = json.dumps(args)
                    if regex.search(args_str):
                        if entry not in results:
                            results.append(entry)
                            if len(results) >= limit:
                                break

        return results

    def get_context_for(
        self,
        keywords: List[str],
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context from chat history by keywords.

        Searches for keywords and returns formatted context.

        Args:
            keywords: List of keywords to search
            max_tokens: Maximum tokens in result (~4 chars per token)

        Returns:
            Formatted context string
        """
        if not keywords:
            return ""

        relevant = []

        # Search for each keyword
        for kw in keywords:
            matches = self.grep(kw, limit=5)
            relevant.extend(matches)

        if not relevant:
            return ""

        # Deduplicate by timestamp
        seen_ts = set()
        unique = []
        for entry in relevant:
            ts = entry.get("ts", "")
            if ts not in seen_ts:
                seen_ts.add(ts)
                unique.append(entry)

        # Sort by timestamp
        unique.sort(key=lambda x: x.get("ts", ""))

        # Format as context
        lines = []
        total_chars = 0
        max_chars = max_tokens * 4  # ~4 chars per token

        for entry in unique:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:500]  # Truncate
            ts = entry.get("ts", "")[:16]  # YYYY-MM-DDTHH:MM

            # Format based on role
            if role == "tool_result":
                tool_name = entry.get("tool_name", "tool")
                args = entry.get("args", {})
                path = args.get("path", args.get("cmd", ""))
                line = f"[{ts}] {tool_name}({path}): {content[:200]}"
            else:
                line = f"[{ts}] {role}: {content[:300]}"

            # Check token limit
            if total_chars + len(line) > max_chars:
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent N messages.

        Args:
            n: Number of messages

        Returns:
            List of recent entries (newest last)
        """
        entries = []

        if not self.raw_file.exists():
            return entries

        # Read all lines (could optimize for large files)
        with open(self.raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        return entries[-n:]

    def get_paths_mentioned(self) -> List[str]:
        """
        Extract all file/folder paths mentioned in chat.

        Useful for finding project paths when LLM forgets.

        Returns:
            List of unique paths
        """
        paths = set()

        if not self.raw_file.exists():
            return list(paths)

        with open(self.raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract from tool args
                if entry.get("role") == "tool_result":
                    args = entry.get("args", {})
                    if "path" in args:
                        paths.add(args["path"])

                # Extract from content (simple heuristic)
                content = entry.get("content", "")
                # Match paths like folder/file.py or ./folder
                path_matches = re.findall(r'[./\w]+/[\w.]+', content)
                for p in path_matches:
                    if not p.startswith("http"):  # Skip URLs
                        paths.add(p)

        return sorted(paths)

    def clear(self) -> None:
        """Clear all chat history."""
        if self.raw_file.exists():
            self.raw_file.unlink()
        self.raw_file.touch()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get chat storage statistics.

        Returns:
            Dict with message counts, file size, etc.
        """
        if not self.raw_file.exists():
            return {"messages": 0, "size_kb": 0}

        message_count = 0
        role_counts = {}

        with open(self.raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    message_count += 1
                    try:
                        entry = json.loads(line)
                        role = entry.get("role", "unknown")
                        role_counts[role] = role_counts.get(role, 0) + 1
                    except json.JSONDecodeError:
                        pass

        size_bytes = self.raw_file.stat().st_size

        return {
            "messages": message_count,
            "size_kb": round(size_bytes / 1024, 2),
            "by_role": role_counts,
            "path": str(self.raw_file),
        }


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract keywords from text for search.

    Simple heuristic: words longer than 4 chars, excluding stop words.

    Args:
        text: Input text
        max_keywords: Maximum keywords to extract

    Returns:
        List of keywords
    """
    # Common stop words to filter out
    stop_words = {
        "this", "that", "these", "those",
        "file", "files", "folder", "folders",
        "create", "make", "show", "open",
        "what", "which", "where", "when",
        "need", "want", "should", "could",
        "then", "after", "before", "first",
        "with", "from", "into", "have", "been",
        "just", "also", "some", "more", "like",
    }

    # Extract words
    words = re.findall(r'\b\w{4,}\b', text.lower())

    # Filter and deduplicate
    keywords = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen:
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= max_keywords:
                break

    return keywords
