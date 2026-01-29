"""
Episodic Memory for PocketCoder v2.3.0.

Episodes = chain of summaries from request to request.
Each user request creates one Episode that accumulates:
- What was asked (user_input)
- What was done (outcome)
- Artifacts (files created, commands run)
- Remaining tasks (for checkpoint)

Storage: .pocketcoder/episodes.jsonl (append-only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class Episode:
    """
    One user request → outcome cycle.

    Lifecycle:
    1. User sends message → Episode created (status="in_progress")
    2. LLM executes tools → Episode accumulates actions
    3. attempt_completion or next user message → Episode closed
    """

    id: int                              # Sequential ID (1, 2, 3...)
    user_input: str                      # What user asked (max 200 chars)
    status: str                          # "in_progress" | "completed" | "checkpoint" | "meta"

    # Outcome (built during execution)
    outcome: str = ""                    # Summary of what was done (max 300 chars)

    # Artifacts (important for small models - they forget paths!)
    files_created: List[str] = field(default_factory=list)   # Full paths
    files_modified: List[str] = field(default_factory=list)  # Full paths
    commands_run: List[str] = field(default_factory=list)    # Commands executed

    # For checkpoint/continuation
    remaining: List[str] = field(default_factory=list)       # What's left to do
    todo_snapshot: List[Dict] = field(default_factory=list)  # Current TODO state

    # Metadata
    timestamp: str = ""                  # ISO format
    tokens_used: int = 0                 # Approximate tokens in this episode

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_xml(self) -> str:
        """
        Format for injection into SESSION_CONTEXT.

        Returns XML like:
        <turn id="1" status="completed">
          <user>create calculator</user>
          <outcome>Created calc.py with add/sub functions</outcome>
          <artifacts>
            <created>calc.py, test_calc.py</created>
            <commands>python calc.py</commands>
          </artifacts>
        </turn>
        """
        lines = [f'<turn id="{self.id}" status="{self.status}">']

        # User input (truncate if needed)
        user_text = self.user_input[:200]
        if len(self.user_input) > 200:
            user_text += "..."
        lines.append(f'  <user>{self._escape_xml(user_text)}</user>')

        # Outcome
        if self.outcome:
            lines.append(f'  <outcome>{self._escape_xml(self.outcome)}</outcome>')

        # Artifacts (files and commands)
        has_artifacts = self.files_created or self.files_modified or self.commands_run
        if has_artifacts:
            lines.append('  <artifacts>')

            if self.files_created:
                files = ", ".join(self.files_created[:10])
                if len(self.files_created) > 10:
                    files += f", ... (+{len(self.files_created) - 10} more)"
                lines.append(f'    <created>{files}</created>')

            if self.files_modified:
                files = ", ".join(self.files_modified[:10])
                if len(self.files_modified) > 10:
                    files += f", ... (+{len(self.files_modified) - 10} more)"
                lines.append(f'    <modified>{files}</modified>')

            if self.commands_run:
                cmds = ", ".join(self.commands_run[:5])
                if len(self.commands_run) > 5:
                    cmds += f", ... (+{len(self.commands_run) - 5} more)"
                lines.append(f'    <commands>{cmds}</commands>')

            lines.append('  </artifacts>')

        # Remaining tasks (for checkpoint episodes)
        if self.remaining:
            lines.append('  <remaining>')
            for r in self.remaining[:5]:
                lines.append(f'    - {self._escape_xml(r)}')
            if len(self.remaining) > 5:
                lines.append(f'    - ... (+{len(self.remaining) - 5} more)')
            lines.append('  </remaining>')

        lines.append('</turn>')
        return "\n".join(lines)

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def estimate_tokens(self) -> int:
        """Estimate tokens used by this episode."""
        tokens = 0
        tokens += len(self.user_input) // 4
        tokens += len(self.outcome) // 4
        tokens += len(self.files_created) * 20  # ~20 tokens per path
        tokens += len(self.files_modified) * 20
        tokens += len(self.commands_run) * 15   # ~15 tokens per command
        tokens += len(self.remaining) * 10
        return tokens


class EpisodeManager:
    """
    Manages episode lifecycle and storage.

    Usage:
        em = EpisodeManager(Path(".pocketcoder"))

        # Start new episode
        ep = em.start_episode("create calculator")

        # Track actions during execution
        em.add_action("write_file", {"path": "calc.py"}, "Created")
        em.add_action("execute_command", {"cmd": "python calc.py"}, "OK")

        # Close episode
        em.close_episode(todo_snapshot=[...])

        # Get history for SESSION_CONTEXT
        xml = em.build_history_xml(max_episodes=20)
    """

    EPISODES_FILE = "episodes.jsonl"

    def __init__(self, project_dir: Path):
        """
        Initialize episode manager.

        Args:
            project_dir: Path to .pocketcoder directory
        """
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.project_dir / self.EPISODES_FILE

        self.current: Optional[Episode] = None
        self._next_id = self._get_next_id()

    def _get_next_id(self) -> int:
        """Get next episode ID from file."""
        if not self.episodes_file.exists():
            return 1

        max_id = 0
        with open(self.episodes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        ep_id = data.get("id", 0)
                        if ep_id > max_id:
                            max_id = ep_id
                    except json.JSONDecodeError:
                        continue

        return max_id + 1

    def start_episode(self, user_input: str) -> Episode:
        """
        Start new episode when user sends message.

        Automatically closes previous episode if still in_progress.

        Args:
            user_input: User's message

        Returns:
            New Episode object
        """
        # Close previous if exists and still in progress
        if self.current and self.current.status == "in_progress":
            self.close_episode()

        self.current = Episode(
            id=self._next_id,
            user_input=user_input[:500],  # Store more for context
            status="in_progress",
            outcome="",
            files_created=[],
            files_modified=[],
            commands_run=[],
            remaining=[],
            todo_snapshot=[],
            timestamp=datetime.now().isoformat(),
            tokens_used=0
        )
        self._next_id += 1

        return self.current

    def add_action(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        """
        Track action during episode execution.

        Called after each tool execution to accumulate artifacts.

        Args:
            tool_name: Name of tool executed
            args: Tool arguments
            result: Tool result
        """
        if not self.current:
            return

        if tool_name == "write_file":
            path = args.get("path", "")
            if path and path not in self.current.files_created:
                # Check if file existed (would be modify, not create)
                if Path(path).exists():
                    if path not in self.current.files_modified:
                        self.current.files_modified.append(path)
                else:
                    self.current.files_created.append(path)

        elif tool_name == "execute_command":
            cmd = args.get("cmd", "")[:100]  # Truncate long commands
            if cmd and cmd not in self.current.commands_run:
                self.current.commands_run.append(cmd)

        elif tool_name == "attempt_completion":
            # LLM provided summary - use it as outcome
            self.current.outcome = args.get("result", "")[:500]

        elif tool_name == "checkpoint_progress":
            # Checkpoint - save done/remaining
            self.current.outcome = args.get("done", "")[:500]
            remaining_str = args.get("remaining", "")
            if remaining_str:
                self.current.remaining = [
                    r.strip() for r in remaining_str.split("\n")
                    if r.strip()
                ][:10]
            self.current.status = "checkpoint"

        # Update token estimate
        self.current.tokens_used = self.current.estimate_tokens()

    def close_episode(self, todo_snapshot: List[Dict] = None) -> None:
        """
        Close current episode and save to file.

        Args:
            todo_snapshot: Current TODO state to save
        """
        if not self.current:
            return

        # If no outcome from attempt_completion, build from actions
        if not self.current.outcome:
            parts = []
            if self.current.files_created:
                files = ", ".join(self.current.files_created[:3])
                if len(self.current.files_created) > 3:
                    files += f" (+{len(self.current.files_created) - 3} more)"
                parts.append(f"Created: {files}")

            if self.current.files_modified:
                files = ", ".join(self.current.files_modified[:3])
                parts.append(f"Modified: {files}")

            if self.current.commands_run:
                cmds = ", ".join(self.current.commands_run[:2])
                parts.append(f"Ran: {cmds}")

            self.current.outcome = "; ".join(parts) if parts else "No actions completed"

        # Set final status
        if self.current.status == "in_progress":
            self.current.status = "completed"

        # Save TODO snapshot
        if todo_snapshot:
            self.current.todo_snapshot = todo_snapshot

        # Update final token count
        self.current.tokens_used = self.current.estimate_tokens()

        # Append to file (not overwrite!)
        with open(self.episodes_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(self.current), ensure_ascii=False) + "\n")

        self.current = None

    def load_all(self) -> List[Episode]:
        """
        Load all episodes from file.

        Returns:
            List of Episode objects, oldest first
        """
        if not self.episodes_file.exists():
            return []

        episodes = []
        with open(self.episodes_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        episodes.append(Episode(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue

        return episodes

    def build_history_xml(self, max_episodes: int = 20) -> str:
        """
        Build <conversation_history> XML for SESSION_CONTEXT injection.

        Args:
            max_episodes: Maximum episodes to include

        Returns:
            XML string for injection
        """
        episodes = self.load_all()

        if not episodes:
            return "<conversation_history/>"

        # Take last N episodes
        recent = episodes[-max_episodes:]

        lines = [
            "<!-- Previous user requests and their outcomes (newest last) -->",
            "<conversation_history>"
        ]

        for ep in recent:
            lines.append(ep.to_xml())

        lines.append("</conversation_history>")

        return "\n".join(lines)

    def should_checkpoint(self, threshold: int = 2500) -> bool:
        """
        Check if current episode needs checkpoint.

        Args:
            threshold: Token threshold for checkpoint

        Returns:
            True if episode is too large
        """
        if not self.current:
            return False

        return self.current.estimate_tokens() > threshold

    def clear(self) -> None:
        """Clear all episodes (for /clear command)."""
        if self.episodes_file.exists():
            self.episodes_file.unlink()
        self._next_id = 1
        self.current = None

    def compact(self, keep_recent: int = 3) -> tuple[int, str]:
        """
        Prepare episodes for compaction.

        Returns old episodes that should be summarized by LLM.

        Args:
            keep_recent: Number of recent episodes to keep intact

        Returns:
            Tuple of (count of old episodes, their XML for summarization)
        """
        episodes = self.load_all()

        if len(episodes) <= keep_recent:
            return 0, ""

        old_episodes = episodes[:-keep_recent]

        # Build XML for LLM to summarize
        xml_parts = []
        for ep in old_episodes:
            xml_parts.append(ep.to_xml())

        return len(old_episodes), "\n".join(xml_parts)

    def save_compacted(
        self,
        meta_summary: str,
        old_files: List[str],
        keep_recent: int = 3
    ) -> None:
        """
        Save compacted episodes - meta summary + recent episodes.

        Args:
            meta_summary: LLM-generated summary of old episodes
            old_files: Aggregated files from old episodes
            keep_recent: Number of recent episodes to keep
        """
        episodes = self.load_all()
        recent = episodes[-keep_recent:] if len(episodes) > keep_recent else episodes

        # Create meta episode
        meta_episode = Episode(
            id=0,  # Special ID for meta
            user_input="[META] Previous work summary",
            status="meta",
            outcome=meta_summary[:500],
            files_created=old_files[:20],
            files_modified=[],
            commands_run=[],
            remaining=[],
            todo_snapshot=[],
            timestamp=datetime.now().isoformat(),
            tokens_used=0
        )

        # Rewrite file with meta + recent
        with open(self.episodes_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(meta_episode), ensure_ascii=False) + "\n")
            for ep in recent:
                f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")

        # Update next_id
        self._next_id = max(ep.id for ep in recent) + 1 if recent else 1

    def get_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        episodes = self.load_all()

        if not episodes:
            return {
                "count": 0,
                "total_tokens": 0,
                "files_created": 0,
                "commands_run": 0
            }

        return {
            "count": len(episodes),
            "total_tokens": sum(ep.tokens_used for ep in episodes),
            "files_created": sum(len(ep.files_created) for ep in episodes),
            "commands_run": sum(len(ep.commands_run) for ep in episodes),
            "statuses": {
                "completed": sum(1 for ep in episodes if ep.status == "completed"),
                "checkpoint": sum(1 for ep in episodes if ep.status == "checkpoint"),
                "meta": sum(1 for ep in episodes if ep.status == "meta"),
            }
        }
