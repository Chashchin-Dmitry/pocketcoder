"""
Project Context for SESSION_CONTEXT architecture.

v2.0.0: Provides persistent project state that enables "continue" to work
without keyword detection. LLM understands context from injected SESSION_CONTEXT.

v2.6.0: Added RepoMapBuilder for code structure overview.

Classes:
- ProjectContext: stores task, project identity, persists to JSON
- TaskSummarizer: summarizes user request via LLM (always short)
- FileTracker: tracks modified/read/mentioned files
- TerminalHistory: tracks command history with exit codes
- RepoMapBuilder: builds repository structure map (from repo_map.py)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pocketcoder.providers.base import BaseProvider


# Storage directory (per-project)
POCKETCODER_DIR = ".pocketcoder"


def get_project_dir() -> Path:
    """Get or create .pocketcoder directory in current working directory."""
    project_dir = Path.cwd() / POCKETCODER_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


# =============================================================================
# Task Summary
# =============================================================================

@dataclass
class TaskSummary:
    """Summarized user task."""

    summary: str                    # 2-3 sentence summary
    task_type: str                  # CREATE | EDIT | DELETE | RUN | EXPLORE | MULTI
    artifact: str                   # what we're building (calculator, bot, etc.)
    stack: str                      # technologies (Python, Flask, etc.)
    constraints: str                # limitations (no Docker, etc.)
    raw_hash: str = ""              # hash of original request for deduplication
    created_at: str = ""            # ISO timestamp


class TaskSummarizer:
    """
    Summarizes user requests via LLM.

    Always produces short summary regardless of input length.
    Raw conversation is stored separately in raw.jsonl for grep.
    """

    SUMMARIZE_PROMPT = """Analyze this user request and extract:
1. A 2-3 sentence summary of what they want
2. Task type: CREATE (new project/file), EDIT (modify existing), DELETE (remove), RUN (execute), EXPLORE (investigate), MULTI (complex multi-step)
3. Artifact: what is being created/modified (e.g., calculator, API, bot)
4. Stack: technologies mentioned or implied (e.g., Python, Flask, React)
5. Constraints: any limitations mentioned (e.g., no Docker, local only)

User request:
{request}

Respond in this exact JSON format:
{{
  "summary": "...",
  "task_type": "CREATE|EDIT|DELETE|RUN|EXPLORE|MULTI",
  "artifact": "...",
  "stack": "...",
  "constraints": "..."
}}

JSON response:"""

    def __init__(self, provider: Optional["BaseProvider"] = None):
        self.provider = provider

    def summarize(self, request: str, provider: Optional["BaseProvider"] = None) -> TaskSummary:
        """
        Summarize user request.

        Args:
            request: Raw user request (any length)
            provider: LLM provider for summarization

        Returns:
            TaskSummary with extracted fields
        """
        provider = provider or self.provider

        # Generate hash for deduplication
        raw_hash = str(hash(request))[:16]

        # If no provider, use heuristics
        if not provider:
            return self._heuristic_summarize(request, raw_hash)

        # LLM summarization
        try:
            from pocketcoder.core.models import Message

            prompt = self.SUMMARIZE_PROMPT.format(request=request[:3000])  # Limit input
            messages = [Message("user", prompt)]

            response = provider.chat(messages, max_tokens=500)

            # Parse JSON response
            json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return TaskSummary(
                    summary=data.get("summary", request[:200]),
                    task_type=data.get("task_type", "MULTI"),
                    artifact=data.get("artifact", ""),
                    stack=data.get("stack", ""),
                    constraints=data.get("constraints", ""),
                    raw_hash=raw_hash,
                    created_at=datetime.now().isoformat()
                )
        except Exception:
            pass

        # Fallback to heuristics
        return self._heuristic_summarize(request, raw_hash)

    def _heuristic_summarize(self, request: str, raw_hash: str) -> TaskSummary:
        """Extract info using regex patterns when LLM unavailable."""
        request_lower = request.lower()

        # Detect task type
        if any(w in request_lower for w in ["create", "make", "build", "write"]):
            task_type = "CREATE"
        elif any(w in request_lower for w in ["edit", "change", "modify", "fix"]):
            task_type = "EDIT"
        elif any(w in request_lower for w in ["delete", "remove", "drop"]):
            task_type = "DELETE"
        elif any(w in request_lower for w in ["run", "execute", "start", "launch"]):
            task_type = "RUN"
        elif any(w in request_lower for w in ["show", "list", "find", "search"]):
            task_type = "EXPLORE"
        else:
            task_type = "MULTI"

        # Detect stack
        stack_patterns = [
            "python", "javascript", "typescript", "react", "vue", "flask",
            "fastapi", "django", "node", "express", "html", "css"
        ]
        stack = ", ".join([s for s in stack_patterns if s in request_lower])

        # Summary is just truncated request
        summary = request[:200] + ("..." if len(request) > 200 else "")

        return TaskSummary(
            summary=summary,
            task_type=task_type,
            artifact="",
            stack=stack,
            constraints="",
            raw_hash=raw_hash,
            created_at=datetime.now().isoformat()
        )


# =============================================================================
# File Tracker
# =============================================================================

@dataclass
class TrackedFile:
    """A file tracked by FileTracker."""

    path: str
    status: str         # created | modified | read | mentioned
    lines: int = 0
    summary: str = ""   # brief description
    timestamp: str = ""


class FileTracker:
    """
    Tracks files modified, read, and mentioned during session.

    Provides data for <files> block in SESSION_CONTEXT.
    """

    def __init__(self):
        self.modified: dict[str, TrackedFile] = {}   # path -> TrackedFile
        self.read: dict[str, TrackedFile] = {}
        self.mentioned: dict[str, TrackedFile] = {}

    def track_write(self, path: str, lines: int = 0, summary: str = "",
                    on_write_callback: callable = None) -> None:
        """
        Track a file write (create or modify).

        Args:
            path: File path
            lines: Number of lines written
            summary: Brief description
            on_write_callback: Optional callback to notify about file change (e.g., invalidate repo_map)
        """
        # Check if file existed before
        existed = Path(path).exists()
        status = "modified" if existed else "created"

        self.modified[path] = TrackedFile(
            path=path,
            status=status,
            lines=lines,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )

        # v2.6.0: Notify about file change (for repo_map invalidation)
        if on_write_callback:
            on_write_callback()

    def track_read(self, path: str, summary: str = "") -> None:
        """Track a file read."""
        if not path:
            return

        if path in self.modified:
            # v2.4.0: Update preview for already modified file
            self.modified[path].summary = summary
        elif path in self.read:
            # v2.4.0: Update preview for already read file
            self.read[path].summary = summary
        else:
            # New file
            self.read[path] = TrackedFile(
                path=path,
                status="read",
                summary=summary,
                timestamp=datetime.now().isoformat()
            )

    def track_mention(self, path: str, context: str = "") -> None:
        """Track a file mentioned in conversation."""
        if path not in self.modified and path not in self.read:
            self.mentioned[path] = TrackedFile(
                path=path,
                status="mentioned",
                summary=context,
                timestamp=datetime.now().isoformat()
            )

    def extract_mentions_from_text(self, text: str) -> list[str]:
        """Extract file paths mentioned in text using regex."""
        # Common file patterns
        patterns = [
            r'[\w\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|txt|sh)',
            r'[\w\-/]+/[\w\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|txt|sh)',
        ]

        mentions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            mentions.extend(matches)

        return list(set(mentions))

    def to_xml(self) -> str:
        """Generate XML for SESSION_CONTEXT <files> block."""
        lines = ["<files>"]

        # Modified files
        if self.modified:
            lines.append("  <modified>")
            for f in self.modified.values():
                lines.append(
                    f'    <file path="{f.path}" status="{f.status}" lines="{f.lines}">'
                    f'{f.summary}</file>'
                )
            lines.append("  </modified>")

        # Read files
        if self.read:
            lines.append("  <read>")
            for f in self.read.values():
                lines.append(f'    <file path="{f.path}">{f.summary}</file>')
            lines.append("  </read>")

        # Mentioned files
        if self.mentioned:
            lines.append("  <mentioned>")
            for f in self.mentioned.values():
                lines.append(f'    <file path="{f.path}">{f.summary}</file>')
            lines.append("  </mentioned>")

        lines.append("</files>")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "modified": {k: asdict(v) for k, v in self.modified.items()},
            "read": {k: asdict(v) for k, v in self.read.items()},
            "mentioned": {k: asdict(v) for k, v in self.mentioned.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileTracker":
        """Load from dict."""
        tracker = cls()
        for path, fdata in data.get("modified", {}).items():
            tracker.modified[path] = TrackedFile(**fdata)
        for path, fdata in data.get("read", {}).items():
            tracker.read[path] = TrackedFile(**fdata)
        for path, fdata in data.get("mentioned", {}).items():
            tracker.mentioned[path] = TrackedFile(**fdata)
        return tracker


# =============================================================================
# TODO State Machine (v2.5.0)
# =============================================================================

@dataclass
class TodoTask:
    """A single TODO task."""
    text: str
    status: str = "pending"  # pending | in_progress | completed
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ValidationResult:
    """Result of task completion validation."""
    ok: bool
    reason: str = ""
    warning: str = ""


class TodoStateMachine:
    """
    TODO State Machine for v2.5.0.

    Manages task state through tools instead of parsed output.
    LLM calls add_todo/mark_done/remove_todo, system validates.

    Replaces:
    - <todo>[...]</todo> parsing
    - _merge_todo() hack
    - todo_retry_count
    """

    def __init__(self, files: "FileTracker", terminal: "TerminalHistory"):
        self.tasks: list[TodoTask] = []
        self.files = files        # for mark_done validation
        self.terminal = terminal  # for command validation

    def add(self, task: str) -> str:
        """
        Add task to plan.

        Returns:
            [ok] Added: ... or [!] error
        """
        # Check for duplicate (exact match)
        task_lower = task.lower().strip()
        for t in self.tasks:
            if t.text.lower().strip() == task_lower:
                return f"[!] Task already exists: {task}"

        # Check for similar task (by filename)
        new_filename = self._extract_filename(task)
        if new_filename:
            for t in self.tasks:
                existing_filename = self._extract_filename(t.text)
                if existing_filename and existing_filename == new_filename:
                    return f"[!] Similar task exists for {new_filename}: {t.text}"

        self.tasks.append(TodoTask(text=task, status="pending"))
        return f"[ok] Added: {task}"

    def mark_done(self, task: str) -> str:
        """
        Mark task as completed.

        Validates that action was actually performed using FileTracker/TerminalHistory.

        Returns:
            [ok] Completed: ... or [!] error
        """
        # Find task
        found = self._find_task(task)
        if not found:
            return f"[!] Task not found: {task}"

        if found.status == "completed":
            return f"[!] Task already completed: {found.text}"

        # Validate
        validation = self._validate_completion(found.text)
        if not validation.ok:
            return f"[!] Cannot complete: {validation.reason}"

        # Mark as completed
        found.status = "completed"
        found.completed_at = datetime.now().isoformat()

        if validation.warning:
            return f"[ok] Completed ({validation.warning}): {found.text}"
        return f"[ok] Completed: {found.text}"

    def remove(self, task: str) -> str:
        """
        Remove task from plan.

        Returns:
            [ok] Removed: ... or [!] error
        """
        for i, t in enumerate(self.tasks):
            if self._matches(task, t.text):
                if t.status == "completed":
                    return f"[!] Cannot remove completed task: {t.text}"
                removed = self.tasks.pop(i)
                return f"[ok] Removed: {removed.text}"

        return f"[!] Task not found: {task}"

    def start(self, task: str) -> str:
        """
        Mark task as in_progress.

        Returns:
            [ok] Started: ... or [!] error
        """
        found = self._find_task(task)
        if not found:
            return f"[!] Task not found: {task}"

        if found.status == "completed":
            return f"[!] Task already completed: {found.text}"

        # First, remove in_progress from other tasks
        for t in self.tasks:
            if t.status == "in_progress":
                t.status = "pending"

        found.status = "in_progress"
        return f"[ok] Started: {found.text}"

    def has_pending(self) -> bool:
        """Check if there are pending or in_progress tasks."""
        for t in self.tasks:
            if t.status in ("pending", "in_progress"):
                return True
        return False

    def get_pending_tasks(self) -> list[str]:
        """Get list of pending/in_progress task texts."""
        return [t.text for t in self.tasks if t.status in ("pending", "in_progress")]

    def format_for_context(self) -> str:
        """
        Format for injection into LLM context.

        Returns:
            <current_todo>...</current_todo> block
        """
        if not self.tasks:
            return "<current_todo>\n(empty)\n</current_todo>"

        lines = ["<current_todo>"]
        for t in self.tasks:
            icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[ok]"}[t.status]
            lines.append(f"  {icon} {t.text}")
        lines.append("</current_todo>")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        """Convert to list of dicts for UI callback."""
        return [
            {"task": t.text, "status": t.status}  # v2.5.1: 'task' key for cli.py compatibility
            for t in self.tasks
        ]

    def clear(self) -> None:
        """Clear all tasks."""
        self.tasks = []

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _find_task(self, query: str) -> Optional[TodoTask]:
        """Find task by query (fuzzy match)."""
        query_lower = query.lower().strip()

        # Exact match first
        for t in self.tasks:
            if t.text.lower().strip() == query_lower:
                return t

        # Partial match
        for t in self.tasks:
            if query_lower in t.text.lower() or t.text.lower() in query_lower:
                return t

        # Filename match
        query_filename = self._extract_filename(query)
        if query_filename:
            for t in self.tasks:
                task_filename = self._extract_filename(t.text)
                if task_filename == query_filename:
                    return t

        return None

    def _matches(self, query: str, text: str) -> bool:
        """Check if query matches task text."""
        query_lower = query.lower().strip()
        text_lower = text.lower().strip()

        if query_lower == text_lower:
            return True
        if query_lower in text_lower or text_lower in query_lower:
            return True

        # Filename match
        query_filename = self._extract_filename(query)
        text_filename = self._extract_filename(text)
        if query_filename and text_filename and query_filename == text_filename:
            return True

        return False

    def _validate_completion(self, task_text: str) -> ValidationResult:
        """
        Validate that task was actually completed.

        Checks FileTracker and TerminalHistory for evidence.
        """
        # Extract filename
        filename = self._extract_filename(task_text)
        if filename:
            # Check in files (modified = created or edited)
            if filename in self.files.modified:
                return ValidationResult(ok=True)
            # May include path
            for path in self.files.modified:
                if path.endswith(filename) or filename in path:
                    return ValidationResult(ok=True)

        # Extract command keyword
        cmd_keyword = self._extract_command_keyword(task_text)
        if cmd_keyword and self.terminal.commands:
            # Check recent commands
            for cmd in reversed(self.terminal.commands[-5:]):
                if cmd_keyword in cmd.command.lower() and cmd.exit_code == 0:
                    return ValidationResult(ok=True)

        # If we can't validate — allow with warning
        # This is for fuzzy tasks like "Write main module"
        return ValidationResult(ok=True, warning="unverified")

    def _extract_filename(self, text: str) -> Optional[str]:
        """Extract filename from task text."""
        # Pattern for filenames: word.ext
        match = re.search(r'([\w\-./\\]+\.[\w]+)', text)
        if match:
            return match.group(1).split('/')[-1].split('\\')[-1]  # basename only
        return None

    def _extract_command_keyword(self, text: str) -> Optional[str]:
        """Extract command keyword from task text."""
        text_lower = text.lower()

        # Common command patterns
        keywords = [
            ("test", "pytest"),
            ("run", "python"),
            ("install", "pip"),
            ("build", "npm"),
        ]

        for trigger, keyword in keywords:
            if trigger in text_lower:
                return keyword

        return None

    def to_dict(self) -> list[dict]:
        """Convert to dict for JSON serialization."""
        return [asdict(t) for t in self.tasks]

    @classmethod
    def from_dict(cls, data: list[dict], files: "FileTracker", terminal: "TerminalHistory") -> "TodoStateMachine":
        """Load from dict."""
        machine = cls(files, terminal)
        for task_data in data:
            machine.tasks.append(TodoTask(**task_data))
        return machine


# =============================================================================
# Current Vector (for <current> block)
# =============================================================================

@dataclass
class CurrentVector:
    """
    Current state vector - what's happening RIGHT NOW.

    Updated after EVERY tool execution (no LLM call needed).
    Enables follow-up requests like "can you run it?" to work.
    """

    last_action: str = ""           # "write_file: app.py"
    last_result: str = ""           # "SUCCESS" or "ERROR: ..."
    last_file: str = ""             # "app.py"
    last_file_type: str = ""        # "python", "javascript", etc.
    last_file_runnable: bool = False  # Can be executed?
    pending_task: str = ""          # Next task from TODO
    timestamp: str = ""


# =============================================================================
# Terminal History
# =============================================================================

@dataclass
class CommandRecord:
    """A recorded terminal command."""

    command: str
    exit_code: int
    error: str = ""
    output: str = ""        # v2.4.0: preview of command output
    output_path: str = ""   # v2.4.0: path to full output file (if saved)
    timestamp: str = ""


class TerminalHistory:
    """
    Tracks terminal command history with exit codes.

    Provides data for <terminal> block in SESSION_CONTEXT.
    """

    MAX_COMMANDS = 20  # Keep last N commands

    def __init__(self):
        self.commands: list[CommandRecord] = []

    def track(self, command: str, exit_code: int, error: str = "",
              output: str = "", output_path: str = "") -> None:
        """Track a command execution."""
        # v2.2.0: M5 fix - Skip duplicate consecutive commands
        if self.commands and self.commands[-1].command == command:
            return  # Don't add duplicate

        self.commands.append(CommandRecord(
            command=command,
            exit_code=exit_code,
            error=error[:500] if error else "",  # Limit error length
            output=output,           # v2.4.0
            output_path=output_path, # v2.4.0
            timestamp=datetime.now().isoformat()
        ))

        # Keep only last N commands
        if len(self.commands) > self.MAX_COMMANDS:
            self.commands = self.commands[-self.MAX_COMMANDS:]

    def to_xml(self) -> str:
        """Generate XML for SESSION_CONTEXT <terminal> block."""
        if not self.commands:
            return "<terminal/>"

        lines = ["<terminal>"]
        for cmd in self.commands[-10:]:  # Last 10 for XML
            # v2.4.0: Show output and output_path
            if cmd.output_path and cmd.output:
                # Large output saved to file
                lines.append(f'  <cmd exit="{cmd.exit_code}">{cmd.command}</cmd>')
                lines.append(f'  <output path="{cmd.output_path}">')
                lines.append(f'{cmd.output}')
                lines.append('  </output>')
            elif cmd.error:
                # Error case
                lines.append(f'  <cmd exit="{cmd.exit_code}">')
                lines.append(f'    <command>{cmd.command}</command>')
                lines.append(f'    <error>{cmd.error}</error>')
                lines.append('  </cmd>')
            elif cmd.output:
                # Small output, no file
                lines.append(f'  <cmd exit="{cmd.exit_code}">{cmd.command}</cmd>')
                lines.append(f'  <output>{cmd.output}</output>')
            else:
                # No output
                lines.append(f'  <cmd exit="{cmd.exit_code}">{cmd.command}</cmd>')
        lines.append("</terminal>")
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        """Convert to dict for JSON serialization."""
        return [asdict(cmd) for cmd in self.commands]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "TerminalHistory":
        """Load from dict."""
        history = cls()
        for cmd_data in data:
            history.commands.append(CommandRecord(**cmd_data))
        return history


# =============================================================================
# Project Context (Main Class)
# =============================================================================

@dataclass
class ProjectIdentity:
    """Project identity info."""

    name: str
    path: str
    entry_point: str = ""


class ProjectContext:
    """
    Main project context that persists between requests.

    Stores:
    - Task summary (what we're doing)
    - Project identity (where we're working)
    - File tracker (what files we touched)
    - Terminal history (what commands we ran)
    - Knowledge base (docs, references)
    - Repo map (code structure overview) — v2.6.0

    Persists to .pocketcoder/project_context.json
    """

    CONTEXT_FILE = "project_context.json"

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.project_dir = self.project_path / POCKETCODER_DIR
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.task: Optional[TaskSummary] = None
        self.identity: ProjectIdentity = self._detect_identity()
        self.files: FileTracker = FileTracker()
        self.terminal: TerminalHistory = TerminalHistory()
        self.todo: TodoStateMachine = TodoStateMachine(self.files, self.terminal)  # v2.5.0
        self.knowledge_base: dict[str, Any] = {}
        self.session_id: str = ""
        self.current: CurrentVector = CurrentVector()  # v2.1.0: Current state

        # v2.6.0: Repo map (lazy loaded)
        self._repo_map = None
        self._repo_map_cache: Optional[str] = None

        # Load existing context if available
        self._load()

        # v2.5.0: Re-link todo machine after load (files/terminal may have changed)
        self.todo.files = self.files
        self.todo.terminal = self.terminal

    def _detect_identity(self) -> ProjectIdentity:
        """Detect project identity from directory."""
        name = self.project_path.name
        path = str(self.project_path)

        # Detect entry point
        entry_candidates = [
            "main.py", "app.py", "index.py", "run.py", "cli.py",
            "index.html", "index.js", "main.js", "src/main.py", "src/app.py"
        ]
        entry_point = ""
        for candidate in entry_candidates:
            if (self.project_path / candidate).exists():
                entry_point = candidate
                break

        return ProjectIdentity(name=name, path=path, entry_point=entry_point)

    def set_task(self, task: TaskSummary) -> None:
        """Set task summary."""
        self.task = task
        self._save()

    # =========================================================================
    # Current Vector Methods (v2.1.0)
    # =========================================================================

    def update_current(
        self,
        action: str,
        result: str,
        file: str = None,
        pending_task: str = ""
    ) -> None:
        """
        Update current vector after each tool execution.

        Called after EVERY tool - no LLM call needed.
        Enables follow-up requests to work correctly.

        Args:
            action: Tool action (e.g., "write_file: app.py")
            result: Result status ("SUCCESS" or "ERROR: ...")
            file: File path if relevant
            pending_task: Next pending task from TODO
        """
        file_type = ""
        runnable = False

        if file:
            file_type = self._detect_file_type(file)
            runnable = self._is_runnable(file, file_type)

        self.current = CurrentVector(
            last_action=action,
            last_result=result,
            last_file=file or "",
            last_file_type=file_type,
            last_file_runnable=runnable,
            pending_task=pending_task,
            timestamp=datetime.now().isoformat()
        )
        self._save()

    def _detect_file_type(self, path: str) -> str:
        """Detect file type from extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "react",
            ".tsx": "react-typescript",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".sh": "shell",
            ".bash": "shell",
            ".sql": "sql",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
        }
        ext = Path(path).suffix.lower()
        return ext_map.get(ext, "text")

    def _is_runnable(self, path: str, file_type: str) -> bool:
        """Check if file can be executed."""
        # Direct runnable types
        if file_type in ("python", "shell", "javascript"):
            return True
        # HTML can be opened in browser
        if file_type == "html":
            return True
        return False

    def current_to_xml(self) -> str:
        """Generate XML for <current> block."""
        if not self.current.last_action:
            return "<current/>"

        lines = ["<current>"]
        lines.append(f"  <last_action>{self.current.last_action}</last_action>")
        lines.append(f"  <last_result>{self.current.last_result}</last_result>")

        if self.current.last_file:
            runnable_attr = ' runnable="true"' if self.current.last_file_runnable else ''
            lines.append(
                f'  <last_file type="{self.current.last_file_type}"{runnable_attr}>'
                f'{self.current.last_file}</last_file>'
            )

        if self.current.pending_task:
            lines.append(f"  <pending_task>{self.current.pending_task}</pending_task>")

        lines.append("</current>")
        return "\n".join(lines)

    def start_session(self) -> str:
        """Start a new session, return session ID."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

        # Create session directory
        session_dir = self.project_dir / "sessions" / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save current session ID
        (self.project_dir / "current_session").write_text(self.session_id)

        self._save()
        return self.session_id

    def get_current_session_id(self) -> str:
        """Get current session ID."""
        if self.session_id:
            return self.session_id

        current_file = self.project_dir / "current_session"
        if current_file.exists():
            self.session_id = current_file.read_text().strip()
        else:
            self.session_id = self.start_session()

        return self.session_id

    def detect_knowledge_base(self) -> dict[str, Any]:
        """Detect documentation and references in project."""
        kb = {"docs": [], "references": []}

        # Check common doc files
        doc_files = [
            "README.md", "README.txt", "docs/README.md",
            "CONTRIBUTING.md", "API.md", "docs/API.md"
        ]

        for doc in doc_files:
            doc_path = self.project_path / doc
            kb["docs"].append({
                "path": doc,
                "exists": doc_path.exists()
            })

        self.knowledge_base = kb
        return kb

    def to_xml(self) -> str:
        """Generate XML for task and project blocks."""
        lines = []

        # Task block
        if self.task:
            lines.append("<task>")
            lines.append(f"  <summary>{self.task.summary}</summary>")
            lines.append(f"  <type>{self.task.task_type}</type>")
            if self.task.artifact:
                lines.append(f"  <artifact>{self.task.artifact}</artifact>")
            if self.task.stack:
                lines.append(f"  <stack>{self.task.stack}</stack>")
            if self.task.constraints:
                lines.append(f"  <constraints>{self.task.constraints}</constraints>")
            lines.append("</task>")

        # Project block
        lines.append("<project>")
        lines.append(f"  <name>{self.identity.name}</name>")
        lines.append(f"  <path>{self.identity.path}</path>")
        if self.identity.entry_point:
            lines.append(f"  <entry_point>{self.identity.entry_point}</entry_point>")
        lines.append("</project>")

        return "\n".join(lines)

    def knowledge_base_to_xml(self) -> str:
        """Generate XML for knowledge_base block."""
        if not self.knowledge_base:
            self.detect_knowledge_base()

        lines = ["<knowledge_base>"]
        for doc in self.knowledge_base.get("docs", []):
            exists = "true" if doc["exists"] else "false"
            lines.append(f'  <doc path="{doc["path"]}" exists="{exists}"/>')
        for ref in self.knowledge_base.get("references", []):
            lines.append(f'  <reference url="{ref["url"]}">{ref.get("title", "")}</reference>')
        lines.append("</knowledge_base>")
        return "\n".join(lines)

    # =========================================================================
    # Repo Map Methods (v2.6.0)
    # =========================================================================

    def get_repo_map(self) -> str:
        """
        Get repository structure map.

        Lazy loads RepoMapBuilder on first call.
        Caches result until invalidated by file changes.

        Returns:
            <repo_map>...</repo_map> XML block
        """
        if self._repo_map_cache is not None:
            return self._repo_map_cache

        # Lazy load RepoMapBuilder
        if self._repo_map is None:
            try:
                from pocketcoder.core.repo_map import RepoMapBuilder
                self._repo_map = RepoMapBuilder(self.project_path)
            except ImportError:
                return "<repo_map>\n  (repo_map module not available)\n</repo_map>"

        self._repo_map_cache = self._repo_map.build()
        return self._repo_map_cache

    def invalidate_repo_map(self) -> None:
        """
        Invalidate repo map cache.

        Call when files are created/modified/deleted.
        """
        self._repo_map_cache = None
        if self._repo_map is not None:
            self._repo_map.invalidate()

    def _save(self) -> None:
        """Save context to JSON file."""
        data = {
            "task": asdict(self.task) if self.task else None,
            "identity": asdict(self.identity),
            "files": self.files.to_dict(),
            "terminal": self.terminal.to_dict(),
            "todo": self.todo.to_dict(),  # v2.5.0
            "knowledge_base": self.knowledge_base,
            "session_id": self.session_id,
            "current": asdict(self.current)  # v2.1.0
        }

        context_file = self.project_dir / self.CONTEXT_FILE
        context_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        """Load context from JSON file."""
        context_file = self.project_dir / self.CONTEXT_FILE
        if not context_file.exists():
            return

        try:
            data = json.loads(context_file.read_text())

            if data.get("task"):
                self.task = TaskSummary(**data["task"])

            if data.get("identity"):
                self.identity = ProjectIdentity(**data["identity"])

            if data.get("files"):
                self.files = FileTracker.from_dict(data["files"])

            if data.get("terminal"):
                self.terminal = TerminalHistory.from_dict(data["terminal"])

            self.knowledge_base = data.get("knowledge_base", {})
            self.session_id = data.get("session_id", "")

            # v2.5.0: Load todo state machine
            if data.get("todo"):
                self.todo = TodoStateMachine.from_dict(
                    data["todo"], self.files, self.terminal
                )

            # v2.1.0: Load current vector
            if data.get("current"):
                self.current = CurrentVector(**data["current"])

        except Exception:
            pass  # Start fresh if load fails

    def reset(self) -> None:
        """Reset context for new task."""
        self.task = None
        self.files = FileTracker()
        self.terminal = TerminalHistory()
        self.todo = TodoStateMachine(self.files, self.terminal)  # v2.5.0
        self.current = CurrentVector()  # v2.1.0
        self.session_id = ""
        self._save()
