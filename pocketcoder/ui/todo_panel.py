"""
TodoPanel - Task list display for PocketCoder v0.9.0.

Displays tasks like Claude Code:
  [ok] Completed task
  [>>] Current task in progress
  [  ] Pending task
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
from rich.panel import Panel
from rich.text import Text
from rich.console import Console, Group
from rich.live import Live


@dataclass
class Task:
    """Single task in the todo list."""
    content: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None  # Result after completion

    def __post_init__(self):
        if self.status not in ("pending", "in_progress", "completed", "failed"):
            self.status = "pending"


class TodoPanel:
    """
    Task panel with live updates.

    Usage:
        todo = TodoPanel()
        todo.add("Read project structure")
        todo.add("Create file")

        with todo.live_context():
            todo.start(0)  # Mark first as in_progress
            # ... do work ...
            todo.complete(0)
            todo.start(1)
            # ... do work ...
            todo.complete(1)
    """

    # Text icons with colors (no emoji)
    # v1.0.3: Changed in_progress from [>>] to [~] to distinguish from action status [>]
    STYLES = {
        "completed":   ("[ok]", "green"),
        "in_progress": ("[~]", "yellow bold"),  # Tilde = work in progress
        "pending":     ("[  ]", "dim"),
        "failed":      ("[x]",  "red"),
    }

    def __init__(self, title: str = "Tasks"):
        self.title = title
        self.tasks: list[Task] = []
        self._live: Optional[Live] = None
        self._console = Console()

    def add(self, content: str, status: str = "pending") -> int:
        """
        Add a task to the list.

        Args:
            content: Task description
            status: Initial status

        Returns:
            Task index
        """
        task = Task(content=content, status=status)
        self.tasks.append(task)
        self._refresh()
        return len(self.tasks) - 1

    def add_many(self, contents: list[str]) -> list[int]:
        """Add multiple tasks at once."""
        indices = []
        for content in contents:
            indices.append(self.add(content))
        return indices

    def start(self, index: int) -> None:
        """Mark task as in_progress."""
        if 0 <= index < len(self.tasks):
            # Mark previous in_progress as pending (only one active at a time)
            for task in self.tasks:
                if task.status == "in_progress":
                    task.status = "pending"
            self.tasks[index].status = "in_progress"
            self._refresh()

    def complete(self, index: int, result: Optional[str] = None) -> None:
        """Mark task as completed."""
        if 0 <= index < len(self.tasks):
            self.tasks[index].status = "completed"
            self.tasks[index].result = result
            self._refresh()

    def fail(self, index: int, error: Optional[str] = None) -> None:
        """Mark task as failed."""
        if 0 <= index < len(self.tasks):
            self.tasks[index].status = "failed"
            self.tasks[index].result = error
            self._refresh()

    def update(self, index: int, status: str, result: Optional[str] = None) -> None:
        """Update task status directly."""
        if 0 <= index < len(self.tasks):
            self.tasks[index].status = status
            if result:
                self.tasks[index].result = result
            self._refresh()

    def clear(self) -> None:
        """Clear all tasks."""
        self.tasks = []
        self._refresh()

    def get_current(self) -> Optional[int]:
        """Get index of current in_progress task."""
        for i, task in enumerate(self.tasks):
            if task.status == "in_progress":
                return i
        return None

    def all_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(t.status == "completed" for t in self.tasks)

    def render(self) -> Panel:
        """
        Render the todo panel.

        Returns:
            Rich Panel object
        """
        if not self.tasks:
            content = Text("  No tasks", style="dim")
        else:
            content = Text()
            for i, task in enumerate(self.tasks):
                icon, style = self.STYLES.get(task.status, ("[?]", "white"))

                # Icon
                content.append(f"  {icon} ", style=style)

                # Task content
                text_style = style if task.status != "pending" else ""
                content.append(task.content, style=text_style)

                # Result hint for completed/failed (optional, truncated)
                if task.result and task.status in ("completed", "failed"):
                    hint = task.result[:30] + "..." if len(task.result) > 30 else task.result
                    content.append(f" ({hint})", style="dim")

                content.append("\n")

        return Panel(
            content,
            title=f"[bold]{self.title}[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

    def _refresh(self) -> None:
        """Refresh live display if active."""
        if self._live:
            self._live.update(self.render())

    def live_context(self) -> Live:
        """
        Get Live context for real-time updates.

        Usage:
            with todo.live_context():
                todo.start(0)
                # ... work ...
                todo.complete(0)
        """
        self._live = Live(
            self.render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,  # Keep panel after exit
        )
        return self._live

    def print_static(self) -> None:
        """Print panel without live updates."""
        self._console.print(self.render())


class TaskDecomposer:
    """
    Detects if task needs decomposition and parses planning output.

    Multi-file triggers:
    - "all files"
    - "each"
    - "multiple"
    - Numbers: "5 files", "3 files"
    """

    MULTI_FILE_TRIGGERS = [
        r"all\s+file",
        r"each\s+file",
        r"every\s+file",
        r"multiple\s+file",
        r"several\s+file",
        # Numbers
        r"\d+\s+file",
        # Folder creation with files
        r"folder.*with\s+file",
        r"create.*project",
    ]

    @classmethod
    def needs_decomposition(cls, user_input: str) -> bool:
        """
        Check if task needs decomposition into subtasks.

        Args:
            user_input: User's request

        Returns:
            True if task should be decomposed
        """
        import re
        text = user_input.lower()

        for pattern in cls.MULTI_FILE_TRIGGERS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def parse_plan(llm_output: str) -> list[str]:
        """
        Parse planning phase output into task list.

        Expects format:
            1. [action] description
            2. [action] description

        Or:
            - Task description
            - Another task

        Args:
            llm_output: LLM's planning response

        Returns:
            List of task descriptions
        """
        import re

        tasks = []

        # Pattern 1: Numbered list with optional [action]
        # 1. [create] file.py
        # 2. Edit config
        numbered_pattern = r'^\s*\d+[\.\)]\s*(?:\[[\w]+\]\s*)?(.+)$'

        # Pattern 2: Bullet points
        # - Create file
        # * Edit config
        bullet_pattern = r'^\s*[-*]\s+(.+)$'

        for line in llm_output.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Try numbered pattern
            match = re.match(numbered_pattern, line)
            if match:
                task = match.group(1).strip()
                if task and len(task) > 3:  # Skip too short
                    tasks.append(task)
                continue

            # Try bullet pattern
            match = re.match(bullet_pattern, line)
            if match:
                task = match.group(1).strip()
                if task and len(task) > 3:
                    tasks.append(task)

        return tasks


# Convenience function for simple usage
def create_todo_panel(title: str = "Tasks") -> TodoPanel:
    """Create a new TodoPanel instance."""
    return TodoPanel(title=title)
