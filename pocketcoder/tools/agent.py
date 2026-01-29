"""
Agent control tools for PocketCoder.

Tools for LLM to interact with user and manage task flow:
- ask_question: Ask user for clarification
- update_todo: Update task progress
- attempt_completion: Signal task completion
- switch_mode: Change operating mode
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable


# =============================================================================
# State Management
# =============================================================================

# Current mode (code, architect, ask, debug)
_current_mode = "code"

# Todo list state (legacy, for backward compat)
_todo_items: list[dict] = []

# v2.5.0: TodoStateMachine reference (set by coder.py)
_todo_machine = None


def set_todo_machine(machine) -> None:
    """Set TodoStateMachine reference for tool functions."""
    global _todo_machine
    _todo_machine = machine

# Callbacks for UI integration (set by CLI)
_ask_callback: Callable[[str, list[str] | None], str] | None = None
_todo_callback: Callable[[list[dict]], None] | None = None
_completion_callback: Callable[[str], None] | None = None
_mode_callback: Callable[[str], None] | None = None


def set_callbacks(
    ask: Callable[[str, list[str] | None], str] | None = None,
    todo: Callable[[list[dict]], None] | None = None,
    completion: Callable[[str], None] | None = None,
    mode: Callable[[str], None] | None = None,
) -> None:
    """Set callbacks for UI integration."""
    global _ask_callback, _todo_callback, _completion_callback, _mode_callback
    _ask_callback = ask
    _todo_callback = todo
    _completion_callback = completion
    _mode_callback = mode


# =============================================================================
# Tool Functions
# =============================================================================

def ask_question(
    question: str,
    options: list[str] | None = None
) -> tuple[bool, str]:
    """
    Ask user a question and get response.

    Use when:
    - LLM needs clarification (which DB, which framework)
    - Critical decision requires user input
    - Multiple valid approaches exist

    Args:
        question: Question to ask user
        options: Optional list of choices (e.g. ["PostgreSQL", "MySQL", "SQLite"])

    Returns:
        Tuple of (success, user_response)
    """
    # BUG 1 fix: Include question in response so LLM sees context
    # Format: "User answered '{question}': {response}"
    # This prevents LLM from asking the same question again

    if _ask_callback:
        try:
            response = _ask_callback(question, options)
            return True, f"User answered '{question}': {response}"
        except Exception as e:
            return False, f"Error asking question: {e}"

    # Fallback: simple input
    try:
        print(f"\n❓ {question}")
        if options:
            for i, opt in enumerate(options, 1):
                print(f"  [{i}] {opt}")
            print("  [0] Other (type your answer)")

        response = input("\n> ").strip()

        # Handle numbered choice
        if options and response.isdigit():
            idx = int(response)
            if 1 <= idx <= len(options):
                return True, f"User answered '{question}': {options[idx - 1]}"

        return True, f"User answered '{question}': {response}"

    except (EOFError, KeyboardInterrupt):
        return False, "Question cancelled"


def update_todo(
    tasks: list[dict]
) -> tuple[bool, str]:
    """
    Update todo list with task progress.

    Use when:
    - Starting a complex task (5+ steps)
    - Showing progress to user
    - Breaking down large task

    Args:
        tasks: List of task dicts with keys:
            - content: Task description
            - status: "pending" | "in_progress" | "completed"

    Returns:
        Tuple of (success, formatted_todo_list)

    Example:
        update_todo([
            {"content": "Install dependencies", "status": "completed"},
            {"content": "Create models", "status": "in_progress"},
            {"content": "Add routes", "status": "pending"},
        ])
    """
    global _todo_items

    # Validate tasks
    valid_statuses = {"pending", "in_progress", "completed"}
    for task in tasks:
        if "content" not in task:
            return False, "Each task must have 'content'"
        if "status" not in task:
            task["status"] = "pending"
        if task["status"] not in valid_statuses:
            return False, f"Invalid status: {task['status']}. Use: {valid_statuses}"

    _todo_items = tasks

    # Notify UI
    if _todo_callback:
        try:
            _todo_callback(_todo_items)
        except Exception:
            pass

    # Format output
    output_lines = ["Todo List:"]
    for i, task in enumerate(tasks, 1):
        status = task["status"]
        content = task["content"]

        if status == "completed":
            icon = "✓"
        elif status == "in_progress":
            icon = "○"
        else:
            icon = "□"

        output_lines.append(f"  {icon} {i}. {content}")

    # Stats
    completed = sum(1 for t in tasks if t["status"] == "completed")
    total = len(tasks)
    output_lines.append(f"\nProgress: {completed}/{total}")

    return True, "\n".join(output_lines)


# =============================================================================
# v2.5.0: New TODO Tools (State Machine based)
# =============================================================================

def add_todo(task: str) -> tuple[bool, str]:
    """
    Add task to plan.

    Use at the beginning of multi-step tasks to create a plan.
    Rejects duplicates (exact or similar filename match).

    Args:
        task: Task description (e.g., "Create main.py")

    Returns:
        Tuple of (success, result_message)

    Example:
        <add_todo><task>Create main.py</task></add_todo>
    """
    if not _todo_machine:
        return False, "[!] TodoStateMachine not initialized"

    result = _todo_machine.add(task)
    success = result.startswith("[ok]")

    # Notify UI callback if available
    if success and _todo_callback:
        try:
            _todo_callback(_todo_machine.to_list())
        except Exception:
            pass

    return success, result


def mark_done(task: str) -> tuple[bool, str]:
    """
    Mark task as completed.

    Validates against project_context: file must exist or command must succeed.
    Use after completing an action (write_file, execute_command).

    Args:
        task: Task description or part of it

    Returns:
        Tuple of (success, result_message)

    Example:
        <mark_done><task>Create main.py</task></mark_done>
    """
    if not _todo_machine:
        return False, "[!] TodoStateMachine not initialized"

    result = _todo_machine.mark_done(task)
    success = result.startswith("[ok]")

    # Notify UI callback if available
    if success and _todo_callback:
        try:
            _todo_callback(_todo_machine.to_list())
        except Exception:
            pass

    return success, result


def remove_todo(task: str) -> tuple[bool, str]:
    """
    Remove task from plan.

    Use when user cancels or task is no longer needed.
    Cannot remove completed tasks.

    Args:
        task: Task description or part of it

    Returns:
        Tuple of (success, result_message)

    Example:
        <remove_todo><task>Write tests</task></remove_todo>
    """
    if not _todo_machine:
        return False, "[!] TodoStateMachine not initialized"

    result = _todo_machine.remove(task)
    success = result.startswith("[ok]")

    # Notify UI callback if available
    if success and _todo_callback:
        try:
            _todo_callback(_todo_machine.to_list())
        except Exception:
            pass

    return success, result


def attempt_completion(
    result: str,
    command: str | None = None
) -> tuple[bool, str]:
    """
    Signal that task is complete.

    Use when:
    - Task has been fully accomplished
    - User should verify the result
    - Clear end point is reached

    Args:
        result: Summary of what was done
        command: Optional command to demonstrate (e.g. "npm start")

    Returns:
        Tuple of (success, completion_message)
    """
    # Notify UI
    if _completion_callback:
        try:
            _completion_callback(result)
        except Exception:
            pass

    # Format output
    output_lines = [
        "✓ Task Complete!",
        "",
        f"Result: {result}",
    ]

    if command:
        output_lines.extend([
            "",
            "To verify, run:",
            f"   {command}",
        ])

    return True, "\n".join(output_lines)


def checkpoint_progress(
    done: str,
    remaining: str,
    status: str = "in_progress"
) -> tuple[bool, str]:
    """
    Save progress checkpoint when episode is getting large.

    WHEN: System asks for checkpoint (episode tokens > threshold)
    NOT: For normal task completion — use attempt_completion instead

    This tool is called when the current episode (user request → outcome cycle)
    is getting too large for context. It saves progress and waits for user
    to continue with a new message.

    Args:
        done: What has been completed (2-3 sentences, max 300 chars)
        remaining: What still needs to be done (bullet points, one per line)
        status: "in_progress" | "blocked" | "needs_input"

    Returns:
        Tuple of (success, checkpoint_message)

    Example:
        checkpoint_progress(
            done="Created main.py with basic structure, added config.py",
            remaining="- Add database models\\n- Create API routes\\n- Write tests",
            status="in_progress"
        )
    """
    valid_statuses = {"in_progress", "blocked", "needs_input"}
    if status not in valid_statuses:
        return False, f"Invalid status: {status}. Use: {valid_statuses}"

    # Truncate done to max 300 chars
    done_truncated = done[:300]
    if len(done) > 300:
        done_truncated += "..."

    # Parse remaining into list
    remaining_items = [r.strip() for r in remaining.split("\n") if r.strip()]

    # Format output
    output_lines = [
        "[ok] Checkpoint Saved",
        "",
        f"Done: {done_truncated}",
        "",
        "Remaining:",
    ]

    for item in remaining_items[:10]:
        # Remove leading "- " if present
        if item.startswith("- "):
            item = item[2:]
        output_lines.append(f"  • {item}")

    if len(remaining_items) > 10:
        output_lines.append(f"  ... (+{len(remaining_items) - 10} more)")

    output_lines.extend([
        "",
        f"Status: {status}",
        "",
        "Waiting for user to continue...",
    ])

    return True, "\n".join(output_lines)


def switch_mode(
    mode: str,
    reason: str | None = None
) -> tuple[bool, str]:
    """
    Switch operating mode.

    Modes:
    - code: Full access, can edit files (default)
    - architect: Only discussion and .md files, no code changes
    - ask: Read-only, can only read and answer questions
    - debug: Focus on debugging, limited edits

    Use when:
    - User wants to discuss without code changes
    - Switching from planning to implementation
    - Debugging session starts

    Args:
        mode: Target mode (code, architect, ask, debug)
        reason: Optional reason for switch

    Returns:
        Tuple of (success, mode_change_message)
    """
    global _current_mode

    valid_modes = {"code", "architect", "ask", "debug"}

    if mode not in valid_modes:
        return False, f"Invalid mode: {mode}. Valid modes: {valid_modes}"

    old_mode = _current_mode
    _current_mode = mode

    # Notify UI
    if _mode_callback:
        try:
            _mode_callback(mode)
        except Exception:
            pass

    # Mode descriptions
    mode_info = {
        "code": "Full access - can read, write, execute",
        "architect": "Planning only - discussion and .md files",
        "ask": "Read-only - can only read and answer",
        "debug": "Debug mode - focused on finding issues",
    }

    output_lines = [
        f"🔄 Mode: {old_mode} → {mode}",
        f"   {mode_info[mode]}",
    ]

    if reason:
        output_lines.append(f"   Reason: {reason}")

    return True, "\n".join(output_lines)


def get_current_mode() -> str:
    """Get current operating mode."""
    return _current_mode


def get_todo_items() -> list[dict]:
    """Get current todo list."""
    return _todo_items.copy()


# =============================================================================
# Tool Groups by Mode
# =============================================================================

TOOL_GROUPS = {
    "code": None,  # All tools allowed
    "architect": [
        "read_file", "list_files", "search_files", "find_file", "glob_files",
        "ask_question", "update_todo", "attempt_completion", "switch_mode",
        "checkpoint_progress",  # For large planning tasks
        "write_file",  # Only for .md files - enforced in execute
        # v2.5.0: New TODO tools
        "add_todo", "mark_done", "remove_todo",
    ],
    "ask": [
        "read_file", "list_files", "search_files", "find_file", "glob_files",
        "ask_question", "switch_mode",
    ],
    "debug": [
        "read_file", "list_files", "search_files", "find_file", "glob_files",
        "execute_command", "ask_question", "update_todo", "switch_mode",
        "checkpoint_progress",  # For long debug sessions
        # v2.5.0: New TODO tools
        "add_todo", "mark_done", "remove_todo",
    ],
}


def is_tool_allowed(tool_name: str) -> bool:
    """Check if tool is allowed in current mode."""
    allowed = TOOL_GROUPS.get(_current_mode)
    if allowed is None:
        return True  # code mode = all allowed
    return tool_name in allowed


# =============================================================================
# Memory Tools - Long-term fact storage (using new memory module)
# =============================================================================

def _get_memory_manager():
    """Get MemoryManager instance."""
    from pocketcoder.core.memory import MemoryManager
    return MemoryManager()


def remember_fact(key: str, value: str) -> str:
    """
    Save a fact to long-term memory.

    Args:
        key: Unique identifier for the fact
        value: The fact to remember

    Returns:
        Confirmation message
    """
    mm = _get_memory_manager()
    return mm.save_fact(key, value, category="explicit", source="llm")


def recall_fact(key: str) -> str:
    """
    Recall a fact from long-term memory.

    Args:
        key: Identifier of the fact to recall

    Returns:
        The remembered fact or error message
    """
    mm = _get_memory_manager()
    fact = mm.get_fact(key)
    if fact:
        return fact.value
    return f"❌ No memory for key: {key}"


def list_facts() -> str:
    """
    List all remembered facts.

    Returns:
        Formatted list of all facts
    """
    mm = _get_memory_manager()
    return mm.list_facts()


def forget_fact(key: str) -> str:
    """
    Forget (delete) a fact from memory.

    Args:
        key: Identifier of the fact to forget

    Returns:
        Confirmation message
    """
    mm = _get_memory_manager()
    return mm.delete_fact(key)


# =============================================================================
# Structured Notes - Project notes in NOTES.md
# =============================================================================

def _get_notes_path() -> Path:
    """Get path to project NOTES.md file."""
    path = Path.cwd() / ".pocketcoder" / "NOTES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_note(category: str, content: str) -> str:
    """
    Save a note to project NOTES.md.

    Args:
        category: Category of the note (architecture, decisions, bugs, todo)
        content: The note content

    Returns:
        Confirmation message
    """
    path = _get_notes_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Read existing content
    existing = ""
    if path.exists():
        existing = path.read_text()

    # Check if category exists
    category_header = f"## {category}"
    if category_header in existing:
        # Append to existing category
        lines = existing.split('\n')
        new_lines = []
        found = False
        for line in lines:
            new_lines.append(line)
            if line.strip() == category_header and not found:
                new_lines.append(f"- [{timestamp}] {content}")
                found = True
        new_content = '\n'.join(new_lines)
    else:
        # Create new category
        new_content = existing + f"\n\n{category_header}\n- [{timestamp}] {content}"

    path.write_text(new_content.strip() + '\n')
    return f"📝 Note saved to {category}"


def load_notes() -> str:
    """
    Load all notes from project NOTES.md.

    Returns:
        Contents of NOTES.md or message if empty
    """
    path = _get_notes_path()
    if path.exists():
        content = path.read_text().strip()
        if content:
            return f"📋 Project Notes:\n\n{content}"
    return "📭 No project notes yet"
