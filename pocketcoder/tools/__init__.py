"""
Tools module for PocketCoder.

Contains:
- files: File operations (read, write, list, search, find, open, glob)
- shell: Shell command execution
- agent: Agent control (ask, todo, completion, mode)
- edit: Advanced editing (diff, multi_edit)
- registry: TOOLS dict for LLM tool calling
"""

from __future__ import annotations

# =============================================================================
# Imports from submodules
# =============================================================================

from pocketcoder.tools.files import (
    read_file,
    is_binary,
    is_ignored,
    resolve_path,
    open_file,
    find_file,
    list_files,
    search_files,
    write_file,
    glob_files,
)
from pocketcoder.tools.shell import execute_command, is_dangerous_command
from pocketcoder.tools.agent import (
    ask_question,
    update_todo,
    attempt_completion,
    checkpoint_progress,  # v2.3.0: Episodic Memory
    switch_mode,
    get_current_mode,
    get_todo_items,
    is_tool_allowed,
    set_callbacks,
    TOOL_GROUPS,
    # v2.5.0: New TODO tools
    add_todo,
    mark_done,
    remove_todo,
    set_todo_machine,
    # Memory tools
    remember_fact,
    recall_fact,
    list_facts,
    forget_fact,
    # Notes tools
    save_note,
    load_notes,
)
from pocketcoder.tools.edit import (
    apply_diff,
    multi_edit,
    insert_at_line,
    delete_lines,
)
from pocketcoder.tools.history import (
    search_history,
    get_recent_history,
    append_to_history,
)


# =============================================================================
# Tool Registry for LLM Tool Calling
# =============================================================================
#
# WHY:  LLM generates XML tool calls, system needs to know how to execute them
# HOW:  Dictionary TOOLS → func, description, params, dangerous, group
#
# Flow:
#   LLM: <list_files><path>src/</path><recursive>true</recursive></list_files>
#   → parse_tools() finds tool_name="list_files", params={"path": "src/", "recursive": "true"}
#   → execute_tools() looks up TOOLS["list_files"]
#   → Checks is_tool_allowed() for current mode
#   → Calls func(**params)
#   → Returns result
# =============================================================================

TOOLS = {
    # =========================================================================
    # FILE OPERATIONS (6 tools)
    # =========================================================================
    "read_file": {
        "func": lambda path, start_line=None, end_line=None: read_file(
            resolve_path(path),
            int(start_line) if start_line else None,
            int(end_line) if end_line else None
        ),
        "description": "Read file content. Default: full file (truncated by preview). Use start_line/end_line for specific line range.",
        "params": ["path", "start_line", "end_line"],
        "dangerous": False,
        "group": "file",
    },
    "write_file": {
        "func": write_file,
        "description": "Create or overwrite file. Preview auto-updates in session context. Use SEARCH/REPLACE for edits!",
        "params": ["path", "content"],
        "dangerous": True,
        "group": "file",
    },
    "list_files": {
        "func": list_files,
        "description": "List files in directory with tree structure",
        "params": ["path", "recursive"],
        "dangerous": False,
        "group": "file",
    },
    "search_files": {
        "func": search_files,
        "description": "Search for pattern in file contents (grep)",
        "params": ["pattern", "path", "include"],
        "dangerous": False,
        "group": "file",
    },
    "find_file": {
        "func": find_file,
        "description": "Find file location in project by name",
        "params": ["filename"],
        "dangerous": False,
        "group": "file",
    },
    "open_file": {
        "func": lambda path: open_file(resolve_path(path)),
        "description": "Open file in default application (browser, editor)",
        "params": ["path"],
        "dangerous": False,
        "group": "file",
    },
    "glob_files": {
        "func": glob_files,
        "description": "Find files matching glob pattern (e.g. **/*.py)",
        "params": ["pattern", "path"],
        "dangerous": False,
        "group": "file",
    },

    # =========================================================================
    # SHELL (1 tool)
    # =========================================================================
    "execute_command": {
        "func": execute_command,
        "description": "Run shell command. Full output saved to file if large — path in context's <output path=...>, use read_file to access.",
        "params": ["cmd"],  # NOTE: parameter is 'cmd' not 'command'
        "dangerous": True,
        "group": "shell",
    },

    # =========================================================================
    # AGENT CONTROL (4 tools)
    # =========================================================================
    "ask_question": {
        "func": ask_question,
        "description": "Ask user a question for clarification",
        "params": ["question", "options"],
        "dangerous": False,
        "group": "agent",
    },
    "update_todo": {
        "func": update_todo,
        "description": "Update task list with progress",
        "params": ["tasks"],
        "dangerous": False,
        "group": "agent",
    },
    "attempt_completion": {
        "func": attempt_completion,
        "description": "Signal that task is complete",
        "params": ["result", "command"],
        "dangerous": False,
        "group": "agent",
    },
    "checkpoint_progress": {
        "func": checkpoint_progress,
        "description": "Save progress checkpoint on large tasks",
        "params": ["done", "remaining", "status"],
        "dangerous": False,
        "group": "agent",
    },
    "switch_mode": {
        "func": switch_mode,
        "description": "Switch operating mode (code/architect/ask/debug)",
        "params": ["mode", "reason"],
        "dangerous": False,
        "group": "agent",
    },

    # =========================================================================
    # TODO STATE MACHINE (3 tools) - v2.5.0
    # =========================================================================
    "add_todo": {
        "func": add_todo,
        "description": "Add task to plan. Rejects duplicates. See <current_todo> in context.",
        "params": ["task"],
        "dangerous": False,
        "group": "agent",
    },
    "mark_done": {
        "func": mark_done,
        "description": "Mark task completed. Validates against project_context (file must exist).",
        "params": ["task"],
        "dangerous": False,
        "group": "agent",
    },
    "remove_todo": {
        "func": remove_todo,
        "description": "Remove task from plan. Use when user cancels or task not needed.",
        "params": ["task"],
        "dangerous": False,
        "group": "agent",
    },

    # =========================================================================
    # MEMORY (4 tools)
    # =========================================================================
    "remember_fact": {
        "func": remember_fact,
        "description": "Save a fact to long-term memory (persists between sessions)",
        "params": ["key", "value"],
        "dangerous": False,
        "group": "memory",
    },
    "recall_fact": {
        "func": recall_fact,
        "description": "Recall a fact from long-term memory",
        "params": ["key"],
        "dangerous": False,
        "group": "memory",
    },
    "list_facts": {
        "func": list_facts,
        "description": "List all remembered facts",
        "params": [],
        "dangerous": False,
        "group": "memory",
    },
    "forget_fact": {
        "func": forget_fact,
        "description": "Forget (delete) a fact from memory",
        "params": ["key"],
        "dangerous": False,
        "group": "memory",
    },

    # =========================================================================
    # NOTES (2 tools)
    # =========================================================================
    "save_note": {
        "func": save_note,
        "description": "Save a note to project NOTES.md (architecture, decisions, bugs)",
        "params": ["category", "content"],
        "dangerous": False,
        "group": "notes",
    },
    "load_notes": {
        "func": load_notes,
        "description": "Load all notes from project NOTES.md",
        "params": [],
        "dangerous": False,
        "group": "notes",
    },

    # =========================================================================
    # ADVANCED EDIT (4 tools)
    # =========================================================================
    "apply_diff": {
        "func": apply_diff,
        "description": "Apply unified diff to file",
        "params": ["path", "diff"],
        "dangerous": True,
        "group": "edit",
    },
    "multi_edit": {
        "func": multi_edit,
        "description": "Apply multiple SEARCH/REPLACE edits to files",
        "params": ["edits"],
        "dangerous": True,
        "group": "edit",
    },
    "insert_at_line": {
        "func": insert_at_line,
        "description": "Insert content at specific line number",
        "params": ["path", "line_number", "content"],
        "dangerous": True,
        "group": "edit",
    },
    "delete_lines": {
        "func": delete_lines,
        "description": "Delete lines from file",
        "params": ["path", "start_line", "end_line"],
        "dangerous": True,
        "group": "edit",
    },

    # =========================================================================
    # HISTORY (1 tool) - v2.0.0
    # =========================================================================
    "search_history": {
        "func": search_history,
        "description": "Search through conversation history (for previous decisions, errors, preferences)",
        "params": ["query", "limit", "session"],
        "dangerous": False,
        "group": "history",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_tool(name: str):
    """Get tool definition by name. Returns dict or None."""
    return TOOLS.get(name)


def list_tool_names() -> list[str]:
    """List all available tool names."""
    return list(TOOLS.keys())


def get_tools_by_group(group: str) -> list[str]:
    """Get tool names by group (file, shell, agent, edit)."""
    return [name for name, tool in TOOLS.items() if tool.get("group") == group]


def get_tools_for_prompt() -> str:
    """
    Generate tools description for system prompt.

    Returns:
        Formatted string describing available tools
    """
    lines = []
    current_group = None

    for name, tool in TOOLS.items():
        group = tool.get("group", "other")

        if group != current_group:
            current_group = group
            lines.append(f"\n### {group.upper()}")

        params = ", ".join(tool["params"])
        dangerous = " ⚠️" if tool.get("dangerous") else ""
        lines.append(f"- {name}({params}): {tool['description']}{dangerous}")

    return "\n".join(lines)


def get_tool_count() -> int:
    """Get total number of tools."""
    return len(TOOLS)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Files
    "read_file",
    "write_file",
    "list_files",
    "search_files",
    "find_file",
    "open_file",
    "glob_files",
    "is_binary",
    "is_ignored",
    "resolve_path",
    # Shell
    "execute_command",
    "is_dangerous_command",
    # Agent
    "ask_question",
    "update_todo",
    "attempt_completion",
    "checkpoint_progress",
    "switch_mode",
    "get_current_mode",
    "get_todo_items",
    "is_tool_allowed",
    "set_callbacks",
    "TOOL_GROUPS",
    # v2.5.0: New TODO tools
    "add_todo",
    "mark_done",
    "remove_todo",
    "set_todo_machine",
    # Memory
    "remember_fact",
    "recall_fact",
    "list_facts",
    "forget_fact",
    # Notes
    "save_note",
    "load_notes",
    # Edit
    "apply_diff",
    "multi_edit",
    "insert_at_line",
    "delete_lines",
    # History (v2.0.0)
    "search_history",
    "get_recent_history",
    "append_to_history",
    # Registry
    "TOOLS",
    "get_tool",
    "list_tool_names",
    "get_tools_by_group",
    "get_tools_for_prompt",
    "get_tool_count",
]
