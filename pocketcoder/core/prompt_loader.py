"""
Dynamic prompt loader for PocketCoder v0.8.0.

Implements Dynamic Context Discovery:
- Compact tool descriptions in prompt
- Dynamic few-shot loading by category
- OS-specific commands
- Hints on tool errors
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Optional

# Path to prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_file(name: str) -> str:
    """Load a prompt file by name."""
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def load_json_file(name: str) -> dict:
    """Load a JSON file from prompts directory."""
    path = PROMPTS_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_current_os_commands() -> str:
    """Get OS-specific commands for current platform."""
    system = platform.system()

    os_file_map = {
        "Darwin": "os/macos.txt",
        "Linux": "os/linux.txt",
        "Windows": "os/windows.txt",
    }

    os_file = os_file_map.get(system, "os/linux.txt")
    commands = load_prompt_file(os_file)

    if commands:
        return f"\n## SYSTEM INFO\nOS: {system} ({platform.release()})\n{commands}\n"
    return f"\n## SYSTEM INFO\nOS: {system} ({platform.release()})\n"


def detect_fewshot_categories(user_input: str) -> list[str]:
    """
    Detect which few-shot categories are relevant for user input.

    Args:
        user_input: User's message

    Returns:
        List of category names to load
    """
    user_lower = user_input.lower()
    categories = []

    # FILE category
    file_triggers = [
        "file", "folder", "create", "read",
        "structure", "what files", "list", "directory"
    ]
    if any(t in user_lower for t in file_triggers):
        categories.append("file")

    # SHELL category
    shell_triggers = [
        "run", "command", "pip", "npm", "python",
        "install", "version", "test", "execute", "shell"
    ]
    if any(t in user_lower for t in shell_triggers):
        categories.append("shell")

    # EDIT category
    edit_triggers = [
        "edit", "change", "add", "fix",
        "replace", "update", "modify", "patch"
    ]
    if any(t in user_lower for t in edit_triggers):
        categories.append("edit")

    # DEBUG category
    debug_triggers = [
        "error", "doesn't work", "fail",
        "broken", "bug", "log", "crash", "exception"
    ]
    if any(t in user_lower for t in debug_triggers):
        categories.append("debug")

    # GIT category
    git_triggers = [
        "commit", "push", "git", "pull", "branch",
        "merge", "checkout", "rebase", "stash"
    ]
    if any(t in user_lower for t in git_triggers):
        categories.append("git")

    # CREATIVE category
    creative_triggers = [
        "make", "build", "website",
        "game", "app", "bot", "project", "create"
    ]
    if any(t in user_lower for t in creative_triggers):
        categories.append("creative")

    # MEMORY category
    memory_triggers = [
        "remember", "forget", "memory",
        "note", "what do you know", "recall"
    ]
    if any(t in user_lower for t in memory_triggers):
        categories.append("memory")

    # AGENT category
    agent_triggers = [
        "plan", "task", "mode",
        "architect", "ask", "thinking"
    ]
    if any(t in user_lower for t in agent_triggers):
        categories.append("agent")

    # Default: file + shell (most common)
    if not categories:
        categories = ["file", "shell"]

    return categories


def load_fewshot_examples(categories: list[str]) -> str:
    """
    Load few-shot examples for specified categories.

    Args:
        categories: List of category names

    Returns:
        Combined few-shot examples string
    """
    examples = []

    for category in categories:
        content = load_prompt_file(f"fewshot/{category}.txt")
        if content:
            examples.append(content)

    if examples:
        return "\n\n".join(examples)
    return ""


def build_compact_prompt(user_input: str = "") -> str:
    """
    Build compact system prompt.

    v2.2.0: Removed fewshot keyword matching (hacky approach).
    Now tools_compact.txt has WHEN/NOT descriptions — LLM understands intent.

    Args:
        user_input: Current user input (not used anymore, kept for API compat)

    Returns:
        Optimized system prompt
    """
    parts = []

    # 1. Base prompt (always)
    base = load_prompt_file("base.txt")
    if base:
        parts.append(base)

    # 2. OS-specific commands (current OS only)
    os_commands = get_current_os_commands()
    parts.append(os_commands)

    # 3. Compact tools with WHEN/NOT descriptions (always)
    tools_compact = load_prompt_file("tools_compact.txt")
    if tools_compact:
        parts.append(f"\n{tools_compact}")

    # v2.2.0: No more fewshot loading — WHEN/NOT in tools_compact.txt is enough

    return "\n".join(parts)


def get_tool_hint(tool_name: str, error: str = "") -> str:
    """
    Get hint for tool when validation fails.

    Args:
        tool_name: Name of the tool
        error: Error message

    Returns:
        Hint string with usage example
    """
    schemas = load_json_file("tools_full.json")

    if tool_name not in schemas:
        return f"Error: Unknown tool '{tool_name}'"

    schema = schemas[tool_name]
    params = ", ".join(schema.get("params", []))
    required = schema.get("required", [])
    example = schema.get("example", "")

    hint = f"""Error with {tool_name}: {error}

Usage: {tool_name}({params})
Required: {', '.join(required)}
Example: {example}"""

    return hint


def validate_tool_params(tool_name: str, params: dict) -> tuple[bool, str]:
    """
    Validate tool parameters against schema.

    Args:
        tool_name: Name of the tool
        params: Parameters provided

    Returns:
        Tuple of (is_valid, error_message)
    """
    schemas = load_json_file("tools_full.json")

    if tool_name not in schemas:
        # Unknown tool - let it pass (might be new tool)
        return True, ""

    schema = schemas[tool_name]
    required = schema.get("required", [])

    # Check required params
    missing = [p for p in required if p not in params or not params[p]]

    if missing:
        return False, f"Missing required parameters: {', '.join(missing)}"

    return True, ""


# For backwards compatibility
def is_compact_mode_enabled(config: dict) -> bool:
    """Check if compact prompt mode is enabled in config."""
    return config.get("compact_prompt", True)  # Default: ON
