"""
Parser for SEARCH/REPLACE blocks from LLM responses.

This is our own implementation with recovery strategies for malformed responses.
"""

from __future__ import annotations

import json
import re
from pocketcoder.core.models import Edit, ParsedResponse, ToolCall, TodoItem
from pocketcoder.tools import TOOLS


# Main SEARCH/REPLACE pattern
EDIT_PATTERN = re.compile(
    r"^(?P<filename>[^\n]+?)\n"
    r"<<<<<<< SEARCH\n"
    r"(?P<search>.*?)"
    r"^=======\n"
    r"(?P<replace>.*?)"
    r"^>>>>>>> REPLACE",
    re.MULTILINE | re.DOTALL,
)

# Alternative patterns (for recovery)
ALT_PATTERNS = [
    # ORIGINAL/MODIFIED
    (
        r"^(?P<filename>[^\n]+?)\n"
        r"<<<<<<< ORIGINAL\n"
        r"(?P<search>.*?)"
        r"^=======\n"
        r"(?P<replace>.*?)"
        r"^>>>>>>> MODIFIED"
    ),
    # OLD/NEW
    (
        r"^(?P<filename>[^\n]+?)\n"
        r"<<<<<<< OLD\n"
        r"(?P<search>.*?)"
        r"^=======\n"
        r"(?P<replace>.*?)"
        r"^>>>>>>> NEW"
    ),
    # BEFORE/AFTER
    (
        r"^(?P<filename>[^\n]+?)\n"
        r"<<<<<<< BEFORE\n"
        r"(?P<search>.*?)"
        r"^=======\n"
        r"(?P<replace>.*?)"
        r"^>>>>>>> AFTER"
    ),
    # Without spaces after markers
    (
        r"^(?P<filename>[^\n]+?)\n"
        r"<<<<<<<SEARCH\n"
        r"(?P<search>.*?)"
        r"^=======\n"
        r"(?P<replace>.*?)"
        r"^>>>>>>>REPLACE"
    ),
]

# Pattern for thinking/reasoning sections
THINKING_PATTERNS = [
    r"##\s*(?:My thoughts|Thinking|Analysis|Reasoning):?\n(.*?)(?=\n##|\n<<<<<<|$)",
    r"\*\*(?:Thinking|Analysis):\*\*\n(.*?)(?=\n\*\*|\n<<<<<<|$)",
]

# Pattern for shell commands
SHELL_BLOCK_PATTERN = re.compile(
    r"<<<<<<< SHELL\n(.+?)\n>>>>>>> SHELL",
    re.DOTALL,
)

BASH_BLOCK_PATTERN = re.compile(
    r"```(?:bash|shell|sh)\n(.+?)```",
    re.DOTALL,
)

# =============================================================================
# TODO Pattern (MANDATORY in every response)
# =============================================================================
# Format: <todo>[{"task": "...", "status": "pending|in_progress|completed"}]</todo>

TODO_PATTERN = re.compile(
    r"<todo>(.*?)</todo>",
    re.DOTALL | re.IGNORECASE,
)

# =============================================================================
# Tool Calling Patterns (Agentic Mode)
# =============================================================================
#
# XML Format (like Kilo Code):
#   <open_file>
#     <path>main.html</path>
#   </open_file>
#
# Pattern: <tool_name>...content with <param>value</param>...</tool_name>
# =============================================================================

# Known tools to parse (avoid matching SEARCH/REPLACE markers)
# v2.5.1: GEARBOX - auto-sync with tools/__init__.py (single source of truth)
KNOWN_TOOLS = set(TOOLS.keys())

# Pattern for outer tool tag
TOOL_PATTERN = re.compile(
    r"<(\w+)>(.*?)</\1>",
    re.DOTALL,
)

# v2.5.1: Self-closing tags pattern (e.g., <attempt_completion/>)
SELF_CLOSING_PATTERN = re.compile(
    r"<(\w+)\s*/>",
    re.DOTALL,
)

# Pattern for inner param tags
PARAM_PATTERN = re.compile(
    r"<(\w+)>(.*?)</\1>",
    re.DOTALL,
)


def parse_edits(response: str) -> list[Edit]:
    """
    Parse SEARCH/REPLACE blocks from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        List of Edit objects
    """
    edits = []

    for match in EDIT_PATTERN.finditer(response):
        filename = match.group("filename").strip()
        search = match.group("search")
        replace = match.group("replace")

        # Remove trailing newline if present
        if search.endswith("\n"):
            search = search[:-1]
        if replace.endswith("\n"):
            replace = replace[:-1]

        edits.append(
            Edit(
                filename=filename,
                search=search,
                replace=replace,
                raw_match=match.group(0),
            )
        )

    return edits


def parse_edits_with_recovery(response: str) -> tuple[list[Edit], list[str]]:
    """
    Parse edits with fallback to alternative patterns.

    Args:
        response: Raw LLM response text

    Returns:
        Tuple of (edits, warnings)
    """
    warnings = []

    # Try standard pattern first
    edits = parse_edits(response)
    if edits:
        return edits, warnings

    # Try alternative patterns
    for alt_pattern in ALT_PATTERNS:
        pattern = re.compile(alt_pattern, re.MULTILINE | re.DOTALL)
        for match in pattern.finditer(response):
            filename = match.group("filename").strip()
            search = match.group("search")
            replace = match.group("replace")

            if search.endswith("\n"):
                search = search[:-1]
            if replace.endswith("\n"):
                replace = replace[:-1]

            edits.append(
                Edit(
                    filename=filename,
                    search=search,
                    replace=replace,
                    raw_match=match.group(0),
                )
            )

        if edits:
            warnings.append("Used alternative SEARCH/REPLACE markers")
            return edits, warnings

    # Check if response contains broken markers (LLM tried but failed)
    if "<<<<<<" in response or ">>>>>>>" in response:
        warnings.append("Found broken SEARCH/REPLACE markers - could not parse")

    return edits, warnings


def parse_thinking(response: str) -> str:
    """
    Extract thinking/reasoning section from response.

    Args:
        response: Raw LLM response text

    Returns:
        Extracted thinking text or empty string
    """
    for pattern in THINKING_PATTERNS:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def parse_commands(response: str) -> list[str]:
    """
    Extract shell commands from response.

    Args:
        response: Raw LLM response text

    Returns:
        List of command strings
    """
    commands = []

    # SHELL blocks
    for match in SHELL_BLOCK_PATTERN.finditer(response):
        cmd = match.group(1).strip()
        if cmd:
            commands.append(cmd)

    # ```bash blocks
    for match in BASH_BLOCK_PATTERN.finditer(response):
        cmd = match.group(1).strip()
        if cmd:
            commands.append(cmd)

    return commands


def parse_options(response: str) -> dict[str, str]:
    """
    Extract options/choices from response.

    Looks for patterns like:
    [a] Option A description
    [1] Option 1 description

    Args:
        response: Raw LLM response text

    Returns:
        Dict of option key -> description
    """
    options = {}

    # Pattern for [a-d] or [0-9] options
    # Match: [key] text until next [key] or end or double newline
    pattern = r"\[([a-d0-9])\]\s*(.+?)(?=\[[a-d0-9]\]|$|\n\n)"

    for match in re.finditer(pattern, response, re.IGNORECASE | re.DOTALL):
        key = match.group(1).lower()
        value = match.group(2).strip()
        # Clean up multi-line values
        value = " ".join(value.split())
        options[key] = value

    return options


def is_question_response(response: str) -> bool:
    """
    Check if response is a question rather than code.

    Args:
        response: Raw LLM response text

    Returns:
        True if response appears to be asking a question
    """
    # No code blocks = likely question
    if "<<<<<<" not in response:
        # Has question marks and is relatively short
        if "?" in response and len(response) < 2000:
            return True
    return False


def parse_todo(response: str) -> list[TodoItem]:
    """
    Parse TODO list from LLM response.

    Format: <todo>[{"task": "...", "status": "pending|in_progress|completed"}]</todo>

    Args:
        response: Raw LLM response text

    Returns:
        List of TodoItem objects
    """
    todo_items = []

    match = TODO_PATTERN.search(response)
    if not match:
        return todo_items

    todo_content = match.group(1).strip()

    # Try to parse as JSON
    try:
        items = json.loads(todo_content)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "task" in item:
                    todo_items.append(TodoItem(
                        task=item.get("task", ""),
                        status=item.get("status", "pending"),
                    ))
    except json.JSONDecodeError:
        # Try line-by-line parsing as fallback
        # Format: "- [x] Task" or "- [ ] Task" or "- Task (status)"
        for line in todo_content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Check for checkbox format: - [x] Task, - [ ] Task
            checkbox_match = re.match(r"^-?\s*\[([xX ])\]\s*(.+)$", line)
            if checkbox_match:
                is_done = checkbox_match.group(1).lower() == "x"
                task = checkbox_match.group(2).strip()
                todo_items.append(TodoItem(
                    task=task,
                    status="completed" if is_done else "pending",
                ))
                continue

            # Check for status suffix: Task (in_progress)
            status_match = re.match(r"^-?\s*(.+?)\s*\((pending|in_progress|completed)\)\s*$", line)
            if status_match:
                todo_items.append(TodoItem(
                    task=status_match.group(1).strip(),
                    status=status_match.group(2),
                ))
                continue

            # Plain task
            if line.startswith("-"):
                line = line[1:].strip()
            if line:
                todo_items.append(TodoItem(
                    task=line,
                    status="pending",
                ))

    return todo_items


def _extract_from_code_blocks(response: str) -> str:
    """
    Extract content from markdown code blocks and combine with original.

    This helps parse tools that LLM wrapped in ```plaintext``` or similar.

    Args:
        response: Raw LLM response

    Returns:
        Combined text (original + extracted from code blocks)
    """
    # Find all code blocks
    code_block_pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

    extracted = []
    for match in code_block_pattern.finditer(response):
        content = match.group(1).strip()
        if content:
            extracted.append(content)

    # Return original + extracted (for broader search)
    if extracted:
        return response + "\n" + "\n".join(extracted)
    return response


def parse_tools(response: str) -> list[ToolCall]:
    """
    Parse tool calls from LLM response.

    Supports multiple formats:

    1. XML format (standard):
        <open_file>
          <path>main.html</path>
        </open_file>

    2. Plaintext format (LLM variation):
        update_todo
        <tasks>[...]</tasks>

    3. Tools inside code blocks:
        ```plaintext
        attempt_completion
        <result>Done</result>
        ```

    Args:
        response: Raw LLM response text

    Returns:
        List of ToolCall objects
    """
    tools = []
    seen = set()  # For deduplication

    # Expand response with code block contents
    expanded_response = _extract_from_code_blocks(response)

    # === Format 1: Standard XML <tool_name>...</tool_name> ===
    for match in TOOL_PATTERN.finditer(expanded_response):
        tool_name = match.group(1)
        tool_content = match.group(2)

        # Skip non-tool tags (SEARCH, REPLACE, thinking, etc.)
        if tool_name not in KNOWN_TOOLS:
            continue

        # Parse params from content
        params = {}
        for param_match in PARAM_PATTERN.finditer(tool_content):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            params[param_name] = param_value

        # Dedup key
        key = (tool_name, frozenset(params.items()))
        if key not in seen:
            seen.add(key)
            tools.append(
                ToolCall(
                    name=tool_name,
                    params=params,
                    raw_match=match.group(0),
                )
            )

    # === Format 2: Self-closing tags <tool_name/> (v2.5.1) ===
    for match in SELF_CLOSING_PATTERN.finditer(expanded_response):
        tool_name = match.group(1)
        if tool_name not in KNOWN_TOOLS:
            continue
        key = (tool_name, frozenset())
        if key not in seen:
            seen.add(key)
            tools.append(
                ToolCall(
                    name=tool_name,
                    params={},
                    raw_match=match.group(0),
                )
            )

    # === Format 3: Plaintext "tool_name\n<param>value</param>" ===
    for tool_name in KNOWN_TOOLS:
        # Pattern: tool_name followed by param tags
        # Example: "update_todo\n<tasks>[...]</tasks>"
        pattern = rf"(?:^|\n){tool_name}\s*\n((?:\s*<\w+>.*?</\w+>\s*)+)"

        for match in re.finditer(pattern, expanded_response, re.DOTALL | re.MULTILINE):
            params_text = match.group(1)

            # Parse params
            params = {}
            for param_match in PARAM_PATTERN.finditer(params_text):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                params[param_name] = param_value

            if params:  # Only add if we found params
                key = (tool_name, frozenset(params.items()))
                if key not in seen:
                    seen.add(key)
                    tools.append(
                        ToolCall(
                            name=tool_name,
                            params=params,
                            raw_match=match.group(0),
                        )
                    )

    return tools


def detect_parse_errors(response: str, found_tools: list) -> list[str]:
    """
    v1.1.0: Detect malformed tool calls that parser couldn't handle.

    Args:
        response: Raw LLM response
        found_tools: Tools that were successfully parsed

    Returns:
        List of error messages for feedback to LLM
    """
    errors = []

    # If we found tools, no need to check for errors
    if found_tools:
        return errors

    # Check for malformed patterns only if no tools were parsed

    # Pattern 1: Unclosed XML tags (e.g., "<write_file>" without "</write_file>")
    for tool in KNOWN_TOOLS:
        open_tag = f"<{tool}>"
        close_tag = f"</{tool}>"
        if open_tag in response and close_tag not in response:
            errors.append(f"Unclosed tag: {open_tag} without {close_tag}")

    # Pattern 2: Attributes in tags (e.g., '<write_file path="x">')
    attr_pattern = r'<(\w+)\s+\w+\s*='
    attr_matches = re.findall(attr_pattern, response)
    for tag in attr_matches:
        if tag in KNOWN_TOOLS:
            errors.append(f"Attributes not supported: <{tag} attr=...>. Use <{tag}><param>value</param></{tag}>")

    # Pattern 3: JSON tool format
    if '"tool"' in response or '"name"' in response and '"params"' in response:
        if '{' in response and '}' in response:
            errors.append("JSON tool format not supported. Use XML: <tool_name><param>value</param></tool_name>")

    # Pattern 4: Self-closing tags — NOW SUPPORTED (v2.5.1)
    # No longer an error, handled in parse_tools()

    # Limit to 3 errors max
    return errors[:3]


def parse_response(response: str) -> ParsedResponse:
    """
    Parse complete LLM response into structured format.

    Args:
        response: Raw LLM response text

    Returns:
        ParsedResponse with all extracted components
    """
    result = ParsedResponse(raw=response)

    # Extract thinking
    result.thinking = parse_thinking(response)

    # Parse edits with recovery
    result.edits, result.warnings = parse_edits_with_recovery(response)

    # Check if it's a question
    if not result.edits and is_question_response(response):
        result.is_question = True
        result.question_text = response
        result.options = parse_options(response)

    # Extract commands
    result.commands = parse_commands(response)

    # Parse tool calls (agentic mode)
    result.tool_calls = parse_tools(response)

    # v1.1.0: Detect parse errors for feedback
    result.parse_errors = detect_parse_errors(response, result.tool_calls)

    # Parse TODO list (MANDATORY)
    result.todo = parse_todo(response)

    return result
