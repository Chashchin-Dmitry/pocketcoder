"""
Tool Result Formatters for PocketCoder v0.9.0.

Beautiful Rich-based output for each tool type:
- list_files: Tree with [D]/[F] icons
- read_file: Syntax highlighted, collapsible
- write_file: Created panel with preview
- execute_command: Status box with exit code
- ask_question: Styled menu with descriptions
- attempt_completion: Success panel
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.tree import Tree
from rich.table import Table


console = Console()


# =============================================================================
# Text Icons (no emoji)
# =============================================================================

ICONS = {
    "dir": "[D]",
    "file": "[F]",
    "ok": "[ok]",
    "error": "[x]",
    "warning": "[!]",
    "info": "[i]",
    "run": "[>]",
    "question": "[?]",
}

COLORS = {
    "dir": "cyan",
    "file": "white",
    "ok": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "run": "cyan",
    "question": "magenta",
}


def _icon(name: str) -> tuple[str, str]:
    """
    Get icon and its color.

    Returns:
        (icon_text, color_style)
    """
    icon = ICONS.get(name, "[?]")
    color = COLORS.get(name, "white")
    return icon, color


# =============================================================================
# list_files Formatter
# =============================================================================

def format_list_files(result: str, path: str = ".", max_items: int = 15) -> Panel:
    """
    Format list_files result as Rich Panel with tree.

    Args:
        result: Raw list_files output
        path: Directory path
        max_items: Maximum items to show before truncation (v1.0.3)

    Input format:
        ./
        ├── folder/
        │   └── file.py
        └── README.md

    Output:
        ┌─ Files: ./ ──────────────────────────┐
        │  [D] folder/                          │
        │  [F]   file.py                        │
        │  [F] README.md                        │
        │  [+5 more items]                      │
        │                              8 items  │
        └───────────────────────────────────────┘
    """
    lines = result.strip().split('\n')
    text = Text()
    item_count = 0
    shown_count = 0

    for line in lines:
        # Skip header line (path itself)
        if line.strip() in ("./", path, f"{path}/"):
            continue

        # Extract item name from tree characters
        # Remove tree chars: ├── │ └── etc.
        clean = re.sub(r'^[│├└─\s]+', '', line)
        if not clean:
            continue

        item_count += 1

        # v1.0.3: Skip if already shown max_items
        if shown_count >= max_items:
            continue

        shown_count += 1

        # Determine indentation level
        indent_chars = len(line) - len(line.lstrip('│├└─ '))
        indent = "  " * (indent_chars // 4)

        # Directory or file?
        is_dir = clean.endswith('/')

        if is_dir:
            icon_text, icon_color = _icon("dir")
            name = clean.rstrip('/')
        else:
            icon_text, icon_color = _icon("file")
            name = clean

        # Build line with proper styling
        text.append(f"  {indent}", style="")
        text.append(icon_text, style=icon_color)
        text.append(f" {name}\n", style=icon_color if is_dir else "")

    # v1.0.3: Show truncation message if items were hidden
    if item_count > max_items:
        hidden = item_count - max_items
        text.append(f"  [F] [{hidden} files]\n", style="dim cyan")

    # Footer with count
    if item_count > 0:
        text.append(f"\n{'':>40}{item_count} items", style="dim")

    return Panel(
        text or Text("  Empty directory", style="dim"),
        title=f"[bold]Files: {path}[/bold]",
        border_style="blue",
        padding=(0, 1),
    )


# =============================================================================
# read_file Formatter
# =============================================================================

def format_read_file(
    content: str,
    path: str,
    max_lines: int = 15,
    syntax_highlight: bool = True
) -> Panel:
    """
    Format read_file result with syntax highlighting and collapse.

    Output:
        ┌─ File: main.py (120 lines) ──────────┐
        │   1 │ def main():                     │
        │   2 │     print("hello")              │
        │   3 │ ...                             │
        │                        +117 more lines│
        └───────────────────────────────────────┘
    """
    lines = content.split('\n')
    total_lines = len(lines)

    # Get file extension for syntax
    ext = Path(path).suffix.lstrip('.')
    lexer_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'tsx': 'tsx',
        'jsx': 'jsx',
        'json': 'json',
        'yaml': 'yaml',
        'yml': 'yaml',
        'md': 'markdown',
        'html': 'html',
        'css': 'css',
        'sh': 'bash',
        'bash': 'bash',
        'sql': 'sql',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'h': 'c',
    }
    lexer = lexer_map.get(ext, 'text')

    # Truncate if too long
    if total_lines > max_lines:
        preview_lines = lines[:max_lines]
        preview_content = '\n'.join(preview_lines)
        more_lines = total_lines - max_lines
    else:
        preview_content = content
        more_lines = 0

    # Create syntax highlighted content
    if syntax_highlight and lexer != 'text':
        syntax = Syntax(
            preview_content,
            lexer,
            line_numbers=True,
            word_wrap=False,
            theme="monokai",
        )
        panel_content = syntax
    else:
        # Plain text with line numbers
        text = Text()
        for i, line in enumerate(preview_content.split('\n'), 1):
            text.append(f" {i:4} ", style="dim")
            text.append(f"│ {line}\n")
        panel_content = text

    # Add "more lines" indicator
    if more_lines > 0:
        footer = Text(f"\n{'':>35}+{more_lines} more lines", style="dim")
        from rich.console import Group
        panel_content = Group(panel_content, footer)

    return Panel(
        panel_content,
        title=f"[bold]File: {path}[/bold] [dim]({total_lines} lines)[/dim]",
        border_style="blue",
        padding=(0, 1),
    )


# =============================================================================
# write_file Formatter
# =============================================================================

def format_write_file(
    path: str,
    content: str,
    is_new: bool = True,
    max_preview_lines: int = 10
) -> Panel:
    """
    Format write_file result with preview.

    Output:
        ┌─ Created: calculator/add.py ─────────┐
        │                                       │
        │  class Add:                           │
        │      def execute(self, a, b):         │
        │          return a + b                 │
        │                                       │
        │                         5 lines, 89b  │
        └───────────────────────────────────────┘
    """
    lines = content.split('\n')
    total_lines = len(lines)
    total_bytes = len(content.encode('utf-8'))

    # Preview
    if total_lines > max_preview_lines:
        preview = '\n'.join(lines[:max_preview_lines])
        preview += f"\n  ... (+{total_lines - max_preview_lines} more lines)"
    else:
        preview = content

    # Get syntax
    ext = Path(path).suffix.lstrip('.')
    lexer_map = {
        'py': 'python', 'js': 'javascript', 'ts': 'typescript',
        'json': 'json', 'yaml': 'yaml', 'yml': 'yaml',
        'md': 'markdown', 'html': 'html', 'css': 'css',
    }
    lexer = lexer_map.get(ext, 'text')

    if lexer != 'text':
        syntax = Syntax(preview, lexer, theme="monokai", word_wrap=True)
        main_content = syntax
    else:
        main_content = Text(preview)

    # Footer stats
    footer = Text(f"\n{'':>30}{total_lines} lines, {total_bytes}b", style="dim")

    from rich.console import Group
    panel_content = Group(main_content, footer)

    action = "Created" if is_new else "Updated"
    color = "green" if is_new else "yellow"

    return Panel(
        panel_content,
        title=f"[bold {color}]{action}: {path}[/bold {color}]",
        border_style=color,
        padding=(0, 1),
    )


# =============================================================================
# execute_command Formatter
# =============================================================================

def format_execute_command(
    command: str,
    output: str,
    exit_code: int = 0,
    elapsed: float = 0.0
) -> Panel:
    """
    Format execute_command result.

    Output:
        ┌─ Shell: python -m calc add 2 3 ──────┐
        │                                       │
        │  5                                    │
        │                                       │
        │                       exit: 0  0.12s  │
        └───────────────────────────────────────┘
    """
    # Truncate long output
    max_output_lines = 30
    lines = output.split('\n')
    if len(lines) > max_output_lines:
        output = '\n'.join(lines[:max_output_lines])
        output += f"\n... (+{len(lines) - max_output_lines} more lines)"

    # Color based on exit code
    if exit_code == 0:
        border_color = "green"
        exit_style = "green"
    else:
        border_color = "red"
        exit_style = "red"

    # Content
    text = Text()
    if output.strip():
        # Check if it's an error
        if exit_code != 0:
            text.append(output, style="red")
        else:
            text.append(output)
    else:
        text.append("(no output)", style="dim")

    # Footer with stats
    text.append(f"\n\n{'':>25}exit: ", style="dim")
    text.append(str(exit_code), style=exit_style)
    if elapsed > 0:
        text.append(f"  {elapsed:.2f}s", style="dim")

    # Truncate command for title
    cmd_display = command if len(command) < 50 else command[:47] + "..."

    return Panel(
        text,
        title=f"[bold]Shell: {cmd_display}[/bold]",
        border_style=border_color,
        padding=(0, 1),
    )


# =============================================================================
# ask_question Formatter
# =============================================================================

def format_ask_question(
    question: str,
    options: list[tuple[str, str]] | list[str] | dict | str
) -> Panel:
    """
    Format ask_question with styled options.

    Input options: list of strings, list of tuples, dict, or raw string

    Output:
        ┌─ Question ───────────────────────────┐
        │                                       │
        │  What type of project?                │
        │                                       │
        │  [1] CLI                              │
        │      Command line, argparse           │
        │                                       │
        │  [0] Other (type your own)            │
        │                                       │
        └───────────────────────────────────────┘
    """
    text = Text()

    # Fix escaped newlines in question
    question = question.replace('\\n', '\n')

    # Question
    text.append(f"\n  {question}\n\n", style="bold")

    # Normalize options to list
    if isinstance(options, dict):
        # Dict: {"1": "Option 1", "2": "Option 2"}
        options_list = list(options.values())
    elif isinstance(options, str):
        # FIX 3: Try JSON first (preferred format from LLM)
        import json
        options = options.strip()
        if options.startswith('['):
            try:
                parsed = json.loads(options)
                if isinstance(parsed, list):
                    options_list = [str(o) for o in parsed]
                else:
                    options_list = [str(parsed)]
            except json.JSONDecodeError:
                options_list = None  # Fall through to other methods
        else:
            options_list = None

        if options_list is None:
            # Normalize escaped newlines
            options = options.replace('\\n', '\n')

            # Try to detect numbered options like "1) ...", "2) ...", "a) ...", etc.
            import re
            numbered_pattern = r'(?:^|\n)\s*(?:\d+\)|[a-z]\))\s*'
            if re.search(numbered_pattern, options, re.IGNORECASE):
                # Split on numbered patterns, keeping content
                parts = re.split(r'(?:^|\n)\s*\d+\)\s*', options)
                options_list = [p.strip().replace('\n', ' ') for p in parts if p.strip()]
            elif ',' in options and '\n' not in options:
                # Comma-separated on single line
                options_list = [o.strip() for o in options.split(',')]
            elif '\n' in options:
                # Multi-line: each line is an option (last resort)
                options_list = [o.strip() for o in options.split('\n')]
            else:
                # Single option
                options_list = [options] if options.strip() else []
    elif isinstance(options, list):
        options_list = options
    else:
        options_list = []

    # Filter out empty options and clean up
    options_list = [str(o).strip() for o in options_list if o and str(o).strip()]
    # Remove duplicates while preserving order
    seen = set()
    unique_options = []
    for opt in options_list:
        if opt not in seen:
            seen.add(opt)
            unique_options.append(opt)
    options_list = unique_options

    # Options
    for i, opt in enumerate(options_list, 1):
        if isinstance(opt, tuple) and len(opt) >= 2:
            label, description = opt[0], opt[1]
        else:
            label, description = str(opt), ""

        # Fix escaped newlines
        label = label.replace('\\n', ' ')

        text.append(f"  [{i}] ", style="cyan bold")
        text.append(f"{label}\n", style="bold")
        if description:
            text.append(f"      {description}\n", style="dim")
        text.append("\n")

    # Other option
    text.append("  [0] ", style="magenta")
    text.append("Other (type your answer)\n", style="dim")

    return Panel(
        text,
        title="[bold magenta]Question[/bold magenta]",
        border_style="magenta",
        padding=(0, 1),
    )


# =============================================================================
# attempt_completion Formatter
# =============================================================================

def format_completion(
    result: str,
    created_files: list[str] | None = None,
    elapsed: float = 0.0,
    tokens: tuple[int, int] | None = None
) -> Panel:
    """
    Format attempt_completion as success panel.

    Output:
        ┌─ Completed ──────────────────────────┐
        │                                       │
        │  Created CLI calculator               │
        │                                       │
        │  Files created:                       │
        │    [ok] calculator/__init__.py        │
        │    [ok] calculator/add.py             │
        │    [ok] calculator/cli.py             │
        │                                       │
        │  Next: python -m calculator.cli       │
        │                                       │
        └───────────────────────────────────────┘
                                  [94.3s | 6211->1118]
    """
    text = Text()

    # Main result
    text.append(f"\n  {result}\n", style="")

    # Created files list
    if created_files:
        text.append("\n  Files:\n", style="dim")
        for f in created_files:
            icon_text, icon_color = _icon("ok")
            text.append(f"    ", style="")
            text.append(icon_text, style=icon_color)
            text.append(f" {f}\n", style="")

    # Stats line (outside panel)
    stats_line = ""
    if elapsed > 0 or tokens:
        parts = []
        if elapsed > 0:
            parts.append(f"{elapsed:.1f}s")
        if tokens:
            input_t, output_t = tokens
            parts.append(f"{input_t}->{output_t}")
        stats_line = f"[{' | '.join(parts)}]"

    panel = Panel(
        text,
        title="[bold green]Completed[/bold green]",
        border_style="green",
        padding=(0, 1),
    )

    return panel, stats_line


# =============================================================================
# Error Formatter
# =============================================================================

def format_error(
    error: str,
    tool_name: str = "",
    hint: str = ""
) -> Panel:
    """
    Format error message.

    Output:
        ┌─ Error ──────────────────────────────┐
        │                                       │
        │  [x] Failed: permission denied        │
        │                                       │
        │  Path: /etc/hosts                     │
        │  Hint: Use sudo or check permissions  │
        │                                       │
        └───────────────────────────────────────┘
    """
    text = Text()

    icon_text, icon_color = _icon("error")
    text.append(f"\n  ", style="")
    text.append(icon_text, style=icon_color)
    text.append(f" {error}\n", style="red")

    if tool_name:
        text.append(f"\n  Tool: {tool_name}", style="dim")

    if hint:
        text.append(f"\n  Hint: {hint}", style="yellow")

    return Panel(
        text,
        title="[bold red]Error[/bold red]",
        border_style="red",
        padding=(0, 1),
    )


# =============================================================================
# Generic Tool Result Formatter
# =============================================================================

def format_tool_result(
    tool_name: str,
    result: str,
    params: dict[str, Any] | None = None,
    success: bool = True
) -> Panel | str:
    """
    Format any tool result based on tool name.

    Routes to specific formatter or returns generic panel.
    """
    params = params or {}

    # Check for cancelled operations FIRST
    if "Cancelled" in result:
        return Panel(
            Text(f"\n  [x] Operation cancelled by user\n", style="yellow"),
            title=f"[bold yellow]Cancelled: {tool_name}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )

    # Route to specific formatters
    if tool_name == "list_files":
        path = params.get("path", ".")
        return format_list_files(result, path)

    elif tool_name == "read_file":
        path = params.get("path", "file")
        return format_read_file(result, path)

    elif tool_name == "write_file":
        # Check for error/failure
        if "[x]" in result or "Error" in result or "Failed" in result:
            return format_error(result, tool_name)
        path = params.get("path", "file")
        content = params.get("content", result)
        return format_write_file(path, content, is_new=True)

    elif tool_name == "execute_command":
        cmd = params.get("cmd", params.get("command", ""))
        # Try to parse exit code from result
        exit_code = 0
        if "[x]" in result or "Error" in result or "error" in result.lower():
            exit_code = 1
        return format_execute_command(cmd, result, exit_code)

    elif tool_name == "ask_question":
        question = params.get("question", "")
        options = params.get("options", result.split(','))
        return format_ask_question(question, options)

    elif tool_name == "attempt_completion":
        return format_completion(result)

    # Generic panel for other tools
    else:
        color = "green" if success else "red"
        # Truncate long results
        if len(result) > 500:
            result = result[:500] + "\n... (truncated)"

        return Panel(
            result,
            title=f"[bold]{tool_name}[/bold]",
            border_style=color,
            padding=(0, 1),
        )
