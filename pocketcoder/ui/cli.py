"""
CLI interface for PocketCoder.

Uses prompt_toolkit for interactive input and rich for output formatting.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from rich.console import Console
from rich.markdown import Markdown

from pocketcoder.tools.shell import cleanup_background_processes, _background_processes
from pocketcoder.core.applier import ChangeTracker

if TYPE_CHECKING:
    from pocketcoder.core.coder import Coder


console = Console()

# =============================================================================
# BVM Pixel Logo (Orange #F5841F)
# =============================================================================

BVM_LOGO = """
██████╗ ██╗   ██╗███╗   ███╗
██╔══██╗██║   ██║████╗ ████║
██████╔╝██║   ██║██╔████╔██║
██╔══██╗╚██╗ ██╔╝██║╚██╔╝██║
██████╔╝ ╚████╔╝ ██║ ╚═╝ ██║
╚═════╝   ╚═══╝  ╚═╝     ╚═╝
"""

BVM_COLOR = "#F5841F"  # Orange from brand

# =============================================================================
# BVM Projects Links
# =============================================================================

BVM_PROJECTS = [
    ("PocketCoder", "bvmax.ru/ai"),
    ("AI Helper", "bvmax.ru/ai_helper"),
    ("Chat AI", "bvmax.ru/chat_ai"),
]

# =============================================================================
# Donation Addresses
# =============================================================================

DONATE_ADDRESSES = {
    "ETH / USDT (ERC-20)": "0xdF5e04d590d44603FDAdDb9f311b9dF7E5dE797c",
    "BTC": "bc1q3q25vw4jm8v4xe2g6uezq35q2uyn5jt6e00mj9",
    "USDT (TRC-20)": "TQj3X5nFQWqPEmRUWNFPjkaRUUFLxmCdok",
    "SOL": "5s5uP66VmnLMSApjq8ro639tXvSp59XEwQittzxF64mF",
}

# =============================================================================
# Slash Commands with descriptions (for autocomplete)
# =============================================================================

SLASH_COMMANDS = {
    "/donate": "Support the project",
    "/help": "Show available commands",
    "/quit": "Exit PocketCoder",
    "/exit": "Exit PocketCoder",
    "/q": "Exit PocketCoder",
    "/add": "Add file to chat context",
    "/drop": "Remove file from chat",
    "/files": "List files in chat",
    "/clear": "Clear chat history",
    "/clear-todo": "Clear current TODO list",
    "/reset": "Full reset (history + files + undo)",
    "/undo": "Undo last change",
    "/model": "Show/switch models & profiles",
    "/session": "Manage saved sessions",
    "/memory": "Show remembered facts",
    "/condense": "Compress chat history",
    "/compact": "Compress episodes (conversation history)",
    "/tokens": "Show context usage",
    "/setup": "Re-run setup wizard",
    "/ps": "Show background processes",
    "/stop": "Stop all background processes",
}


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands (like Claude Code)."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only trigger on "/" at start
        if not text.startswith("/"):
            return

        # Get partial command (e.g., "/mo" -> "mo")
        partial = text[1:].lower()

        for cmd, description in SLASH_COMMANDS.items():
            cmd_name = cmd[1:]  # Remove leading "/"
            if cmd_name.startswith(partial):
                # Show command with description
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=description,
                )


# =============================================================================
# Icons & Symbols with Colors (no emoji - cleaner look)
# =============================================================================

# Plain symbols
ICONS = {
    # Status
    "success": "+",
    "error": "x",
    "warning": "!",
    "info": "i",
    "pending": "o",
    "active": "*",
    "blocked": "!!",

    # Actions
    "run": ">",
    "stop": "#",
    "thinking": "*",
    "done": "-",

    # Files
    "file": "_",
    "folder": "#",
    "edit": "~",
    "read": "<",
    "write": ">",

    # Arrows
    "arrow_r": "->",
    "arrow_l": "<-",
    "arrow_d": "|",
    "bullet": "-",
}

# Rich markup colors for icons
ICON_COLORS = {
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "pending": "dim",
    "active": "cyan",
    "blocked": "red bold",
    "run": "cyan",
    "stop": "red",
    "thinking": "magenta",
    "done": "green",
    "file": "white",
    "folder": "blue",
    "edit": "yellow",
    "read": "cyan",
    "write": "green",
}


def icon(name: str, with_brackets: bool = True) -> str:
    """
    Get icon with Rich color markup.

    Args:
        name: Icon name from ICONS dict
        with_brackets: Wrap in [] brackets

    Returns:
        Rich-formatted string like "[green][ok][/green]"
    """
    symbol = ICONS.get(name, "?")
    color = ICON_COLORS.get(name, "white")
    if with_brackets:
        return f"[{color}][{symbol}][/{color}]"
    return f"[{color}]{symbol}[/{color}]"


def icon_plain(name: str, with_brackets: bool = True) -> str:
    """
    Get icon without color (for non-Rich output).

    Args:
        name: Icon name from ICONS dict
        with_brackets: Wrap in [] brackets

    Returns:
        Plain string like "[ok]"
    """
    symbol = ICONS.get(name, "?")
    if with_brackets:
        return f"[{symbol}]"
    return symbol

# =============================================================================
# XML Tag Stripper (clean output for user)
# =============================================================================

def strip_xml_tags(text: str) -> str:
    """
    Remove XML tool tags from LLM response for clean user output.

    Removes tags like <attempt_completion>, <list_files>, etc.
    Keeps the content between tags if it's meaningful text.
    """
    if not text:
        return ""

    # List of tool tags to remove completely (including content)
    tool_tags = [
        "attempt_completion", "list_files", "read_file", "write_file",
        "find_file", "search_files", "glob_files", "open_file",
        "execute_command", "ask_question", "update_todo", "switch_mode",
        "remember_fact", "recall_fact", "list_facts", "forget_fact",
        "save_note", "load_notes", "apply_diff", "multi_edit",
        "insert_at_line", "delete_lines",
        "todo"  # BUG 3 fix: remove raw JSON todo from output
    ]

    result = text

    # Remove tool tags with their content
    for tag in tool_tags:
        # Pattern: <tag>...</tag> or <tag>...<tag> (self-closing style)
        pattern = rf'<{tag}[^>]*>.*?</{tag}>'
        result = re.sub(pattern, '', result, flags=re.DOTALL)

        # Also remove unclosed tags
        pattern = rf'<{tag}[^>]*>[^<]*'
        result = re.sub(pattern, '', result, flags=re.DOTALL)

    # Clean up multiple newlines
    result = re.sub(r'\n{3,}', '\n\n', result)

    # Strip leading/trailing whitespace
    result = result.strip()

    return result


# =============================================================================
# Thinking Phrases (random, adds personality)
# =============================================================================

THINKING_PHRASES = [
    "[magenta][*][/magenta] Analyzing...",
    "[magenta][*][/magenta] Processing...",
    "[magenta][*][/magenta] Computing...",
    "[magenta][*][/magenta] Reasoning...",
    "[magenta][*][/magenta] Evaluating...",
    "[magenta][*][/magenta] Generating...",
    "[magenta][*][/magenta] Synthesizing...",
    "[magenta][*][/magenta] Formulating...",
]

TOOL_PHRASES = {
    "execute_command": ["[cyan][>][/cyan] Executing...", "[cyan][>][/cyan] Running...", "[cyan][>][/cyan] Processing..."],
    "read_file": ["[cyan][<][/cyan] Reading...", "[cyan][<][/cyan] Loading...", "[cyan][<][/cyan] Scanning..."],
    "write_file": ["[green][>][/green] Writing...", "[green][>][/green] Saving...", "[green][>][/green] Creating..."],
    "list_files": ["[blue][#][/blue] Listing...", "[blue][#][/blue] Scanning...", "[blue][#][/blue] Exploring..."],
    "search_files": ["[yellow][@][/yellow] Searching...", "[yellow][@][/yellow] Finding...", "[yellow][@][/yellow] Locating..."],
    "open_file": ["[white][_][/white] Opening...", "[white][_][/white] Launching...", "[white][_][/white] Displaying..."],
}


# =============================================================================
# Project Docs Auto-Load (like Claude Code reads CLAUDE.md)
# =============================================================================

# Fuzzy patterns for "docs" folder detection
DOCS_PATTERNS = [
    # Standard names
    r"^docs?$",           # doc, docs
    r"^documentation$",
    r"^readme$",
    # Typos
    r"^doks?$",           # dok, doks
    r"^dokc$",
    r"^dosc$",
    r"^dcoc$",
]


def _is_docs_like(name: str) -> bool:
    """Check if folder name looks like 'docs' (fuzzy match)."""
    name_lower = name.lower()
    for pattern in DOCS_PATTERNS:
        if re.match(pattern, name_lower, re.IGNORECASE):
            return True
    return False


def _find_project_docs(cwd: Path) -> dict:
    """
    Find project documentation files and folders.

    Returns:
        dict with keys:
            - readme: Path to README.md or None
            - docs_folder: Path to docs folder or None
            - docs_files: List of .md files in docs folder
    """
    result = {
        "readme": None,
        "docs_folder": None,
        "docs_files": [],
    }

    # 1. Find README (case-insensitive)
    readme_names = ["README.md", "readme.md", "Readme.md", "README.MD", "README"]
    for name in readme_names:
        readme_path = cwd / name
        if readme_path.exists() and readme_path.is_file():
            result["readme"] = readme_path
            break

    # 2. Find docs folder (fuzzy match)
    for item in cwd.iterdir():
        if item.is_dir() and _is_docs_like(item.name):
            result["docs_folder"] = item
            # Find .md files in docs folder
            result["docs_files"] = list(item.glob("*.md"))[:10]  # Max 10 files
            break

    return result


def _load_readme_context(readme_path: Path) -> str | None:
    """Load README.md content for context."""
    try:
        content = readme_path.read_text(encoding="utf-8")
        # Limit to first 2000 chars to not bloat context
        if len(content) > 2000:
            content = content[:2000] + "\n\n... (truncated)"
        return content
    except Exception:
        return None


def run_cli(coder: "Coder", debug: bool = False) -> int:
    """
    Run interactive CLI loop.

    Args:
        coder: Coder instance
        debug: Show debug output (iterations, parsing info)

    Returns:
        Exit code
    """
    # Setup history
    history_path = Path.home() / ".pocketcoder" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        completer=SlashCommandCompleter(),
        complete_while_typing=True,  # Show completions when typing "/" (completer filters non-slash input)
    )

    # Print welcome with BVM logo
    from rich.text import Text
    console.print(Text(BVM_LOGO, style=f"bold {BVM_COLOR}"))
    console.print(f"[bold]PocketCoder[/bold] v1.0.0 [dim]by[/dim] [{BVM_COLOR}]bvmax.ru[/{BVM_COLOR}]")

    # Show BVM projects links
    console.print(f"[dim]Projects:[/dim]")
    for name, url in BVM_PROJECTS:
        console.print(f"  [{BVM_COLOR}]→[/{BVM_COLOR}] {name}: [cyan underline]https://{url}[/cyan underline]")

    console.print(f"[dim]Provider:[/dim] {coder.provider_name} • [dim]Model:[/dim] {coder.model}")
    console.print(f"[dim]Support:[/dim] /donate")
    if coder.files:
        console.print(f"Files: {', '.join(f.name for f in coder.files)}")

    # ==========================================================================
    # Auto-load project docs (like Claude Code reads CLAUDE.md)
    # ==========================================================================
    cwd = Path.cwd()
    project_docs = _find_project_docs(cwd)

    # 1. POCKETCODER.md (shared instructions, highest priority)
    pocketcoder_md = cwd / "POCKETCODER.md"
    if pocketcoder_md.exists():
        try:
            content = pocketcoder_md.read_text(encoding="utf-8")
            if len(content) > 3000:
                content = content[:3000] + "\n\n... (truncated)"
            from pocketcoder.core.models import Message
            coder.history.append(Message("user", f"# Project Instructions (POCKETCODER.md)\n\n{content}"))
            coder.history.append(Message("assistant", "I've read the project instructions."))
            console.print(f"[dim]{ICONS['file']} Loaded: POCKETCODER.md[/dim]")
        except Exception:
            pass

    # 2. Auto-load README.md into context
    if project_docs["readme"]:
        readme_content = _load_readme_context(project_docs["readme"])
        if readme_content:
            # Add README as initial context message
            from pocketcoder.core.models import Message
            readme_msg = f"# Project README\n\n{readme_content}"
            coder.history.append(Message("user", readme_msg))
            coder.history.append(Message("assistant", "I've read the project README and understand the context."))
            console.print(f"[dim]{ICONS['read']} Loaded: {project_docs['readme'].name}[/dim]")

    # 2. Ask about docs folder if found
    if project_docs["docs_folder"]:
        docs_count = len(project_docs["docs_files"])
        folder_name = project_docs["docs_folder"].name
        console.print(f"[cyan]{ICONS['folder']} Found docs folder:[/cyan] {folder_name}/ ({docs_count} files)")

        try:
            response = input("   Load docs into context? [y/N]: ").strip().lower()
            if response in ("y", "yes"):
                from pocketcoder.core.models import Message
                loaded = 0
                for doc_file in project_docs["docs_files"]:
                    try:
                        content = doc_file.read_text(encoding="utf-8")
                        # Limit each file to 1500 chars
                        if len(content) > 1500:
                            content = content[:1500] + "\n\n... (truncated)"
                        doc_msg = f"# {doc_file.name}\n\n{content}"
                        coder.history.append(Message("user", doc_msg))
                        coder.history.append(Message("assistant", f"I've read {doc_file.name}."))
                        loaded += 1
                    except Exception:
                        pass
                console.print(f"[green]{ICONS['success']}[/green] Loaded {loaded} doc files")
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Skipped docs loading[/dim]")

    # 4. .pocketcoder/NOTES.md (project notes from previous sessions)
    notes_path = cwd / ".pocketcoder" / "NOTES.md"
    if notes_path.exists():
        try:
            content = notes_path.read_text(encoding="utf-8")
            if content.strip():
                from pocketcoder.core.models import Message
                coder.history.append(Message("user", f"# My Notes\n\n{content}"))
                coder.history.append(Message("assistant", "I've read your project notes."))
            console.print(f"[dim]{ICONS['edit']} Loaded: .pocketcoder/NOTES.md[/dim]")
        except Exception:
            pass

    # 5. Memory facts (injected into system prompt automatically)
    try:
        from pocketcoder.core.memory import MemoryManager
        mm = MemoryManager()
        stats = mm.get_stats()
        if stats.total_facts > 0:
            console.print(f"[dim]{ICONS['active']} Memory: {stats.total_facts} facts ({stats.permanent_facts} permanent)[/dim]")
    except Exception:
        pass

    console.print(f"\nType [bold]/[/bold] for commands, /quit to exit\n")

    while True:
        try:
            # Get input
            user_input = session.prompt("> ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                result = handle_command(coder, user_input)
                if result == "quit":
                    break
                continue

            # v0.9.0: Agent loop callback with Rich formatting
            from pocketcoder.ui.formatters import (
                format_tool_result, format_completion, format_error
            )

            # Track created files for completion summary
            created_files = []
            debug_mode = debug  # Use CLI flag
            current_todo = None  # FIX 1: Store todo for single render at end
            previous_todo_hash = None  # v1.0.4: Track TODO changes to avoid spam

            def _todo_hash(todo_list):
                """Create hash of TODO list for comparison."""
                if not todo_list:
                    return None
                items = []
                for item in todo_list:
                    if hasattr(item, 'task'):  # TodoItem
                        items.append((item.task, item.status))
                    elif isinstance(item, dict):
                        items.append((item.get("task", ""), item.get("status", "pending")))
                return str(items)

            def on_iteration(iteration: int, action_type: str, result: str, tool_info: dict = None):
                nonlocal current_todo
                nonlocal created_files
                nonlocal previous_todo_hash

                if action_type == "completion":
                    # v0.9.0: Beautiful completion panel
                    panel, stats_line = format_completion(result, created_files)
                    console.print()
                    console.print(panel)
                    if stats_line:
                        console.print(f"[dim]{stats_line:>60}[/dim]")

                elif action_type == "tool":
                    # v0.9.0: Formatted tool output
                    tool_name = tool_info.get("name", "tool") if tool_info else "tool"
                    tool_params = tool_info.get("params", {}) if tool_info else {}

                    # FIX 2: ask_question already printed the question interactively, don't duplicate
                    if tool_name == "ask_question":
                        return  # Skip formatting

                    # v2.5.1: Hide TODO tools from output (they update TODO panel, not shown separately)
                    if tool_name in ("add_todo", "mark_done", "remove_todo"):
                        return  # Skip — TODO panel shows state

                    # Track created files
                    if tool_name == "write_file" and "[ok]" in result:
                        path = tool_params.get("path", "")
                        if path:
                            created_files.append(path)

                    # Show status message
                    status_msgs = {
                        "list_files": "Exploring...",
                        "read_file": f"Reading {tool_params.get('path', '')}...",
                        "write_file": f"Writing {tool_params.get('path', '')}...",
                        "execute_command": f"Running command...",
                        "search_files": "Searching...",
                        "find_file": "Finding...",
                    }
                    status = status_msgs.get(tool_name, f"Executing {tool_name}...")
                    # v1.0.3: Use [>] for actions, [~] reserved for TODO in_progress
                    console.print(f"\n[cyan][>][/cyan] {status}")

                    # Format and show result
                    try:
                        # Extract actual result from "[ok] tool_name: result" format
                        clean_result = result
                        if result.startswith("[ok]") or result.startswith("[x]"):
                            parts = result.split(":", 1)
                            if len(parts) > 1:
                                clean_result = parts[1].strip()

                        panel = format_tool_result(tool_name, clean_result, tool_params)
                        console.print(panel)
                    except Exception:
                        # Fallback to simple output
                        if len(result) > 300:
                            console.print(f"[dim]{result[:300]}...[/dim]")
                        else:
                            console.print(f"[dim]{result}[/dim]")

                elif action_type == "edit":
                    console.print(f"\n[yellow][~][/yellow] Edit applied")
                    console.print(f"[dim]{result}[/dim]")

                elif action_type == "todo":
                    # v1.0.4: Render TODO only if changed (avoid spam)
                    current_todo = result if isinstance(result, list) else []
                    current_hash = _todo_hash(current_todo)

                    # Only render if TODO actually changed
                    if current_todo and current_hash != previous_todo_hash:
                        from pocketcoder.ui.todo_panel import TodoPanel
                        from pocketcoder.core.models import TodoItem
                        todo_panel = TodoPanel()
                        for item in current_todo:
                            if isinstance(item, TodoItem):
                                todo_panel.add(item.task, item.status)
                            elif isinstance(item, dict):
                                todo_panel.add(item.get("task", ""), item.get("status", "pending"))
                        console.print(todo_panel.render())
                        previous_todo_hash = current_hash

                elif action_type == "debug":
                    # v0.9.0: Hide debug by default
                    if debug_mode:
                        console.print(f"[dim]{result}[/dim]")

            # Run agent loop
            console.print(f"\n[dim]{random.choice(THINKING_PHRASES)}[/dim]")

            parsed = coder.run_agent_loop(
                user_input,
                max_iterations=10,
                on_iteration=on_iteration,
            )

            # v1.0.3: TODO now rendered immediately in on_iteration callback
            # (removed duplicate render at end)

            if not parsed:
                console.print("[yellow]No response from LLM[/yellow]")
                continue

            # Show warnings
            for warning in parsed.warnings:
                console.print(f"[yellow]Warning: {warning}[/yellow]")

            # Show thinking if present
            if parsed.thinking:
                console.print(f"\n[dim]{parsed.thinking}[/dim]")

            # Handle question response
            if parsed.is_question:
                # v0.9.0: Use formatted question panel
                from pocketcoder.ui.formatters import format_ask_question

                # v1.0.3: Clean XML tags from question text
                clean_question = strip_xml_tags(parsed.question_text)

                # Pass options as-is (formatter handles dict, list, string)
                panel = format_ask_question(
                    clean_question,
                    parsed.options if parsed.options else []
                )
                console.print(panel)
                continue

            # Show final text response (if no special actions)
            has_actions = parsed.edits or any(tc.name != "attempt_completion" for tc in parsed.tool_calls)
            if not has_actions and parsed.raw:
                # Filter out XML tags from output
                clean_text = strip_xml_tags(parsed.raw)
                if clean_text:
                    console.print(Markdown(clean_text))

            # Show stats (always)
            if parsed.stats:
                console.print(f"\n[dim]{parsed.stats.format()}[/dim]")

            # Check context usage with adaptive limits for current model
            from pocketcoder.core.condense import ContextCondenser
            condenser = ContextCondenser(coder.provider, model_name=coder.model)
            status, tokens = condenser.check_context(coder.history)
            max_tokens = condenser.limits["max_tokens"]

            if status == "overflow":
                pct = tokens * 100 // max_tokens
                console.print(f"\n[red]{ICONS['error']} Context OVERFLOW ({pct}%) — /condense required![/red]")
            elif status == "critical":
                pct = tokens * 100 // max_tokens
                console.print(f"\n[red]{ICONS['warning']} Context {pct}% full — run /condense[/red]")
            elif status == "warning":
                pct = tokens * 100 // max_tokens
                console.print(f"\n[yellow]{ICONS['info']} Context {pct}% — consider /condense[/yellow]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
            continue

        except EOFError:
            break

    return 0


def handle_command(coder: "Coder", cmd: str) -> str | None:
    """
    Handle slash commands.

    Args:
        coder: Coder instance
        cmd: Command string (with /)

    Returns:
        "quit" to exit, None otherwise
    """
    parts = cmd.split()
    command = parts[0].lower()
    args = parts[1:]

    if command in ("/quit", "/exit", "/q"):
        # Cleanup background processes (servers, watchers)
        cleanup_background_processes()
        return "quit"

    elif command in ("/help", "/h"):
        console.print("""
[bold]Commands:[/bold] [dim](type / + Tab for autocomplete)[/dim]

  /add <file>      Add file to chat
  /drop <file>     Remove file from chat
  /files           List files in chat
  /clear           Clear chat history
  /clear-todo      Clear current TODO list
  /undo            Undo last change
  /model           Show models & profiles
  /model <name>    Switch model or profile
  /model save <n>  Save current as profile
  /session         List saved sessions
  /session <id>    Load session
  /session save    Save current session
  /memory          Show remembered facts
  /memory clear    Clear all memory
  /init            Initialize project knowledge base
  /condense        Compress chat history
  /compact         Compress episodes (conversation history)
  /tokens          Show context usage
  /setup           Re-run setup wizard (change provider)
  /ps              Show background processes
  /stop            Stop all background processes
  /donate          Support the project
  /help            Show this help
  /quit            Exit
        """)

    elif command == "/donate":
        console.print()
        console.print("[bold]Support PocketCoder[/bold]")
        console.print("[dim]If you find this project useful, consider donating:[/dim]")
        console.print()
        # ETH - blue/purple brand color
        console.print("  [#627EEA]ETH[/#627EEA] [dim]/ USDT (ERC-20)[/dim]")
        console.print("  [dim]0xdF5e04d590d44603FDAdDb9f311b9dF7E5dE797c[/dim]")
        console.print()
        # BTC - orange brand color
        console.print("  [#F7931A]BTC[/#F7931A]")
        console.print("  [dim]bc1q3q25vw4jm8v4xe2g6uezq35q2uyn5jt6e00mj9[/dim]")
        console.print()
        # USDT TRC-20 (Tron) - red brand color
        console.print("  [#FF0013]TRX[/#FF0013] [dim]/ USDT (TRC-20)[/dim]")
        console.print("  [dim]TQj3X5nFQWqPEmRUWNFPjkaRUUFLxmCdok[/dim]")
        console.print()
        # SOL - purple/gradient brand color
        console.print("  [#9945FF]SOL[/#9945FF]")
        console.print("  [dim]5s5uP66VmnLMSApjq8ro639tXvSp59XEwQittzxF64mF[/dim]")
        console.print()
        console.print("[dim]Thank you for your support![/dim]")
        console.print()

    elif command == "/add":
        for arg in args:
            for path in Path.cwd().glob(arg):
                coder.add_file(path)

    elif command == "/drop":
        for arg in args:
            coder.remove_file(arg)

    elif command == "/files":
        if coder.files:
            console.print("[bold]Files in chat:[/bold]")
            for path, ctx in coder.files.items():
                console.print(f"  {path.name} ({ctx.lines} lines)")
        else:
            console.print("No files in chat")

    elif command == "/clear":
        coder.history.clear()
        # v2.3.0: Also clear episodes
        coder.episode_manager.clear()
        console.print(f"{ICONS['success']} Chat history cleared (including episodes)")
        if coder.files:
            console.print(f"[dim]  Files still in context: {len(coder.files)}. Use /reset for full reset.[/dim]")

    elif command == "/clear-todo":
        # v1.0.7: Clear current TODO list
        if coder.current_todo:
            count = len(coder.current_todo)
            coder.current_todo = []
            console.print(f"[green]{ICONS['success']}[/green] TODO cleared ({count} tasks)")
        else:
            console.print("[dim]TODO list is already empty[/dim]")

    elif command == "/compact":
        # v2.3.0: Compress old episodes into meta-summary
        episodes = coder.episode_manager.load_all()

        if len(episodes) < 5:
            console.print(f"[dim]Not enough episodes to compact ({len(episodes)}/5 required)[/dim]")
        else:
            # Get old episodes for compaction
            count, old_xml = coder.episode_manager.compact(keep_recent=3)

            if count == 0:
                console.print("[dim]Nothing to compact[/dim]")
            else:
                console.print(f"[>] Compacting {count} episodes...")

                # Ask LLM to summarize
                from pocketcoder.core.models import Message
                prompt = f"""Summarize these conversation episodes into ONE compact summary (max 500 chars):

{old_xml}

Focus on:
- Main tasks completed
- Key files created
- Important decisions made

Respond with just the summary text, no markdown."""

                try:
                    response = coder.provider.chat([Message("user", prompt)])
                    meta_summary = response.content[:500]

                    # Collect all files from old episodes
                    old_files = []
                    for ep in episodes[:-3]:
                        old_files.extend(ep.files_created)

                    # Save compacted
                    coder.episode_manager.save_compacted(
                        meta_summary=meta_summary,
                        old_files=old_files[:20],
                        keep_recent=3
                    )

                    console.print(f"[green]{ICONS['success']}[/green] Compacted {count} episodes into 1 meta-summary")
                    console.print(f"[dim]  Kept last 3 episodes intact[/dim]")

                except Exception as e:
                    console.print(f"[red]{ICONS['error']}[/red] Compact failed: {e}")

    elif command == "/reset":
        # Full reset: history + files + change tracker
        # Show what will be lost
        has_history = len(coder.history) > 0
        has_files = len(coder.files) > 0
        has_changes = coder.change_tracker.changes  # Check if there are undoable changes

        if not has_history and not has_files and not has_changes:
            console.print("[dim]Nothing to reset — session is already empty[/dim]")
        else:
            # Show warning with details
            console.print(f"[yellow]{ICONS['warning']} This will clear:[/yellow]")
            if has_history:
                console.print(f"  • {len(coder.history)} messages in history")
            if has_files:
                console.print(f"  • {len(coder.files)} files in context")
            if has_changes:
                console.print(f"  • {len(coder.change_tracker.changes)} undoable changes")

            try:
                confirm = input("\nProceed with reset? [y/N]: ").strip().lower()
                if confirm == "y":
                    coder.history.clear()
                    coder.files.clear()
                    coder.change_tracker = ChangeTracker()
                    console.print(f"[green]{ICONS['success']}[/green] Session fully reset")
                else:
                    console.print("[dim]Cancelled[/dim]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Cancelled[/dim]")

    elif command == "/undo":
        if args and args[0] == "all":
            count = coder.change_tracker.undo_all()
            console.print(f"Reverted {count} changes")
        else:
            ok, msg = coder.change_tracker.undo_last()
            console.print(msg)

    elif command == "/model":
        from pocketcoder.config.settings import (
            list_profiles, get_profile, save_profile,
            set_active_profile, save_config, add_to_recent
        )
        from pocketcoder.providers import create_provider

        config = coder.config
        active_profile = config.get("active_profile", "default")

        # /model save <name> — save current setup as profile
        if args and args[0] == "save":
            profile_name = args[1] if len(args) > 1 else None
            if not profile_name:
                console.print("[yellow]Usage: /model save <profile_name>[/yellow]")
            else:
                # Save current provider+model as profile
                new_profile = {
                    "type": config.get("provider", {}).get("type", "ollama"),
                    "base_url": config.get("provider", {}).get("base_url", ""),
                    "model": coder.model,
                }
                save_profile(config, profile_name, new_profile)
                set_active_profile(config, profile_name)
                save_config(config)
                console.print(f"[green]{ICONS['success']}[/green] Saved profile: {profile_name}")

        # /model <name_or_number> — switch to profile or model
        elif args:
            target = args[0]
            profiles = dict(list_profiles(config))

            # Check if it's a number (select from list)
            if target.isdigit():
                idx = int(target) - 1
                profile_list = list(profiles.keys())
                if 0 <= idx < len(profile_list):
                    target = profile_list[idx]
                else:
                    # Maybe it's a model number?
                    try:
                        models = coder.provider.list_models()
                        if 0 <= idx < len(models):
                            target = models[idx]
                    except Exception:
                        pass

            # Is it a saved profile?
            if target in profiles:
                profile = profiles[target]
                console.print(f"[dim]Switching to profile '{target}'...[/dim]")

                try:
                    new_provider = create_provider(profile)
                    ok, msg = new_provider.check_connection()

                    if ok:
                        coder.provider = new_provider
                        coder.provider_name = target
                        coder.model = profile.get("model", "")
                        set_active_profile(config, target)
                        save_config(config)
                        console.print(f"[green]{ICONS['success']}[/green] {target}: {coder.model}")
                    else:
                        # Still switch, just warn
                        coder.provider = new_provider
                        coder.provider_name = target
                        coder.model = profile.get("model", "")
                        set_active_profile(config, target)
                        save_config(config)
                        console.print(f"[yellow]{ICONS['warning']}[/yellow] {target}: {coder.model}")
                        console.print(f"[dim]  {msg} — will try anyway[/dim]")
                except Exception as e:
                    console.print(f"[yellow]{ICONS['warning']}[/yellow] Could not switch: {e}")

            # Otherwise treat as model name for current provider
            else:
                # Check if context fits in new model
                from pocketcoder.core.condense import ContextCondenser
                condenser = ContextCondenser(coder.provider, model_name=coder.model)
                fits, msg = condenser.check_model_switch(coder.history, target)

                if not fits:
                    console.print(f"[red]{ICONS['warning']}[/red] {msg}")
                    console.print("[dim]Switch anyway? Context may be truncated.[/dim]")
                    response = input("Continue? [y/N]: ").strip().lower()
                    if response != "y":
                        return None

                old_model = coder.model
                coder.model = target
                # Save to config
                config["provider"]["default_model"] = target
                save_config(config)
                add_to_recent(config, active_profile)
                console.print(f"[green]{ICONS['success']}[/green] Model: {target}")

                if fits and msg:
                    console.print(f"[yellow]{ICONS['info']}[/yellow] {msg}")

        # /model — show current + list profiles & models
        else:
            console.print(f"[bold]Active:[/bold] {active_profile} → {coder.model}")
            console.print(f"[dim]Provider: {coder.provider_name}[/dim]")
            console.print()

            # Show saved profiles
            profiles = list_profiles(config)
            if profiles:
                console.print("[bold]Saved profiles:[/bold]")
                for i, (name, cfg) in enumerate(profiles, 1):
                    marker = "→" if name == active_profile else " "
                    model = cfg.get("model", "?")
                    ptype = cfg.get("type", "?")
                    console.print(f"  {marker} [{i}] {name} ({ptype}: {model})")
                console.print()

            # Show available models from current provider
            try:
                models = coder.provider.list_models()
                if models:
                    console.print("[bold]Available models:[/bold]")
                    start_idx = len(profiles) + 1
                    for i, m in enumerate(models[:10], start_idx):
                        marker = "→" if m == coder.model else " "
                        console.print(f"  {marker} [{i}] {m}")
                    if len(models) > 10:
                        console.print(f"  [dim]... and {len(models) - 10} more[/dim]")
            except Exception:
                console.print("[dim]Could not list models from provider[/dim]")

            console.print()
            console.print("[dim]/model <num|name>  — switch[/dim]")
            console.print("[dim]/model save <name> — save current as profile[/dim]")

    elif command == "/session":
        from pocketcoder.core.session import SessionManager
        from datetime import datetime

        sm = SessionManager()

        # /session save — save current session
        if args and args[0] == "save":
            if not coder.history:
                console.print("[yellow]Nothing to save — chat is empty[/yellow]")
            else:
                session_id = sm.create_session(
                    profile=coder.config.get("active_profile"),
                    working_dir=str(Path.cwd()),
                )
                for msg in coder.history:
                    sm.save_message(session_id, msg.role, msg.content)
                sm.save_files(session_id, [str(p) for p in coder.files.keys()])

                # Auto-generate title from first user message
                first_user = next((m for m in coder.history if m.role == "user"), None)
                if first_user:
                    sm.auto_title(session_id, first_user.content)

                console.print(f"[green]{ICONS['success']}[/green] Session saved: {session_id}")

        # /session <id> — load session
        elif args:
            session_id = args[0]
            session = sm.load_session(session_id)

            if session:
                # Restore history
                coder.history = list(session["messages"])

                # Restore files
                for path_str in session["files"]:
                    path = Path(path_str)
                    if path.exists():
                        coder.add_file(path)

                console.print(f"[green]{ICONS['success']}[/green] Loaded: {session['title']}")
                console.print(f"[dim]  {len(coder.history)} messages, {len(session['files'])} files[/dim]")
            else:
                console.print(f"[yellow]Session not found: {session_id}[/yellow]")

        # /session — list sessions
        else:
            sessions = sm.list_sessions(limit=10)

            if sessions:
                console.print("[bold]Recent sessions:[/bold]")
                for s in sessions:
                    updated = datetime.fromtimestamp(s["updated_at"])
                    time_str = updated.strftime("%m/%d %H:%M")
                    console.print(f"  [{s['id']}] {s['title'][:40]}")
                    console.print(f"       [dim]{time_str} • {s['profile'] or 'default'}[/dim]")
                console.print()
                console.print("[dim]/session <id> — load session[/dim]")
                console.print("[dim]/session save — save current[/dim]")
            else:
                console.print("[dim]No saved sessions yet[/dim]")
                console.print("[dim]Use /session save to save current chat[/dim]")

    elif command == "/memory":
        from pocketcoder.core.memory import MemoryManager

        mm = MemoryManager()

        # /memory clear — clear all facts
        if args and args[0] == "clear":
            confirm = input("Clear all memory? [y/N]: ").strip().lower()
            if confirm == "y":
                facts = mm.get_all_facts()
                for key in list(facts.keys()):
                    mm.delete_fact(key)
                console.print(f"[green]{ICONS['success']}[/green] Memory cleared")
            else:
                console.print("[dim]Cancelled[/dim]")

        # /memory — show all facts
        else:
            console.print(mm.list_facts())
            stats = mm.get_stats()
            console.print()
            console.print(f"[dim]Total: {stats.total_facts} facts[/dim]")
            console.print(f"[dim]Permanent: {stats.permanent_facts}[/dim]")
            if stats.expiring_soon > 0:
                console.print(f"[yellow]Expiring soon: {stats.expiring_soon}[/yellow]")

    elif command == "/init":
        # v2.2.0: Initialize project knowledge base
        from pocketcoder.core.memory import MemoryManager
        from pathlib import Path

        mm = MemoryManager()
        cwd = Path.cwd()
        added = 0

        console.print(f"[dim]Scanning {cwd.name}...[/dim]")

        # 1. Project type detection
        project_type = "unknown"
        if (cwd / "package.json").exists():
            project_type = "node"
        elif (cwd / "requirements.txt").exists() or (cwd / "pyproject.toml").exists():
            project_type = "python"
        elif (cwd / "Cargo.toml").exists():
            project_type = "rust"
        elif (cwd / "go.mod").exists():
            project_type = "go"

        if project_type != "unknown":
            existing = mm.get_fact("project_type")
            if not existing or existing.value != project_type:
                mm.save_fact("project_type", project_type, category="project")
                console.print(f"  [green]+[/green] project_type: {project_type}")
                added += 1

        # 2. Entry point detection
        entry_point = None
        for candidate in ["main.py", "app.py", "index.js", "main.js", "src/index.js", "src/main.py"]:
            if (cwd / candidate).exists():
                entry_point = candidate
                break

        if entry_point:
            existing = mm.get_fact("entry_point")
            if not existing or existing.value != entry_point:
                mm.save_fact("entry_point", entry_point, category="project")
                console.print(f"  [green]+[/green] entry_point: {entry_point}")
                added += 1

        # 3. README summary (first 500 chars)
        readme = None
        for name in ["README.md", "readme.md", "README.txt", "README"]:
            if (cwd / name).exists():
                readme = (cwd / name).read_text()[:500]
                break

        if readme:
            existing = mm.get_fact("readme_summary")
            if not existing:
                mm.save_fact("readme_summary", readme.replace("\n", " ")[:200], category="project")
                console.print(f"  [green]+[/green] readme_summary: (extracted)")
                added += 1

        # 4. File count
        py_files = len(list(cwd.rglob("*.py")))
        js_files = len(list(cwd.rglob("*.js")))
        total_code = py_files + js_files

        if total_code > 0:
            file_stats = f"py:{py_files}, js:{js_files}"
            existing = mm.get_fact("file_stats")
            if not existing or existing.value != file_stats:
                mm.save_fact("file_stats", file_stats, category="project")
                console.print(f"  [green]+[/green] file_stats: {file_stats}")
                added += 1

        # Summary
        if added > 0:
            console.print(f"[green]{ICONS['success']}[/green] Initialized: {added} facts added")
        else:
            console.print(f"[dim]All up to date — nothing new to add[/dim]")

    elif command == "/condense":
        from pocketcoder.core.condense import ContextCondenser, estimate_messages_tokens

        if not coder.history:
            console.print("[dim]Nothing to condense — chat is empty[/dim]")
        else:
            # Use adaptive limits for current model
            condenser = ContextCondenser(coder.provider, model_name=coder.model)
            tokens_before = estimate_messages_tokens(coder.history)
            max_tokens = condenser.limits["max_tokens"]

            console.print(f"[dim]Condensing {len(coder.history)} messages ({tokens_before:,} tokens)...[/dim]")
            console.print(f"[dim]Model: {coder.model} (limit: {max_tokens:,})[/dim]")

            try:
                new_history, summary = condenser.condense(coder.history, coder.model)
                tokens_after = estimate_messages_tokens(new_history)
                saved = tokens_before - tokens_after

                coder.history = new_history
                console.print(f"[green]{ICONS['success']}[/green] Condensed: {tokens_before:,} {ICONS['arrow_r']} {tokens_after:,} tokens")
                console.print(f"[dim]  Saved {saved:,} tokens ({saved * 100 // tokens_before}%)[/dim]")
            except Exception as e:
                console.print(f"[yellow]{ICONS['warning']}[/yellow] Could not condense: {e}")

    elif command == "/tokens":
        from pocketcoder.core.condense import (
            ContextCondenser, estimate_messages_tokens, estimate_tokens,
            get_model_context_limit
        )

        # Use adaptive limits for current model
        condenser = ContextCondenser(coder.provider, model_name=coder.model)
        limits = condenser.limits

        # Calculate usage
        history_tokens = estimate_messages_tokens(coder.history)
        file_tokens = sum(estimate_tokens(ctx.content) for ctx in coder.files.values())
        total = history_tokens + file_tokens

        status, _ = condenser.check_context(coder.history)
        max_tokens = limits["max_tokens"]
        raw_context = limits["raw_context"]
        pct = total * 100 // max_tokens if max_tokens else 0

        # Color based on status
        status_colors = {
            "overflow": ("red", f" {ICONS['error']} OVERFLOW!"),
            "critical": ("red", " — run /condense!"),
            "warning": ("yellow", " — consider /condense"),
            "ok": ("green", ""),
        }
        color, hint = status_colors.get(status, ("white", ""))

        console.print(f"[bold]Context usage:[/bold] {coder.model}")
        console.print(f"  History:  {history_tokens:,} tokens ({len(coder.history)} messages)")
        console.print(f"  Files:    {file_tokens:,} tokens ({len(coder.files)} files)")
        console.print(f"  [{color}]Total:    {total:,} / {max_tokens:,} ({pct}%){hint}[/{color}]")
        console.print()
        console.print(f"[dim]Model context: {raw_context:,} tokens[/dim]")
        console.print(f"[dim]Reserve buffer: {limits['reserve_buffer']:,} tokens[/dim]")

    elif command == "/setup":
        from pocketcoder.config.settings import run_wizard, save_config
        from pocketcoder.providers import create_provider

        console.print("[dim]Starting setup wizard...[/dim]\n")

        new_config = run_wizard()

        if new_config:
            try:
                # Create new provider
                new_provider = create_provider(new_config["provider"])
                ok, msg = new_provider.check_connection()

                if ok:
                    coder.provider = new_provider
                    coder.provider_name = new_config["provider"].get("name", "unknown")
                    coder.model = new_config["provider"].get("default_model", "")
                    coder.config = new_config
                    save_config(new_config)
                    console.print(f"\n[green]{ICONS['success']}[/green] Switched to {coder.provider_name}: {coder.model}")
                else:
                    console.print(f"\n[yellow]{ICONS['warning']}[/yellow] Connection issue: {msg}")
                    console.print("[dim]Config saved, but connection may not work[/dim]")
                    coder.provider = new_provider
                    coder.provider_name = new_config["provider"].get("name", "unknown")
                    coder.model = new_config["provider"].get("default_model", "")
                    coder.config = new_config
                    save_config(new_config)

            except Exception as e:
                console.print(f"\n[red]{ICONS['error']}[/red] Error: {e}")
        else:
            console.print("\n[dim]Setup cancelled[/dim]")

    elif command == "/ps":
        # Show background processes
        running = [p for p in _background_processes if p.poll() is None]
        if running:
            console.print(f"[bold]Background processes ({len(running)}):[/bold]")
            for p in running:
                console.print(f"  PID {p.pid}")
        else:
            console.print("[dim]No background processes running[/dim]")

    elif command == "/stop":
        # Stop all background processes
        running = [p for p in _background_processes if p.poll() is None]
        if running:
            cleanup_background_processes()
            console.print(f"[green]{ICONS['success']}[/green] Stopped {len(running)} background process(es)")
        else:
            console.print("[dim]No background processes to stop[/dim]")

    else:
        console.print(f"Unknown command: {command}")
        console.print("Type /help for available commands")

    return None
