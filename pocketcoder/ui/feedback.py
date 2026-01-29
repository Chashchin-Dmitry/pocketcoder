"""
Universal Feedback UI System for PocketCoder v2.1.0.

Provides consistent CLI output for all feedback types:
- thinking: LLM reasoning (box with title)
- action: Tool execution (prefix only)
- success: Completed action (prefix only)
- error: Failed action (box with title)
- question: User prompt (box)
- rejected: User rejection (box)
- reflect: Post-action analysis (box)

Usage:
    from pocketcoder.ui.feedback import show

    show("thinking", "Analyzing request", ["Context: app.py created", "Decision: run"])
    show("action", "Executing: python app.py")
    show("success", "Created: app.py (230 lines)")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any

from rich.console import Console
from rich.panel import Panel


@dataclass
class FeedbackStyle:
    """Style configuration for feedback type."""
    prefix: str
    use_box: bool
    title: Optional[str] = None
    border_style: str = "dim"


# Universal style registry - add new types here
STYLES = {
    "thinking": FeedbackStyle("[~]", use_box=True, title="Thinking", border_style="blue"),
    "action": FeedbackStyle("[>]", use_box=False),
    "success": FeedbackStyle("[ok]", use_box=False),
    "error": FeedbackStyle("[x]", use_box=True, title="Error", border_style="red"),
    "question": FeedbackStyle("[?]", use_box=True, title="Question", border_style="yellow"),
    "rejected": FeedbackStyle("[!]", use_box=True, title="Rejected", border_style="yellow"),
    "reflect": FeedbackStyle("[=]", use_box=True, title="Reflecting", border_style="cyan"),
    "debug": FeedbackStyle("[D]", use_box=False),
    "info": FeedbackStyle("[*]", use_box=False),
    "warning": FeedbackStyle("[!]", use_box=False),
}

# Shared console instance
_console = Console()


def show(
    feedback_type: str,
    message: str,
    details: List[str] = None,
    console: Console = None
) -> None:
    """
    Display feedback to user.

    Universal function - works for any feedback type.

    Args:
        feedback_type: One of STYLES keys (thinking, action, success, etc.)
        message: Main message to display
        details: Optional list of detail lines
        console: Optional custom Console instance
    """
    c = console or _console
    style = STYLES.get(feedback_type, STYLES["info"])

    if style.use_box:
        # Build content with details
        content = message
        if details:
            content += "\n" + "\n".join(f"  {d}" for d in details)

        panel = Panel(
            content,
            title=style.title,
            border_style=style.border_style,
            padding=(0, 1)
        )
        c.print(panel)
    else:
        # Simple prefix output
        c.print(f"{style.prefix} {message}")
        if details:
            for d in details:
                c.print(f"    {d}")


def show_debug(message: str, obj: any = None, enabled: bool = True) -> None:
    """
    Show debug message if enabled.

    Args:
        message: Debug message
        obj: Optional object to pretty-print
        enabled: Whether debug mode is enabled
    """
    if not enabled:
        return

    _console.print(f"[dim][D] {message}[/dim]")
    if obj is not None:
        _console.print(f"[dim]    {obj}[/dim]")


def show_session_context(context_xml: str, enabled: bool = True) -> None:
    """
    Show SESSION_CONTEXT in debug mode.

    Args:
        context_xml: The SESSION_CONTEXT XML string
        enabled: Whether debug mode is enabled
    """
    if not enabled:
        return

    # Truncate if too long
    if len(context_xml) > 2000:
        context_xml = context_xml[:2000] + "\n... (truncated)"

    panel = Panel(
        context_xml,
        title="SESSION_CONTEXT",
        border_style="dim",
        padding=(0, 1)
    )
    _console.print(panel)


def parse_thinking(llm_response: str) -> tuple[str, str]:
    """
    Extract <thinking> block from LLM response.

    Args:
        llm_response: Raw LLM response

    Returns:
        Tuple of (thinking_content, rest_of_response)
    """
    import re

    match = re.search(r'<thinking>(.*?)</thinking>', llm_response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        rest = llm_response[:match.start()] + llm_response[match.end():]
        return thinking, rest.strip()

    return "", llm_response


def show_thinking_if_present(llm_response: str, debug: bool = False) -> str:
    """
    Parse and display <thinking> block if present.

    Args:
        llm_response: Raw LLM response
        debug: Whether in debug mode (always show in debug)

    Returns:
        Response with thinking block removed
    """
    thinking, rest = parse_thinking(llm_response)

    if thinking:
        # Show thinking box
        lines = thinking.split('\n')
        # Limit to 5 lines for display
        if len(lines) > 5:
            display_lines = lines[:5] + [f"... (+{len(lines)-5} more lines)"]
        else:
            display_lines = lines

        show("thinking", display_lines[0], display_lines[1:] if len(display_lines) > 1 else None)

    return rest
