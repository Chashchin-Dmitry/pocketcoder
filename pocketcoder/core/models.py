"""
Data models for PocketCoder.

Contains all dataclasses used throughout the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class Message:
    """A chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for API calls."""
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    """A parsed tool call from LLM response."""

    name: str
    params: dict[str, str] = field(default_factory=dict)
    raw_match: str = ""


@dataclass
class Edit:
    """A single file edit (SEARCH/REPLACE block)."""

    filename: str
    search: str
    replace: str
    raw_match: str = ""

    @property
    def is_new_file(self) -> bool:
        """Check if this creates a new file (empty SEARCH)."""
        return self.search.strip() == ""

    @property
    def is_delete_file(self) -> bool:
        """Check if this deletes a file (empty REPLACE and full file SEARCH)."""
        return self.replace.strip() == "" and len(self.search) > 0


@dataclass
class FileContext:
    """Context for a file added to chat."""

    content: str
    lines: int
    added_at: datetime = field(default_factory=datetime.now)
    mtime: float = 0.0
    is_partial: bool = False
    partial_range: tuple[int, int] | None = None

    @classmethod
    def from_path(cls, path: Path) -> "FileContext":
        """Create FileContext from a file path."""
        content = path.read_text()
        return cls(
            content=content,
            lines=len(content.splitlines()),
            added_at=datetime.now(),
            mtime=path.stat().st_mtime,
        )


@dataclass
class ChatResponse:
    """Response from LLM provider."""

    content: str
    finish_reason: str  # "stop" | "length" | "error"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


@dataclass
class AgentStats:
    """Statistics for agent loop execution."""

    elapsed_time: float = 0.0       # seconds
    input_tokens: int = 0           # prompt tokens
    output_tokens: int = 0          # completion tokens
    iterations: int = 0             # number of loop iterations
    tools_executed: int = 0         # number of tools run

    def format(self) -> str:
        """Format stats for display: [1.2s | 847->156 tokens]"""
        time_str = f"{self.elapsed_time:.1f}s"
        tokens_str = f"{self.input_tokens}->{self.output_tokens}"
        return f"[{time_str} | {tokens_str} tokens]"


@dataclass
class TodoItem:
    """A single TODO item."""

    task: str
    status: str  # "pending" | "in_progress" | "completed"


@dataclass
class ParsedResponse:
    """Parsed LLM response with extracted components."""

    # Raw content
    raw: str = ""

    # Extracted thinking/reasoning
    thinking: str = ""

    # Extracted edits
    edits: list[Edit] = field(default_factory=list)

    # If response is a question
    is_question: bool = False
    question_text: str = ""
    options: dict[str, str] = field(default_factory=dict)

    # Shell commands found
    commands: list[str] = field(default_factory=list)

    # Tool calls (agentic mode)
    tool_calls: list["ToolCall"] = field(default_factory=list)

    # TODO list (MANDATORY in every response)
    todo: list[TodoItem] = field(default_factory=list)

    # Warnings during parsing
    warnings: list[str] = field(default_factory=list)

    # Parse errors (malformed tool calls detected)
    parse_errors: list[str] = field(default_factory=list)

    # Agent loop statistics
    stats: AgentStats | None = None


@dataclass
class Change:
    """A recorded file change for undo."""

    file: Path
    old_content: str
    new_content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    type: str  # "ollama" | "openai_compat"
    base_url: str
    default_model: str
    api_key: str | None = None
    timeout: int = 300
    max_retries: int = 3
