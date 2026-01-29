"""
Context condensation for PocketCoder.

Summarizes chat history to fit within context limits while preserving
important information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pocketcoder.core.models import Message

if TYPE_CHECKING:
    from pocketcoder.providers.base import BaseProvider


# Prompt for condensing context (structured like Cline)
CONDENSE_PROMPT = """Summarize this conversation into a structured context that preserves everything needed to continue.

## FORMAT (use exactly this structure):

### CURRENT WORK
[What is being worked on right now, the immediate task]

### KEY DECISIONS
- [Important decisions made during the conversation]
- [Technical choices, approaches selected]

### FILES MODIFIED
- [file1.py]: [what was changed]
- [file2.py]: [what was changed]

### PENDING TASKS
- [ ] [Task not yet done]
- [ ] [Another pending task]

### IMPORTANT CONTEXT
[Any critical information needed to continue: error messages, requirements, constraints]

---

Conversation to summarize:
{conversation}

---

Structured summary (be concise but complete):"""


# Known model context limits (fallback database)
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Claude models
    "claude-3": 200000,
    "claude-sonnet": 200000,
    "claude-opus": 200000,
    "claude-haiku": 200000,
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5": 16385,
    "o1": 128000,
    "o3": 200000,
    # Local models (conservative)
    "qwen": 32768,
    "qwen2.5-coder": 32768,
    "llama": 8192,
    "llama3": 8192,
    "mistral": 32768,
    "codellama": 16384,
    "deepseek": 32768,
    "phi": 4096,
    "gemma": 8192,
    # Default for unknown
    "default": 4096,
}

# Output limits (max tokens per response)
MODEL_OUTPUT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 16384,
    "gpt-4-turbo": 4096,
    "gpt-4": 8192,
    "gpt-3.5": 4096,
    "o1": 32768,
    "o3": 32768,
    # Claude
    "claude-3": 4096,
    "claude-sonnet": 8192,
    "claude-opus": 4096,
    "claude-haiku": 4096,
    # Local models
    "qwen2.5-coder": 4096,
    "qwen": 4096,
    "llama3": 4096,
    "llama": 2048,
    "mistral": 4096,
    "codellama": 4096,
    "deepseek": 4096,
    "phi": 2048,
    "gemma": 2048,
    # Default
    "default": 4096,
}


def get_model_output_limit(model_name: str) -> int:
    """Get output token limit for a model."""
    model_lower = model_name.lower()
    for key, limit in MODEL_OUTPUT_LIMITS.items():
        if key in model_lower:
            return limit
    return MODEL_OUTPUT_LIMITS["default"]


def get_model_context_limit(model_name: str) -> int:
    """
    Get context limit for a model.

    Args:
        model_name: Model name (e.g., "qwen2.5-coder:7b")

    Returns:
        Context limit in tokens
    """
    model_lower = model_name.lower()

    # Check exact matches first, then partial
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_lower:
            return limit

    return MODEL_CONTEXT_LIMITS["default"]


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple heuristic: ~4 chars per token for English.
    More conservative for code (~3.5 chars per token).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Slightly more conservative for safety
    return int(len(text) / 3.5)


def estimate_messages_tokens(messages: list[Message]) -> int:
    """
    Estimate total tokens in messages.

    Args:
        messages: List of messages

    Returns:
        Estimated token count
    """
    total = 0
    for msg in messages:
        # Role overhead (~4 tokens for role formatting)
        total += 4
        # Content
        total += estimate_tokens(msg.content)
    return total


def calculate_adaptive_limits(model_name: str) -> dict[str, int]:
    """
    Calculate adaptive limits based on model context.

    Like Aider: history = min(max(context/16, 1024), 8192)

    Args:
        model_name: Model name

    Returns:
        Dict with max_tokens, warn_at, condense_at, reserve_buffer
    """
    context = get_model_context_limit(model_name)

    # Reserve for response (like Aider: 512, but scale with context)
    reserve_buffer = min(max(context // 20, 512), 4096)

    # Effective limit
    effective = context - reserve_buffer

    # Warn threshold (80% of effective)
    warn_at = int(effective * 0.8)

    # Condense threshold (90% of effective)
    condense_at = int(effective * 0.9)

    return {
        "max_tokens": effective,
        "warn_at": warn_at,
        "condense_at": condense_at,
        "reserve_buffer": reserve_buffer,
        "raw_context": context,
    }


class ContextCondenser:
    """
    Manages context window by condensing old messages.

    When context exceeds threshold, summarizes older messages
    while keeping recent ones intact.

    Automatically adapts to model's context size.
    """

    def __init__(
        self,
        provider: "BaseProvider",
        model_name: str = "",
        max_tokens: int | None = None,
        keep_recent: int = 4,
    ):
        """
        Initialize condenser with adaptive limits.

        Args:
            provider: LLM provider for summarization
            model_name: Model name for adaptive limits
            max_tokens: Override max tokens (auto-calculated if None)
            keep_recent: Number of recent messages to keep intact
        """
        self.provider = provider
        self.model_name = model_name
        self.keep_recent = keep_recent

        # Calculate adaptive limits based on model
        if max_tokens is not None:
            self.limits = {
                "max_tokens": max_tokens,
                "warn_at": int(max_tokens * 0.8),
                "condense_at": int(max_tokens * 0.9),
                "reserve_buffer": 512,
                "raw_context": max_tokens,
            }
        else:
            self.limits = calculate_adaptive_limits(model_name)

    def update_model(self, model_name: str) -> dict[str, int]:
        """
        Update limits for new model.

        Args:
            model_name: New model name

        Returns:
            New limits dict
        """
        self.model_name = model_name
        self.limits = calculate_adaptive_limits(model_name)
        return self.limits

    def check_context(self, messages: list[Message]) -> tuple[str, int]:
        """
        Check context size and return status.

        Args:
            messages: Current messages

        Returns:
            Tuple of (status, token_count)
            Status: "ok" | "warning" | "critical" | "overflow"
        """
        tokens = estimate_messages_tokens(messages)

        if tokens >= self.limits["max_tokens"]:
            return "overflow", tokens
        elif tokens >= self.limits["condense_at"]:
            return "critical", tokens
        elif tokens >= self.limits["warn_at"]:
            return "warning", tokens
        return "ok", tokens

    def check_model_switch(
        self,
        messages: list[Message],
        new_model: str,
    ) -> tuple[bool, str]:
        """
        Check if context fits in new model's limit.

        Args:
            messages: Current messages
            new_model: Model to switch to

        Returns:
            Tuple of (fits, message)
        """
        tokens = estimate_messages_tokens(messages)
        new_limits = calculate_adaptive_limits(new_model)

        if tokens >= new_limits["max_tokens"]:
            return False, (
                f"Context ({tokens:,} tokens) exceeds {new_model}'s limit "
                f"({new_limits['max_tokens']:,}). Run /condense first."
            )
        elif tokens >= new_limits["condense_at"]:
            return True, (
                f"Context ({tokens:,} tokens) near {new_model}'s limit. "
                f"Consider /condense soon."
            )
        return True, ""

    def condense(
        self,
        messages: list[Message],
        model: str = "",
    ) -> tuple[list[Message], str]:
        """
        Condense messages to fit context.

        Args:
            messages: Current messages
            model: Model to use for summarization

        Returns:
            Tuple of (condensed_messages, summary_text)
        """
        if len(messages) <= self.keep_recent + 1:
            # Not enough to condense
            return messages, ""

        # Split into old and recent
        # Keep system prompt (first) and recent messages (last N)
        system_msg = messages[0] if messages[0].role == "system" else None
        start_idx = 1 if system_msg else 0

        old_messages = messages[start_idx:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]

        if not old_messages:
            return messages, ""

        # Format old messages for summarization
        conversation = self._format_messages(old_messages)

        # Get summary from LLM
        prompt = CONDENSE_PROMPT.format(conversation=conversation)

        try:
            response = self.provider.chat(
                [Message("user", prompt)],
                model=model,
                stream=False,
            )
            summary = response.content.strip()
        except Exception as e:
            # Fallback: simple truncation
            summary = f"[Previous context truncated due to error: {e}]"

        # Build new message list
        condensed = []

        if system_msg:
            condensed.append(system_msg)

        # Add summary as context
        condensed.append(Message(
            "user",
            f"[Previous conversation summary]\n{summary}"
        ))
        condensed.append(Message(
            "assistant",
            "I understand the context. Let's continue."
        ))

        # Add recent messages
        condensed.extend(recent_messages)

        return condensed, summary

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages for summarization prompt."""
        parts = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content[:500]  # Truncate long messages
            if len(msg.content) > 500:
                content += "..."
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)


def auto_condense_if_needed(
    messages: list[Message],
    provider: "BaseProvider",
    model: str,
    max_tokens: int = 8000,
) -> tuple[list[Message], bool, str]:
    """
    Convenience function to auto-condense if needed.

    Args:
        messages: Current messages
        provider: LLM provider
        model: Model name
        max_tokens: Context limit

    Returns:
        Tuple of (messages, was_condensed, summary)
    """
    condenser = ContextCondenser(provider, max_tokens=max_tokens)
    status, tokens = condenser.check_context(messages)

    if status == "critical":
        new_messages, summary = condenser.condense(messages, model)
        return new_messages, True, summary

    return messages, False, ""
