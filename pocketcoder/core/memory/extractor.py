"""
Fact extractor for PocketCoder.

Automatically extracts facts from conversations using regex patterns.
"""

from __future__ import annotations

import re
from typing import Optional

from pocketcoder.core.memory.types import Fact


# Extraction patterns by category
# Each pattern should have a named group for the value
PATTERNS = {
    # Identity - permanent facts about user
    "identity": {
        "user_name": [
            r"(?:my name is|i'?m|call me)\s+([A-Za-z]+)",
        ],
        "user_email": [
            r"(?:my email|email)[:\s]+(\S+@\S+\.\S+)",
        ],
    },

    # Preferences - permanent user preferences
    "preference": {
        "preferred_editor": [
            r"(?:i use|prefer|using)\s+(vim|neovim|nvim|vscode|vs code|emacs|sublime|atom|intellij|pycharm)",
        ],
        "preferred_language": [
            r"(?:i code in|i write in|coding in)\s+(\w+)",
        ],
        "preferred_os": [
            r"(?:i use|i'm on|running)\s+(windows|mac|macos|linux|ubuntu)",
        ],
    },

    # Project - context about current project (30 days TTL)
    "project": {
        "project_framework": [
            r"(?:project uses|using|built with)\s+(react|vue|angular|svelte|next|nuxt|django|flask|fastapi|express|nest)",
        ],
        "project_database": [
            r"(?:database|db|using)\s+(postgres|postgresql|mysql|sqlite|mongodb|redis)",
        ],
        "project_language": [
            r"(?:written in|built in|using)\s+(python|javascript|typescript|go|rust|java|kotlin|swift)",
        ],
    },

    # Task - current work (7 days TTL)
    "task": {
        "current_task": [
            r"(?:working on|currently building|implementing)\s+(.{10,50})",
            r"(?:need to|have to|must)\s+(.{10,50})",
        ],
    },

    # Context - recent discussions (1 day TTL)
    "context": {
        "discussed_topic": [
            r"(?:discussed|talked about|mentioned)\s+(.{10,30})",
        ],
    },
}


class FactExtractor:
    """
    Extracts facts from text using regex patterns.

    Designed to run after every LLM response to automatically
    capture important information without relying on LLM tool calls.
    """

    def __init__(self, custom_patterns: Optional[dict] = None):
        """
        Initialize extractor.

        Args:
            custom_patterns: Additional patterns to use
        """
        self.patterns = PATTERNS.copy()
        if custom_patterns:
            for category, facts in custom_patterns.items():
                if category not in self.patterns:
                    self.patterns[category] = {}
                self.patterns[category].update(facts)

    def extract(self, text: str) -> dict[str, Fact]:
        """
        Extract facts from text.

        Args:
            text: Text to analyze (user message + LLM response)

        Returns:
            Dict of extracted facts (key -> Fact)
        """
        text_lower = text.lower()
        extracted = {}

        for category, fact_patterns in self.patterns.items():
            for fact_key, patterns in fact_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        # Clean up value
                        value = self._clean_value(value)

                        if value and len(value) >= 2:
                            extracted[fact_key] = Fact(
                                key=fact_key,
                                value=value,
                                category=category,
                                source="auto",
                            )
                        break  # First match wins

        return extracted

    def extract_from_conversation(
        self,
        user_message: str,
        llm_response: str
    ) -> dict[str, Fact]:
        """
        Extract facts from a conversation turn.

        Args:
            user_message: What user said
            llm_response: What LLM replied

        Returns:
            Dict of extracted facts
        """
        # Combine for analysis (user message is more important)
        combined = f"{user_message}\n{llm_response}"
        return self.extract(combined)

    def _clean_value(self, value: str) -> str:
        """Clean extracted value."""
        # Remove trailing punctuation
        value = value.rstrip(".,!?;:")
        # Capitalize first letter for names
        if len(value) > 0:
            value = value[0].upper() + value[1:]
        return value


def extract_explicit_memory_request(text: str) -> Optional[tuple[str, str]]:
    """
    Check if user explicitly asks to remember something.

    Patterns like:
    - "remember that X"
    - "save this: X"
    - "note that X"

    Args:
        text: User message

    Returns:
        Tuple of (key, value) or None
    """
    patterns = [
        r"remember\s+(?:that\s+)?(.+)",
        r"save\s+(?:this|that)?:?\s*(.+)",
        r"note\s+(?:that\s+)?(.+)",
        r"keep in mind\s+(?:that\s+)?(.+)",
    ]

    text_lower = text.lower().strip()

    for pattern in patterns:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Generate key from value
            key = f"explicit_{hash(value) & 0xFFFFFF:06x}"
            return (key, value)

    return None


def detect_recall_intent(text: str) -> bool:
    """
    Check if user is asking to recall something.

    Args:
        text: User message

    Returns:
        True if user wants to recall memory
    """
    patterns = [
        r"what do you remember",
        r"what do you know about me",
        r"what's my name",
        r"recall",
        r"show memory",
    ]

    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)
