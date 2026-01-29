"""
Memory types for PocketCoder.

Dataclasses for facts with TTL support.
Includes Smart Memory Architecture v0.6.0:
- POINTER: command to retrieve data (pip freeze, list_files)
- VALUE: actual value (errors, solutions, ML metrics)
- REF: external reference (URL + summary, PDF path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class FactType(Enum):
    """
    Type of fact determining how it's stored and used.

    POINTER: Store command to retrieve data (saves tokens, always fresh)
        Example: pip_packages → "pip freeze"

    VALUE: Store actual value (for non-reproducible facts)
        Example: last_error → "TypeError in main.py:42"

    REF: External reference (URL + summary, not full content)
        Example: nginx_guide → {url: "...", summary: "..."}
    """
    POINTER = "pointer"
    VALUE = "value"
    REF = "ref"


# Default TTL by category (days)
DEFAULT_TTL = {
    "identity": None,      # permanent (name, email, preferences)
    "preference": None,    # permanent (editor, language)
    "project": 30,         # 30 days (tech stack, architecture)
    "task": 7,             # 7 days (current work)
    "context": 1,          # 1 day (recent discussions)
    "explicit": None,      # permanent (user explicitly asked to remember)
    "auto": 14,            # 14 days (auto-extracted)
}


@dataclass
class Fact:
    """
    A single fact with metadata.

    Attributes:
        key: Unique identifier (e.g., "user_name", "pip_packages")
        value: The fact content (command for POINTER, value for VALUE, url+summary for REF)
        fact_type: POINTER, VALUE, or REF (determines how it's used)
        category: Category (identity, preference, project, task, context, error, external)
        created: When the fact was created
        ttl_days: Days until expiration (None = permanent)
        source: How it was created (auto, explicit, llm, tool)

    Examples:
        POINTER: Fact("pip_packages", "pip freeze", FactType.POINTER, "project")
        VALUE: Fact("last_error", "TypeError in main.py:42", FactType.VALUE, "error")
        REF: Fact("nginx_guide", "https://... | Official SSL guide", FactType.REF, "external")
    """
    key: str
    value: str
    fact_type: FactType = FactType.VALUE  # Default to VALUE for backward compat
    category: str = "auto"
    created: datetime = field(default_factory=datetime.now)
    ttl_days: Optional[int] = None
    source: str = "auto"  # "auto" | "explicit" | "llm" | "tool"

    def __post_init__(self):
        """Set default TTL based on category if not specified."""
        if self.ttl_days is None and self.category in DEFAULT_TTL:
            self.ttl_days = DEFAULT_TTL[self.category]

    def is_expired(self) -> bool:
        """Check if fact has expired based on TTL."""
        if self.ttl_days is None:
            return False  # Permanent

        age = datetime.now() - self.created
        return age > timedelta(days=self.ttl_days)

    def days_until_expiry(self) -> Optional[int]:
        """Get days until this fact expires."""
        if self.ttl_days is None:
            return None

        age = datetime.now() - self.created
        remaining = timedelta(days=self.ttl_days) - age
        return max(0, remaining.days)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "value": self.value,
            "fact_type": self.fact_type.value,  # Store as string
            "category": self.category,
            "created": self.created.isoformat(),
            "ttl_days": self.ttl_days,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, key: str, data: dict) -> "Fact":
        """Create Fact from dictionary."""
        created = data.get("created")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif created is None:
            created = datetime.now()

        # Parse fact_type (backward compat: default to VALUE)
        fact_type_str = data.get("fact_type", "value")
        try:
            fact_type = FactType(fact_type_str)
        except ValueError:
            fact_type = FactType.VALUE

        return cls(
            key=key,
            value=data.get("value", ""),
            fact_type=fact_type,
            category=data.get("category", "auto"),
            created=created,
            ttl_days=data.get("ttl_days"),
            source=data.get("source", "auto"),
        )


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    total_facts: int = 0
    permanent_facts: int = 0
    expiring_soon: int = 0  # Within 3 days
    by_category: dict = field(default_factory=dict)
    last_cleanup: Optional[datetime] = None
