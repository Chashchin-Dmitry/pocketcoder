"""
Memory storage for PocketCoder.

Handles JSON persistence with versioning and cleanup.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from pocketcoder.core.memory.types import Fact, MemoryStats


# Current memory format version
MEMORY_VERSION = "1.0"


class MemoryStorage:
    """
    Persistent storage for facts.

    v2.0.0: Changed from global ~/.pocketcoder/ to per-project .pocketcoder/
    Stores facts in .pocketcoder/memory.json with:
    - Version tracking
    - TTL-based cleanup
    - Project-specific notes
    """

    def __init__(self, base_path: Optional[Path] = None, per_project: bool = True):
        """
        Initialize storage.

        Args:
            base_path: Override base path
            per_project: If True, use .pocketcoder/ in cwd (v2.0.0 default)
                        If False, use global ~/.pocketcoder/
        """
        if base_path:
            self.base_path = base_path
        elif per_project:
            # v2.0.0: Per-project storage
            self.base_path = Path.cwd() / ".pocketcoder"
        else:
            # Legacy: global storage
            self.base_path = Path.home() / ".pocketcoder"

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.base_path / "memory.json"

    def _load_raw(self) -> dict:
        """Load raw JSON data."""
        if not self.memory_file.exists():
            return {
                "version": MEMORY_VERSION,
                "facts": {},
                "meta": {
                    "created": datetime.now().isoformat(),
                    "last_cleanup": None,
                    "requests_since_cleanup": 0,
                },
            }

        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            # Migrate old format if needed
            if "version" not in data:
                data = self._migrate_v0(data)
            return data
        except (json.JSONDecodeError, IOError):
            return {
                "version": MEMORY_VERSION,
                "facts": {},
                "meta": {"created": datetime.now().isoformat()},
            }

    def _save_raw(self, data: dict) -> None:
        """Save raw JSON data."""
        data["version"] = MEMORY_VERSION
        self.memory_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _migrate_v0(self, old_data: dict) -> dict:
        """Migrate from old format (just key: {value, created})."""
        new_data = {
            "version": MEMORY_VERSION,
            "facts": {},
            "meta": {"created": datetime.now().isoformat()},
        }

        for key, value in old_data.items():
            if isinstance(value, dict) and "value" in value:
                new_data["facts"][key] = {
                    "value": value["value"],
                    "category": "explicit",
                    "created": value.get("created", datetime.now().isoformat()),
                    "ttl_days": None,
                    "source": "explicit",
                }

        return new_data

    def load_facts(self) -> dict[str, Fact]:
        """
        Load all facts as Fact objects.

        Returns:
            Dict of key -> Fact
        """
        data = self._load_raw()
        facts = {}

        for key, fact_data in data.get("facts", {}).items():
            facts[key] = Fact.from_dict(key, fact_data)

        return facts

    def save_facts(self, facts: dict[str, Fact]) -> None:
        """
        Save all facts.

        Args:
            facts: Dict of key -> Fact
        """
        data = self._load_raw()
        data["facts"] = {key: fact.to_dict() for key, fact in facts.items()}
        self._save_raw(data)

    def save_fact(self, fact: Fact) -> None:
        """
        Save a single fact.

        Args:
            fact: Fact to save
        """
        data = self._load_raw()
        data["facts"][fact.key] = fact.to_dict()
        self._save_raw(data)

    def delete_fact(self, key: str) -> bool:
        """
        Delete a fact by key.

        Args:
            key: Fact key to delete

        Returns:
            True if deleted, False if not found
        """
        data = self._load_raw()
        if key in data.get("facts", {}):
            del data["facts"][key]
            self._save_raw(data)
            return True
        return False

    def get_meta(self) -> dict:
        """Get metadata."""
        data = self._load_raw()
        return data.get("meta", {})

    def set_meta(self, key: str, value) -> None:
        """Set metadata value."""
        data = self._load_raw()
        if "meta" not in data:
            data["meta"] = {}
        data["meta"][key] = value
        self._save_raw(data)

    def cleanup_expired(self) -> int:
        """
        Remove expired facts.

        Returns:
            Number of facts removed
        """
        facts = self.load_facts()
        initial_count = len(facts)

        # Filter out expired
        active = {k: f for k, f in facts.items() if not f.is_expired()}

        removed = initial_count - len(active)
        if removed > 0:
            self.save_facts(active)
            self.set_meta("last_cleanup", datetime.now().isoformat())

        self.set_meta("requests_since_cleanup", 0)
        return removed

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        facts = self.load_facts()
        meta = self.get_meta()

        stats = MemoryStats(
            total_facts=len(facts),
            permanent_facts=sum(1 for f in facts.values() if f.ttl_days is None),
            expiring_soon=sum(1 for f in facts.values()
                            if f.days_until_expiry() is not None
                            and f.days_until_expiry() <= 3),
            by_category={},
        )

        # Count by category
        for fact in facts.values():
            cat = fact.category
            stats.by_category[cat] = stats.by_category.get(cat, 0) + 1

        # Last cleanup
        if meta.get("last_cleanup"):
            stats.last_cleanup = datetime.fromisoformat(meta["last_cleanup"])

        return stats


def get_project_id(project_path: str) -> str:
    """
    Generate stable ID for project.

    Args:
        project_path: Path to project root

    Returns:
        12-char hash ID
    """
    import os
    normalized = os.path.abspath(project_path)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]
