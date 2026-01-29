"""
Memory manager for PocketCoder.

Central class for all memory operations:
- Fact storage with TTL
- Auto-extraction from conversations
- Smart Memory Architecture v0.6.0 (POINTER/VALUE/REF)
- Injection into prompts
- Cleanup of expired facts
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pocketcoder.core.memory.types import Fact, FactType, MemoryStats, DEFAULT_TTL
from pocketcoder.core.memory.storage import MemoryStorage
from pocketcoder.core.memory.extractor import (
    FactExtractor,
    extract_explicit_memory_request,
    detect_recall_intent,
)
from pocketcoder.core.memory.classifier import FactClassifier, ExtractionResult


# How often to run cleanup (every N requests)
CLEANUP_INTERVAL = 50


class MemoryManager:
    """
    Central manager for PocketCoder memory system.

    Provides:
    - save_fact(): Manual fact saving
    - auto_extract(): Automatic extraction from conversations
    - get_all_facts(): Get all active (non-expired) facts
    - build_memory_context(): Format facts for prompt injection
    - cleanup(): Remove expired facts
    """

    def __init__(self, storage: Optional[MemoryStorage] = None):
        """
        Initialize memory manager.

        Args:
            storage: Override storage backend (default: MemoryStorage)
        """
        self.storage = storage or MemoryStorage()
        self.extractor = FactExtractor()
        self.classifier = FactClassifier()  # v0.6.0: Smart Memory
        self._maybe_cleanup()

    def _maybe_cleanup(self) -> None:
        """Run cleanup if needed based on request count."""
        meta = self.storage.get_meta()
        requests = meta.get("requests_since_cleanup", 0)

        if requests >= CLEANUP_INTERVAL:
            removed = self.storage.cleanup_expired()
            if removed > 0:
                pass  # Could log this
        else:
            self.storage.set_meta("requests_since_cleanup", requests + 1)

    # =========================================================================
    # Core Operations
    # =========================================================================

    def save_fact(
        self,
        key: str,
        value: str,
        fact_type: FactType = FactType.VALUE,
        category: str = "explicit",
        source: str = "explicit",
        ttl_days: Optional[int] = None,
    ) -> str:
        """
        Save a fact to memory.

        Args:
            key: Unique identifier for the fact
            value: The fact content (command for POINTER, value for VALUE, url|summary for REF)
            fact_type: POINTER, VALUE, or REF (v0.6.0 Smart Memory)
            category: Fact category (identity, preference, project, task, context, error, external)
            source: How it was created (auto, explicit, llm, tool)
            ttl_days: Override TTL (None = use category default)

        Returns:
            Confirmation message
        """
        # Use category default TTL if not specified
        if ttl_days is None:
            ttl_days = DEFAULT_TTL.get(category)

        fact = Fact(
            key=key,
            value=value,
            fact_type=fact_type,
            category=category,
            created=datetime.now(),
            ttl_days=ttl_days,
            source=source,
        )

        self.storage.save_fact(fact)

        type_icon = {"pointer": "→", "value": "=", "ref": "🔗"}.get(fact_type.value, "=")
        ttl_info = f" ({ttl_days}d)" if ttl_days else ""
        return f"✓ [{fact_type.value}] {key} {type_icon} {value}{ttl_info}"

    def get_fact(self, key: str) -> Optional[Fact]:
        """
        Get a single fact by key.

        Args:
            key: Fact key

        Returns:
            Fact or None if not found/expired
        """
        facts = self.storage.load_facts()
        fact = facts.get(key)

        if fact and not fact.is_expired():
            return fact
        return None

    def get_all_facts(self) -> dict[str, Fact]:
        """
        Get all active (non-expired) facts.

        Returns:
            Dict of key -> Fact
        """
        facts = self.storage.load_facts()
        return {k: f for k, f in facts.items() if not f.is_expired()}

    def delete_fact(self, key: str) -> str:
        """
        Delete a fact.

        Args:
            key: Fact key to delete

        Returns:
            Result message
        """
        if self.storage.delete_fact(key):
            return f"🗑️ Forgot: {key}"
        return f"❌ No memory for key: {key}"

    def list_facts(self) -> str:
        """
        List all facts formatted for display.

        Returns:
            Formatted string
        """
        facts = self.get_all_facts()

        if not facts:
            return "📭 Memory is empty"

        lines = ["📚 Remembered facts:"]

        # Group by category
        by_category: dict[str, list] = {}
        for key, fact in facts.items():
            cat = fact.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((key, fact))

        for category, items in sorted(by_category.items()):
            lines.append(f"\n  [{category}]")
            for key, fact in items:
                ttl_info = ""
                if fact.ttl_days is not None:
                    remaining = fact.days_until_expiry()
                    if remaining is not None and remaining <= 7:
                        ttl_info = f" ({remaining}d left)"
                lines.append(f"    • {key}: {fact.value}{ttl_info}")

        return "\n".join(lines)

    # =========================================================================
    # Auto-Extraction
    # =========================================================================

    def auto_extract(
        self,
        user_message: str,
        llm_response: str,
    ) -> dict[str, Fact]:
        """
        Automatically extract and save facts from conversation.

        Called after every LLM response to capture information
        without relying on LLM tool calls.

        Args:
            user_message: What user said
            llm_response: What LLM replied

        Returns:
            Dict of newly saved facts
        """
        # 1. Check for explicit memory request
        explicit = extract_explicit_memory_request(user_message)
        if explicit:
            key, value = explicit
            self.save_fact(key, value, category="explicit", source="explicit")
            return {key: self.get_fact(key)}

        # 2. Auto-extract from conversation
        extracted = self.extractor.extract_from_conversation(
            user_message,
            llm_response
        )

        # 3. Save new facts (don't overwrite existing)
        existing = self.get_all_facts()
        new_facts = {}

        for key, fact in extracted.items():
            if key not in existing:
                self.storage.save_fact(fact)
                new_facts[key] = fact

        return new_facts

    def check_recall_intent(self, user_message: str) -> bool:
        """
        Check if user wants to recall memory.

        Args:
            user_message: User input

        Returns:
            True if user is asking about memory
        """
        return detect_recall_intent(user_message)

    # =========================================================================
    # Prompt Injection
    # =========================================================================

    def build_memory_context(self, max_facts: int = 20) -> str:
        """
        Build memory context string for prompt injection.

        v0.7.0 Smart Memory Architecture:
        - KNOWLEDGE ACCESS: Commands to get data (POINTER)
        - WORKING CONTEXT: Current paths, read files (VALUE, context category)
        - PROJECT MEMORY: Saved values (VALUE, other categories)
        - EXTERNAL REFS: URLs and references (REF)

        Args:
            max_facts: Maximum facts to include

        Returns:
            Formatted string for system prompt (~150-200 tokens)
        """
        facts = self.get_all_facts()

        if not facts:
            return ""

        # Separate by fact_type and category
        pointers = []
        working_context = []  # v0.7.0: paths, read files
        values = []
        refs = []

        for key, fact in facts.items():
            if fact.fact_type == FactType.POINTER:
                pointers.append((key, fact))
            elif fact.fact_type == FactType.REF:
                refs.append((key, fact))
            elif fact.category == "context" or key.startswith("file_read:") or key.startswith("file_summary:"):
                working_context.append((key, fact))
            elif key in ("working_project", "last_created_folder", "last_created_file", "visible_folders"):
                working_context.append((key, fact))
            else:
                values.append((key, fact))

        sections = []

        # 1. KNOWLEDGE ACCESS (commands to get data)
        if pointers:
            lines = ["## KNOWLEDGE ACCESS", "Commands to get current data:"]
            for key, fact in pointers[:8]:
                lines.append(f"- {key}: `{fact.value}`")
            sections.append("\n".join(lines))

        # 2. WORKING CONTEXT (v0.7.0: current session paths and files)
        if working_context:
            lines = ["## WORKING CONTEXT", "Current session state:"]

            # Group by type for readability
            paths = [(k, f) for k, f in working_context
                     if k in ("working_project", "last_created_folder", "last_created_file", "visible_folders")]
            files_read = [(k, f) for k, f in working_context if k.startswith("file_read:")]

            if paths:
                for key, fact in paths[:5]:
                    lines.append(f"- {key}: {fact.value}")

            if files_read:
                # Compact format: list read files
                read_paths = [k.replace("file_read:", "") for k, _ in files_read[:10]]
                lines.append(f"- files_already_read: {', '.join(read_paths)}")
                lines.append("  ⚠️ Do NOT call read_file on these again!")

            sections.append("\n".join(lines))

        # 3. PROJECT MEMORY (saved values)
        if values:
            priority = ["error", "project", "identity", "preference", "task", "explicit", "auto"]
            sorted_values = sorted(
                values,
                key=lambda x: (
                    priority.index(x[1].category) if x[1].category in priority else 99,
                    x[0]
                )
            )[:max_facts - len(pointers) - len(working_context)]

            if sorted_values:
                lines = ["## PROJECT MEMORY"]
                for key, fact in sorted_values:
                    lines.append(f"- {key}: {fact.value}")
                sections.append("\n".join(lines))

        # 4. EXTERNAL REFS (URLs, docs)
        if refs:
            lines = ["## EXTERNAL REFS"]
            for key, fact in refs[:5]:
                lines.append(f"- {key}: {fact.value}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # =========================================================================
    # Tool Results Extraction (v0.7.0)
    # =========================================================================

    def extract_from_tool_results(
        self,
        tool_results: List[dict]
    ) -> List[str]:
        """
        Extract and save facts from tool results.

        v0.7.0 Smart Memory: Uses classify_tool_result_multi for
        extracting MULTIPLE facts per tool call:
        - write_file → last_created_file, last_created_folder, working_project
        - list_files → project_structure (POINTER), visible_folders, files_count
        - read_file → file_read:{path}, file_summary:{path}
        - execute_command → errors, metrics, POINTER patterns

        Args:
            tool_results: List of {"name": str, "args": dict, "result": str}

        Returns:
            List of messages about what was saved
        """
        messages = []

        for tr in tool_results:
            tool_name = tr.get("name", "")
            args = tr.get("args", {})
            result = tr.get("result", "")

            # Skip empty results
            if not result:
                continue

            # v0.7.0: Get MULTIPLE extractions per tool
            extractions = self.classifier.classify_tool_result_multi(
                tool_name, args, result
            )

            for extraction in extractions:
                if not extraction.should_save:
                    continue

                # Check if we already have this fact
                existing = self.get_fact(extraction.key)

                # Decide whether to save/update
                should_save = self._should_save_extraction(extraction, existing)

                if should_save:
                    msg = self.save_fact(
                        key=extraction.key,
                        value=extraction.value,
                        fact_type=extraction.fact_type,
                        category=extraction.category,
                        source="tool",
                        ttl_days=extraction.ttl_days,  # v0.7.0: custom TTL
                    )
                    messages.append(msg)

        return messages

    def _should_save_extraction(
        self,
        extraction,
        existing: Optional[Fact]
    ) -> bool:
        """
        Determine if extraction should be saved.

        Rules:
        - POINTER: always update (command might have changed)
        - VALUE with "last_" prefix: always update (most recent wins)
        - VALUE with "file_read:" prefix: don't overwrite (already read)
        - Other VALUE: only save if new
        """
        # New fact → save
        if existing is None:
            return True

        # POINTER → always update
        if extraction.fact_type == FactType.POINTER:
            return True

        # "last_" facts → always update (most recent wins)
        if extraction.key.startswith("last_"):
            return True

        # "file_read:" → don't overwrite (file already read in this session)
        if extraction.key.startswith("file_read:"):
            return False

        # "working_project" → update if different
        if extraction.key == "working_project":
            return extraction.value != existing.value

        # Default: don't overwrite existing
        return False

    # =========================================================================
    # Maintenance
    # =========================================================================

    def cleanup(self) -> int:
        """
        Force cleanup of expired facts.

        Returns:
            Number of facts removed
        """
        return self.storage.cleanup_expired()

    def get_stats(self) -> MemoryStats:
        """
        Get memory statistics.

        Returns:
            MemoryStats object
        """
        return self.storage.get_stats()
