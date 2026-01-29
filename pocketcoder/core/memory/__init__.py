"""
Memory & Context system for PocketCoder.

v0.7.0 Smart Memory Architecture:
- POINTER: commands to get data (pip freeze, list_files)
- VALUE: saved values (errors, decisions, paths, read files)
- REF: external references (URL + summary)

v0.7.0 Chat Storage:
- Raw chat persistence in JSONL format
- Grep-based search for context retrieval
- Keyword extraction for automatic context injection

Provides automatic fact extraction, storage with TTL,
and prompt injection for persistent memory across sessions.

Usage:
    from pocketcoder.core.memory import MemoryManager, FactType, ChatStorage

    mm = MemoryManager()

    # Save a POINTER (command to get data)
    mm.save_fact("pip_packages", "pip freeze", fact_type=FactType.POINTER)

    # Save a VALUE (actual data)
    mm.save_fact("last_error", "TypeError: ...", fact_type=FactType.VALUE)

    # Auto-extract from tool results
    mm.extract_from_tool_results(tool_results)

    # Build context for prompt injection (~150-200 tokens)
    context = mm.build_memory_context()

    # Chat storage for grep-based retrieval
    chat = ChatStorage()
    chat.append("user", "create folder test")
    chat.grep("test")  # Find messages about "test"
"""

from pocketcoder.core.memory.types import Fact, FactType, MemoryStats, DEFAULT_TTL
from pocketcoder.core.memory.storage import MemoryStorage, get_project_id
from pocketcoder.core.memory.extractor import (
    FactExtractor,
    extract_explicit_memory_request,
    detect_recall_intent,
)
from pocketcoder.core.memory.classifier import FactClassifier, ExtractionResult
from pocketcoder.core.memory.manager import MemoryManager
from pocketcoder.core.memory.chat_storage import (
    ChatStorage,
    get_session_dir,
    extract_keywords,
)


__all__ = [
    # Main class
    "MemoryManager",
    # Types (v0.6.0)
    "Fact",
    "FactType",
    "MemoryStats",
    "DEFAULT_TTL",
    # Classifier (v0.7.0)
    "FactClassifier",
    "ExtractionResult",
    # Storage
    "MemoryStorage",
    "get_project_id",
    # Chat Storage (v0.7.0)
    "ChatStorage",
    "get_session_dir",
    "extract_keywords",
    # Extractor
    "FactExtractor",
    "extract_explicit_memory_request",
    "detect_recall_intent",
]
