"""
RepoMap module - project context generation.

Provides repository structure and symbol extraction for LLM context.
Two backends:
- simple: Regex-based (default, no dependencies)
- treesitter: tree-sitter based (optional, more accurate)
"""

from pocketcoder.repomap.simple import SimpleRepoMap

__all__ = ["SimpleRepoMap"]
