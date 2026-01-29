"""
Simple regex-based RepoMap implementation.

Extracts function and class definitions using regex patterns.
No external dependencies required.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Symbol:
    """A code symbol (function, class, method)."""

    name: str
    type: str  # "function" | "class" | "method"
    file: Path
    line: int


# Regex patterns for different languages
PATTERNS: dict[str, list[tuple[str, str]]] = {
    ".py": [
        (r"^def\s+(\w+)", "function"),
        (r"^class\s+(\w+)", "class"),
        (r"^\s+def\s+(\w+)", "method"),
    ],
    ".js": [
        (r"^function\s+(\w+)", "function"),
        (r"^class\s+(\w+)", "class"),
        (r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
        (r"^let\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
        (r"^\s+(\w+)\s*\([^)]*\)\s*\{", "method"),
    ],
    ".ts": [
        (r"^function\s+(\w+)", "function"),
        (r"^class\s+(\w+)", "class"),
        (r"^const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
        (r"^export\s+function\s+(\w+)", "function"),
        (r"^export\s+class\s+(\w+)", "class"),
        (r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)", "method"),
    ],
    ".go": [
        (r"^func\s+(\w+)", "function"),
        (r"^func\s+\([^)]+\)\s+(\w+)", "method"),
        (r"^type\s+(\w+)\s+struct", "class"),
        (r"^type\s+(\w+)\s+interface", "class"),
    ],
    ".rs": [
        (r"^pub\s+fn\s+(\w+)", "function"),
        (r"^fn\s+(\w+)", "function"),
        (r"^pub\s+struct\s+(\w+)", "class"),
        (r"^struct\s+(\w+)", "class"),
        (r"^\s+pub\s+fn\s+(\w+)", "method"),
        (r"^\s+fn\s+(\w+)", "method"),
    ],
    ".java": [
        (r"^\s*public\s+class\s+(\w+)", "class"),
        (r"^\s*class\s+(\w+)", "class"),
        (r"^\s*public\s+\w+\s+(\w+)\s*\(", "method"),
        (r"^\s*private\s+\w+\s+(\w+)\s*\(", "method"),
    ],
}


class SimpleRepoMap:
    """
    Simple regex-based repository map generator.

    Extracts symbols (functions, classes, methods) from source files
    using regex patterns.
    """

    def __init__(self, root: Path | None = None):
        self.root = root or Path.cwd()
        self.cache: dict[Path, tuple[float, list[Symbol]]] = {}

    def extract_symbols(self, path: Path) -> list[Symbol]:
        """
        Extract symbols from a single file.

        Args:
            path: Path to source file

        Returns:
            List of Symbol objects
        """
        patterns = PATTERNS.get(path.suffix, [])
        if not patterns:
            return []

        try:
            content = path.read_text()
        except Exception:
            return []

        symbols = []

        for i, line in enumerate(content.splitlines(), 1):
            for pattern, sym_type in patterns:
                match = re.match(pattern, line)
                if match:
                    symbols.append(
                        Symbol(
                            name=match.group(1),
                            type=sym_type,
                            file=path,
                            line=i,
                        )
                    )
                    break  # One match per line

        return symbols

    def get_symbols(self, files: list[Path] | None = None) -> list[Symbol]:
        """
        Get symbols from multiple files with caching.

        Args:
            files: List of files (defaults to all source files in root)

        Returns:
            List of all Symbol objects
        """
        if files is None:
            files = self._find_source_files()

        all_symbols = []

        for f in files:
            if not f.exists():
                continue

            mtime = f.stat().st_mtime

            # Check cache
            if f in self.cache and self.cache[f][0] == mtime:
                all_symbols.extend(self.cache[f][1])
            else:
                symbols = self.extract_symbols(f)
                self.cache[f] = (mtime, symbols)
                all_symbols.extend(symbols)

        return all_symbols

    def _find_source_files(self) -> list[Path]:
        """Find all source files in root directory."""
        files = []
        extensions = list(PATTERNS.keys())

        for ext in extensions:
            files.extend(self.root.rglob(f"*{ext}"))

        # Filter out common ignore patterns
        ignore = {"node_modules", "__pycache__", ".git", "venv", ".venv"}
        files = [
            f
            for f in files
            if not any(part in ignore for part in f.parts)
        ]

        return files

    def format_map(self, max_tokens: int = 2000) -> str:
        """
        Generate formatted repository map for LLM context.

        Args:
            max_tokens: Approximate max tokens (chars / 4)

        Returns:
            Formatted string with repository structure
        """
        symbols = self.get_symbols()

        if not symbols:
            return ""

        # Group by file
        by_file: dict[Path, list[Symbol]] = defaultdict(list)
        for s in symbols:
            by_file[s.file].append(s)

        lines = ["# Repository Structure\n"]

        for file_path in sorted(by_file.keys()):
            syms = by_file[file_path]

            try:
                rel_path = file_path.relative_to(self.root)
            except ValueError:
                rel_path = file_path

            lines.append(f"\n## {rel_path}")

            for s in syms:
                if s.type == "class":
                    lines.append(f"- class {s.name}")
                elif s.type == "function":
                    lines.append(f"- def {s.name}()")
                elif s.type == "method":
                    lines.append(f"  - {s.name}()")

        result = "\n".join(lines)

        # Truncate if too long
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"

        return result
