"""
RepoMap — Repository structure map for LLM context.

v2.6.0: Provides code structure overview without reading full files.

Gears:
  N (Neutral): Empty project → "(empty project)"
  1 (Small):   ≤10 files → full signatures for all
  2 (Medium):  ≤50 files → structure + key signatures
  3 (Large):   >50 files → folder tree + entry points only

Uses grep-ast (tree-sitter wrapper) when available, falls back to Python AST.

Causal chain:
    Project path → scan files (with ignore) → extract structure → format for XML

Usage:
    repo_map = RepoMapBuilder(project_path)
    xml_block = repo_map.build()  # Returns <repo_map>...</repo_map>

    # Force rebuild after file changes:
    repo_map.invalidate()
    xml_block = repo_map.build()
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Set
from fnmatch import fnmatch


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RepoMapConfig:
    """
    Configuration for repo map generation — the gear ratios.

    Attributes:
        small_threshold: Max files for Gear 1 (full signatures)
        medium_threshold: Max files for Gear 2 (structure + key)
        max_signatures_per_file: Limit signatures in Gear 2/3
        ignore_patterns: Glob patterns to ignore
    """
    small_threshold: int = 10
    medium_threshold: int = 50
    max_signatures_per_file: int = 10
    ignore_patterns: List[str] = None

    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = DEFAULT_IGNORE_PATTERNS


# Default ignore patterns (like .gitignore but smarter)
DEFAULT_IGNORE_PATTERNS = [
    # Dependencies
    "venv/", ".venv/", "env/", ".env/",
    "node_modules/",
    "vendor/",
    ".tox/",

    # Cache
    "__pycache__/", "*.pyc", "*.pyo",
    ".cache/", ".pytest_cache/",
    ".mypy_cache/", ".ruff_cache/",
    "*.egg-info/",

    # Version control
    ".git/",
    ".hg/",
    ".svn/",

    # IDE
    ".idea/", ".vscode/",
    "*.swp", "*.swo",

    # Build artifacts
    "dist/", "build/",
    "*.so", "*.dylib", "*.dll",

    # Data files (usually large)
    "*.csv", "*.parquet", "*.feather",
    "*.db", "*.sqlite", "*.sqlite3",
    "*.pkl", "*.pickle",

    # Archives
    "*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.rar",

    # Media
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico",
    "*.mp3", "*.mp4", "*.wav",
    "*.pdf",

    # Logs
    "*.log",

    # PocketCoder internal
    ".pocketcoder/",
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class FunctionSignature:
    """A function/method signature extracted from code."""
    name: str
    args: List[str]
    returns: Optional[str] = None
    decorators: List[str] = None
    line: int = 0

    def format(self, indent: str = "") -> str:
        """Format as readable signature."""
        args_str = ", ".join(self.args) if self.args else ""
        ret_str = f" → {self.returns}" if self.returns else ""
        dec_str = ""
        if self.decorators:
            dec_str = " ".join(f"@{d}" for d in self.decorators[:2]) + " "
        return f"{indent}{dec_str}{self.name}({args_str}){ret_str}"


@dataclass
class ClassStructure:
    """A class structure extracted from code."""
    name: str
    methods: List[FunctionSignature]
    decorators: List[str] = None
    bases: List[str] = None
    line: int = 0

    def format(self, indent: str = "") -> str:
        """Format as readable class summary."""
        lines = []
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines.append(f"{indent}class {self.name}{bases_str}:")
        for method in self.methods[:10]:  # Limit methods shown
            lines.append(method.format(indent + "  ."))
        if len(self.methods) > 10:
            lines.append(f"{indent}  ... ({len(self.methods) - 10} more methods)")
        return "\n".join(lines)


@dataclass
class FileStructure:
    """Structure extracted from a single file."""
    path: str
    language: str
    classes: List[ClassStructure]
    functions: List[FunctionSignature]
    imports: List[str]
    level: str = "full"  # full | partial | filename_only
    error: str = ""

    def format(self, indent: str = "") -> str:
        """Format as readable file summary."""
        lines = [f"{indent}{self.path}"]

        if self.error:
            lines.append(f"{indent}  (error: {self.error})")
            return "\n".join(lines)

        if self.level == "filename_only":
            return lines[0]

        for cls in self.classes:
            lines.append(cls.format(indent + "  "))

        for func in self.functions:
            lines.append(func.format(indent + "  "))

        return "\n".join(lines)


# =============================================================================
# Extractors
# =============================================================================

class PythonASTExtractor:
    """
    Extract structure from Python files using built-in ast module.

    Fallback when grep-ast is not available.
    """

    def extract(self, file_path: Path) -> FileStructure:
        """
        Extract structure from Python file.

        Degradation levels:
          1. Full extraction (classes, methods, types)
          2. Partial extraction (only names, via regex)
          3. Filename only (on error)
        """
        rel_path = str(file_path)

        # Try to read file
        try:
            code = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                code = file_path.read_text(encoding='latin-1')
            except Exception as e:
                return FileStructure(
                    path=rel_path, language="python",
                    classes=[], functions=[], imports=[],
                    level="filename_only", error=f"encoding: {e}"
                )
        except PermissionError:
            return FileStructure(
                path=rel_path, language="python",
                classes=[], functions=[], imports=[],
                level="filename_only", error="permission denied"
            )
        except Exception as e:
            return FileStructure(
                path=rel_path, language="python",
                classes=[], functions=[], imports=[],
                level="filename_only", error=str(e)
            )

        # Try AST parsing
        try:
            tree = ast.parse(code)
            return self._full_extraction(tree, rel_path)
        except SyntaxError as e:
            # Fallback to regex
            return self._regex_fallback(code, rel_path, error=f"syntax: {e.msg}")
        except Exception as e:
            return FileStructure(
                path=rel_path, language="python",
                classes=[], functions=[], imports=[],
                level="filename_only", error=str(e)
            )

    def _full_extraction(self, tree: ast.AST, rel_path: str) -> FileStructure:
        """Full AST extraction with types and decorators."""
        classes = []
        functions = []
        imports = []

        for node in ast.iter_child_nodes(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)

            # Classes
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node)
                classes.append(cls)

            # Top-level functions
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func = self._extract_function(node)
                functions.append(func)

        return FileStructure(
            path=rel_path,
            language="python",
            classes=classes,
            functions=functions,
            imports=imports[:10],  # Limit imports
            level="full"
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassStructure:
        """Extract class structure."""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private methods except __init__
                if item.name.startswith('_') and item.name != '__init__':
                    continue
                methods.append(self._extract_function(item))

        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except:
                pass

        decorators = [self._get_decorator_name(d) for d in node.decorator_list[:3]]

        return ClassStructure(
            name=node.name,
            methods=methods,
            decorators=decorators,
            bases=bases[:3],
            line=node.lineno
        )

    def _extract_function(self, node) -> FunctionSignature:
        """Extract function signature."""
        args = []
        for arg in node.args.args:
            if arg.arg == 'self' or arg.arg == 'cls':
                continue
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args.append(arg_str)

        returns = None
        if node.returns:
            try:
                returns = ast.unparse(node.returns)
            except:
                pass

        decorators = [self._get_decorator_name(d) for d in node.decorator_list[:2]]

        return FunctionSignature(
            name=node.name,
            args=args,
            returns=returns,
            decorators=decorators,
            line=node.lineno
        )

    def _get_decorator_name(self, node) -> str:
        """Get decorator name as string."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{ast.unparse(node.value)}.{node.attr}"
            elif isinstance(node, ast.Call):
                return self._get_decorator_name(node.func)
            else:
                return ast.unparse(node)
        except:
            return "?"

    def _regex_fallback(self, code: str, rel_path: str, error: str = "") -> FileStructure:
        """
        Fallback when AST fails.

        Uses regex to find class/def names even in broken code.
        """
        classes = []
        functions = []

        # Find class definitions
        for match in re.finditer(r'^class\s+(\w+)', code, re.MULTILINE):
            classes.append(ClassStructure(
                name=match.group(1),
                methods=[],
                line=code[:match.start()].count('\n') + 1
            ))

        # Find function definitions (top-level only, no indent)
        for match in re.finditer(r'^def\s+(\w+)\s*\(', code, re.MULTILINE):
            functions.append(FunctionSignature(
                name=match.group(1),
                args=[],
                line=code[:match.start()].count('\n') + 1
            ))

        return FileStructure(
            path=rel_path,
            language="python",
            classes=classes,
            functions=functions,
            imports=[],
            level="partial",
            error=error
        )


class GrepASTExtractor:
    """
    Extract structure using grep-ast (tree-sitter wrapper).

    Supports multiple languages, more accurate than Python AST.
    """

    def __init__(self):
        self._available = None
        self._parsers: Dict[str, any] = {}

    @property
    def available(self) -> bool:
        """Check if grep-ast is available."""
        if self._available is None:
            try:
                from grep_ast import filename_to_lang
                from grep_ast.tsl import get_parser
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def get_language(self, file_path: Path) -> Optional[str]:
        """Get language for file."""
        if not self.available:
            return None
        try:
            from grep_ast import filename_to_lang
            return filename_to_lang(str(file_path))
        except:
            return None

    def extract(self, file_path: Path) -> Optional[FileStructure]:
        """
        Extract structure using tree-sitter.

        Returns None if extraction fails, caller should fallback.
        """
        if not self.available:
            return None

        try:
            from grep_ast import filename_to_lang
            from grep_ast.tsl import get_parser, get_language

            lang = filename_to_lang(str(file_path))
            if not lang:
                return None

            parser = get_parser(lang)
            code = file_path.read_bytes()
            tree = parser.parse(code)

            # Extract based on language
            if lang == "python":
                return self._extract_python(tree, str(file_path), lang)
            else:
                return self._extract_generic(tree, str(file_path), lang)

        except Exception as e:
            return None  # Caller will fallback

    def _extract_python(self, tree, rel_path: str, lang: str) -> FileStructure:
        """Extract from Python using tree-sitter."""
        # For now, delegate to AST extractor since it works well for Python
        # tree-sitter is more useful for other languages
        return None

    def _extract_generic(self, tree, rel_path: str, lang: str) -> FileStructure:
        """Generic extraction for non-Python languages."""
        # Walk tree and find definitions
        classes = []
        functions = []

        def walk(node, depth=0):
            # Look for common definition patterns
            node_type = node.type

            # Class-like definitions
            if 'class' in node_type and 'definition' in node_type:
                name = self._get_name_child(node)
                if name:
                    classes.append(ClassStructure(name=name, methods=[]))

            # Function-like definitions
            if ('function' in node_type or 'method' in node_type) and 'definition' in node_type:
                name = self._get_name_child(node)
                if name:
                    functions.append(FunctionSignature(name=name, args=[]))

            # Recurse
            for child in node.children:
                walk(child, depth + 1)

        walk(tree.root_node)

        return FileStructure(
            path=rel_path,
            language=lang,
            classes=classes,
            functions=functions,
            imports=[],
            level="partial"
        )

    def _get_name_child(self, node) -> Optional[str]:
        """Find name/identifier child of a node."""
        for child in node.children:
            if child.type in ('identifier', 'name', 'type_identifier'):
                return child.text.decode('utf-8')
        return None


# =============================================================================
# Main RepoMap Builder
# =============================================================================

class RepoMapBuilder:
    """
    Repository map builder — the main 'gearbox'.

    Automatically selects gear based on project size:
      Gear N: Empty project
      Gear 1: Small (≤10 files) — full signatures
      Gear 2: Medium (≤50 files) — structure + key signatures
      Gear 3: Large (>50 files) — folders + entry points

    Caches result, call invalidate() after file changes.
    """

    # Supported extensions for structure extraction
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.rb': 'ruby',
        '.php': 'php',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
    }

    def __init__(self, project_path: Path, config: Optional[RepoMapConfig] = None):
        """
        Initialize repo map builder.

        Args:
            project_path: Root path of the project
            config: Optional configuration
        """
        self.project_path = Path(project_path).resolve()
        self.config = config or RepoMapConfig()

        # Extractors
        self._grep_ast = GrepASTExtractor()
        self._python_ast = PythonASTExtractor()

        # Cache
        self._cached_map: Optional[str] = None
        self._cached_files: Optional[List[Path]] = None

    def build(self) -> str:
        """
        Build repo map XML block.

        This is the main 'gearbox' method that:
        1. Scans files (with ignore patterns)
        2. Selects gear based on file count
        3. Extracts structure appropriate for gear
        4. Formats as XML

        Returns:
            <repo_map>...</repo_map> XML block
        """
        if self._cached_map is not None:
            return self._cached_map

        # Scan files
        files = self._scan_files()
        self._cached_files = files

        # Select gear and build
        if len(files) == 0:
            result = self._gear_n()
        elif len(files) <= self.config.small_threshold:
            result = self._gear_1(files)
        elif len(files) <= self.config.medium_threshold:
            result = self._gear_2(files)
        else:
            result = self._gear_3(files)

        self._cached_map = result
        return result

    def invalidate(self) -> None:
        """Invalidate cache. Call after file changes."""
        self._cached_map = None
        self._cached_files = None

    def _scan_files(self) -> List[Path]:
        """
        Scan project for source files, respecting ignore patterns.

        Also reads .gitignore if present.
        """
        files = []
        ignore_patterns = list(self.config.ignore_patterns)

        # Read .gitignore
        gitignore = self.project_path / '.gitignore'
        if gitignore.exists():
            try:
                for line in gitignore.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        ignore_patterns.append(line)
            except:
                pass

        # Walk directory
        for path in self.project_path.rglob('*'):
            if not path.is_file():
                continue

            rel_path = path.relative_to(self.project_path)
            rel_str = str(rel_path)

            # Check ignore patterns
            if self._should_ignore(rel_str, ignore_patterns):
                continue

            # Check if we can extract structure
            if path.suffix in self.SUPPORTED_EXTENSIONS:
                files.append(path)

        return sorted(files, key=lambda p: str(p))

    def _should_ignore(self, rel_path: str, patterns: List[str]) -> bool:
        """Check if path matches any ignore pattern."""
        for pattern in patterns:
            # Directory pattern (ends with /)
            if pattern.endswith('/'):
                dir_pattern = pattern.rstrip('/')
                if rel_path.startswith(dir_pattern + '/') or f'/{dir_pattern}/' in f'/{rel_path}/':
                    return True
                # Also check each path component
                for part in Path(rel_path).parts:
                    if fnmatch(part, dir_pattern):
                        return True
            # File pattern
            elif fnmatch(rel_path, pattern) or fnmatch(Path(rel_path).name, pattern):
                return True
        return False

    # =========================================================================
    # Gears
    # =========================================================================

    def _gear_n(self) -> str:
        """Gear N: Empty project."""
        return "<repo_map>\n  (empty project - no source files found)\n</repo_map>"

    def _gear_1(self, files: List[Path]) -> str:
        """
        Gear 1: Small project — full signatures for all files.

        Shows complete structure: classes, methods, functions with types.
        """
        lines = ["<repo_map>"]
        lines.append(f"  <!-- {len(files)} files, full structure -->")
        lines.append("")

        for file_path in files:
            structure = self._extract_structure(file_path)
            formatted = structure.format("  ")
            lines.append(formatted)
            lines.append("")

        lines.append("</repo_map>")
        return "\n".join(lines)

    def _gear_2(self, files: List[Path]) -> str:
        """
        Gear 2: Medium project — structure + key signatures.

        Shows folder structure and most important definitions.
        """
        lines = ["<repo_map>"]
        lines.append(f"  <!-- {len(files)} files, showing key structure -->")
        lines.append("")

        # Group by directory
        by_dir: Dict[str, List[Path]] = {}
        for file_path in files:
            rel_path = file_path.relative_to(self.project_path)
            dir_name = str(rel_path.parent) if rel_path.parent != Path('.') else "(root)"
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(file_path)

        for dir_name in sorted(by_dir.keys()):
            lines.append(f"  {dir_name}/")
            for file_path in by_dir[dir_name]:
                structure = self._extract_structure(file_path)
                # Limit output per file
                lines.append(f"    {file_path.name}")
                for cls in structure.classes[:3]:
                    methods_str = ", ".join(m.name for m in cls.methods[:5])
                    if len(cls.methods) > 5:
                        methods_str += ", ..."
                    lines.append(f"      class {cls.name}: {methods_str}")
                for func in structure.functions[:5]:
                    lines.append(f"      {func.name}()")
            lines.append("")

        lines.append("</repo_map>")
        return "\n".join(lines)

    def _gear_3(self, files: List[Path]) -> str:
        """
        Gear 3: Large project — folders + entry points only.

        Shows directory tree and detected entry points.
        """
        lines = ["<repo_map>"]
        lines.append(f"  <!-- {len(files)} files, showing overview -->")
        lines.append("")

        # Count files per directory
        dir_counts: Dict[str, int] = {}
        for file_path in files:
            rel_path = file_path.relative_to(self.project_path)
            dir_name = str(rel_path.parent) if rel_path.parent != Path('.') else "(root)"
            # Get top-level directory
            parts = dir_name.split('/')
            top_dir = parts[0] if parts[0] != "(root)" else "(root)"
            dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1

        lines.append("  Directory structure:")
        for dir_name in sorted(dir_counts.keys()):
            lines.append(f"    {dir_name}/ ({dir_counts[dir_name]} files)")
        lines.append("")

        # Find entry points
        entry_points = []
        entry_names = ['main.py', 'app.py', 'index.py', 'cli.py', 'run.py',
                       'main.js', 'index.js', 'app.js',
                       'main.go', 'main.rs', 'Main.java']
        for file_path in files:
            if file_path.name in entry_names:
                entry_points.append(file_path)

        if entry_points:
            lines.append("  Entry points:")
            for ep in entry_points[:5]:
                rel = ep.relative_to(self.project_path)
                lines.append(f"    - {rel}")
            lines.append("")

        lines.append(f"  Total: {len(files)} source files")
        lines.append("  Use list_files(path) or read_file(path) for details")
        lines.append("")
        lines.append("</repo_map>")
        return "\n".join(lines)

    # =========================================================================
    # Extraction
    # =========================================================================

    def _extract_structure(self, file_path: Path) -> FileStructure:
        """
        Extract structure from file with fallback chain.

        Try order:
        1. grep-ast (tree-sitter) — most accurate, multi-language
        2. Python AST — good for Python
        3. Regex fallback — works on broken code
        4. Filename only — last resort
        """
        # Try grep-ast first
        if self._grep_ast.available:
            result = self._grep_ast.extract(file_path)
            if result is not None:
                return result

        # Fallback to Python AST for .py files
        if file_path.suffix == '.py':
            return self._python_ast.extract(file_path)

        # For other languages without grep-ast, just return filename
        return FileStructure(
            path=str(file_path.relative_to(self.project_path)),
            language=self.SUPPORTED_EXTENSIONS.get(file_path.suffix, "unknown"),
            classes=[],
            functions=[],
            imports=[],
            level="filename_only",
            error="grep-ast not installed for non-Python files"
        )
