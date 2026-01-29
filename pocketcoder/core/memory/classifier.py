"""
Fact Classifier for Smart Memory Architecture v0.7.0.

Determines whether to save POINTER (command) or VALUE (actual data)
based on tool results and content patterns.

Key insight:
- Reproducible data → POINTER (saves tokens, always fresh)
- Non-reproducible data → VALUE (errors, decisions, metrics)
- External sources → REF (URL + summary, not full content)

v0.7.0 Changes:
- Returns List[ExtractionResult] instead of single result
- Extracts paths after write_file (last_created_folder, last_created_file)
- Extracts visible folders after list_files
- Tracks read files (file_read:{path})
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

from pocketcoder.core.memory.types import FactType, Fact


@dataclass
class ExtractionResult:
    """Result of fact extraction from tool result."""
    key: str
    value: str
    fact_type: FactType
    category: str
    should_save: bool = True
    ttl_days: Optional[int] = None  # v0.7.0: custom TTL


class FactClassifier:
    """
    Classifies facts from tool results.

    Determines:
    1. Should we save this? (not volatile, not sensitive)
    2. What type? (POINTER vs VALUE)
    3. What key/category?
    """

    # Commands that produce reproducible output → save as POINTER
    POINTER_PATTERNS: dict[str, Tuple[str, str, str]] = {
        # pattern: (key, command_to_save, category)
        r"pip (freeze|list)": ("pip_packages", "pip freeze", "project"),
        r"pip3 (freeze|list)": ("pip_packages", "pip3 freeze", "project"),
        r"npm (list|ls)": ("npm_packages", "npm list --depth=0", "project"),
        r"yarn list": ("npm_packages", "yarn list --depth=0", "project"),
        r"go list": ("go_modules", "go list -m all", "project"),
        r"cargo tree": ("cargo_deps", "cargo tree --depth=1", "project"),

        r"git status": ("git_status", "git status --short", "project"),
        r"git branch": ("git_branch", "git branch --show-current", "project"),
        r"git log": ("recent_commits", "git log --oneline -5", "project"),
        r"git remote": ("git_remote", "git remote -v", "project"),

        r"docker ps": ("docker_containers", "docker ps --format 'table {{.Names}}\t{{.Status}}'", "project"),
        r"docker images": ("docker_images", "docker images --format 'table {{.Repository}}\t{{.Tag}}'", "project"),

        r"ls |list_files": ("project_structure", "list_files .", "project"),
        r"tree": ("project_structure", "list_files .", "project"),
    }

    # Patterns indicating errors/failures → save VALUE
    ERROR_PATTERNS: List[str] = [
        r"error",
        r"exception",
        r"failed",
        r"failure",
        r"traceback",
        r"TypeError",
        r"ImportError",
        r"ModuleNotFoundError",
        r"SyntaxError",
        r"NameError",
        r"AttributeError",
        r"KeyError",
        r"ValueError",
        r"RuntimeError",
        r"ConnectionError",
        r"FileNotFoundError",
        r"PermissionError",
        r"command not found",
        r"not found",
        r"permission denied",
        r"ECONNREFUSED",
        r"ENOENT",
    ]

    # Patterns indicating ML metrics → save VALUE
    ML_METRIC_PATTERNS: List[str] = [
        r"RMSE[:\s]+[\d.]+",
        r"MSE[:\s]+[\d.]+",
        r"MAE[:\s]+[\d.]+",
        r"accuracy[:\s]+[\d.]+",
        r"precision[:\s]+[\d.]+",
        r"recall[:\s]+[\d.]+",
        r"F1[:\s]+[\d.]+",
        r"AUC[:\s]+[\d.]+",
        r"loss[:\s]+[\d.]+",
        r"score[:\s]+[\d.]+",
        r"\d+ passed.*\d+ failed",  # pytest results
        r"tests? passed",
        r"tests? failed",
    ]

    # Sensitive patterns → NEVER save value, only pointer to .env
    SENSITIVE_PATTERNS: List[str] = [
        r"api[_-]?key",
        r"secret[_-]?key",
        r"access[_-]?token",
        r"auth[_-]?token",
        r"password",
        r"credential",
        r"private[_-]?key",
        r"Bearer\s+\w+",
        r"sk-[a-zA-Z0-9]+",  # OpenAI keys
        r"ghp_[a-zA-Z0-9]+",  # GitHub tokens
    ]

    # Volatile data → don't save at all
    VOLATILE_PATTERNS: List[str] = [
        r"^\d+$",  # Just a PID
        r"^\d{4}-\d{2}-\d{2}",  # Just a timestamp
        r"cpu[:\s]+\d+%",  # CPU usage
        r"mem[:\s]+\d+%",  # Memory usage
        r"load average",
    ]

    def classify_tool_result(
        self,
        tool_name: str,
        args: dict,
        result: str
    ) -> Optional[ExtractionResult]:
        """
        Classify a tool result and determine what to extract.
        Returns single result for backward compatibility.

        Args:
            tool_name: Name of tool (execute_command, list_files, etc.)
            args: Tool arguments
            result: Tool output

        Returns:
            ExtractionResult or None if nothing to save
        """
        results = self.classify_tool_result_multi(tool_name, args, result)
        return results[0] if results else None

    def classify_tool_result_multi(
        self,
        tool_name: str,
        args: dict,
        result: str
    ) -> List[ExtractionResult]:
        """
        v0.7.0: Classify a tool result and extract MULTIPLE facts.

        Args:
            tool_name: Name of tool (execute_command, list_files, etc.)
            args: Tool arguments
            result: Tool output

        Returns:
            List of ExtractionResult (may be empty)
        """
        extractions: List[ExtractionResult] = []

        # Skip empty results
        if not result or len(result.strip()) < 5:
            return extractions

        # Check for volatile data
        if self._is_volatile(result):
            return extractions

        # Check for sensitive data
        cmd = args.get("cmd", args.get("command", ""))
        if self._is_sensitive(result) or self._is_sensitive(cmd):
            return extractions

        # =================================================================
        # v0.7.0: Tool-specific extractions
        # =================================================================

        # 1. write_file → save created folder and file paths
        if tool_name == "write_file":
            path_extractions = self._extract_write_file_paths(args, result)
            extractions.extend(path_extractions)

        # 2. list_files → save POINTER + visible folders
        elif tool_name == "list_files":
            list_extractions = self._extract_list_files_info(args, result)
            extractions.extend(list_extractions)

        # 3. read_file → track that file was read
        elif tool_name == "read_file":
            read_extractions = self._extract_read_file_info(args, result)
            extractions.extend(read_extractions)

        # 4. execute_command → check for POINTER patterns, errors, metrics
        elif tool_name == "execute_command":
            cmd_extractions = self._extract_command_info(args, result)
            extractions.extend(cmd_extractions)

        # 5. Other tools → legacy single-extraction logic
        else:
            legacy = self._legacy_classify(tool_name, args, result)
            if legacy:
                extractions.append(legacy)

        return extractions

    def _extract_write_file_paths(
        self,
        args: dict,
        result: str
    ) -> List[ExtractionResult]:
        """
        v0.7.0: Extract paths from write_file result.

        Saves:
        - last_created_file: full path
        - last_created_folder: folder part (if exists)
        - working_project: folder name for project tracking
        """
        extractions = []
        path = args.get("path", "")

        if not path:
            return extractions

        # Check if write was successful
        if "Error" in result or "failed" in result.lower():
            return extractions

        # Save full file path
        extractions.append(ExtractionResult(
            key="last_created_file",
            value=path,
            fact_type=FactType.VALUE,
            category="project",
            ttl_days=7,
        ))

        # Extract and save folder
        if "/" in path:
            folder = path.rsplit("/", 1)[0]
            extractions.append(ExtractionResult(
                key="last_created_folder",
                value=folder,
                fact_type=FactType.VALUE,
                category="project",
                ttl_days=7,
            ))

            # Save top-level folder as working project
            top_folder = path.split("/")[0]
            if top_folder and top_folder != ".":
                extractions.append(ExtractionResult(
                    key="working_project",
                    value=top_folder,
                    fact_type=FactType.VALUE,
                    category="project",
                    ttl_days=30,
                ))

        return extractions

    def _extract_list_files_info(
        self,
        args: dict,
        result: str
    ) -> List[ExtractionResult]:
        """
        v0.7.0: Extract info from list_files result.

        Saves:
        - project_structure: POINTER to command
        - visible_folders: comma-separated folder names
        - files_count: number of files/folders
        """
        extractions = []
        path = args.get("path", ".")

        # Always save POINTER
        extractions.append(ExtractionResult(
            key="project_structure",
            value=f"list_files {path}",
            fact_type=FactType.POINTER,
            category="project",
        ))

        # Extract folder names from result
        folders = self._extract_folders_from_tree(result)
        if folders:
            extractions.append(ExtractionResult(
                key="visible_folders",
                value=", ".join(folders[:15]),  # Max 15 folders
                fact_type=FactType.VALUE,
                category="project",
                ttl_days=1,  # Short TTL - structure changes
            ))

        # Count files
        lines = result.strip().split('\n')
        file_count = len([l for l in lines if l.strip() and not l.strip().startswith('[')])
        if file_count > 0:
            extractions.append(ExtractionResult(
                key="files_count",
                value=str(file_count),
                fact_type=FactType.VALUE,
                category="project",
                ttl_days=1,
            ))

        return extractions

    def _extract_folders_from_tree(self, result: str) -> List[str]:
        """Extract folder names from list_files tree output."""
        folders = []
        lines = result.strip().split('\n')

        for line in lines:
            # Remove tree characters: ├── │ └──
            clean = re.sub(r'^[│├└─\s]+', '', line).strip()

            # Folder ends with /
            if clean.endswith('/'):
                folder_name = clean.rstrip('/')
                if folder_name and not folder_name.startswith('.'):
                    folders.append(folder_name)

        return folders

    def _extract_read_file_info(
        self,
        args: dict,
        result: str
    ) -> List[ExtractionResult]:
        """
        v0.7.0: Track that file was read.

        Saves:
        - file_read:{path}: "yes" (TTL 1 day)
        - file_summary:{path}: "{lines} lines, {chars} chars"
        """
        extractions = []
        path = args.get("path", "")

        if not path:
            return extractions

        # Check if read was successful (not an error)
        if "Error" in result or "not found" in result.lower():
            return extractions

        # Mark file as read
        extractions.append(ExtractionResult(
            key=f"file_read:{path}",
            value="yes",
            fact_type=FactType.VALUE,
            category="context",
            ttl_days=1,  # Short TTL - file may change
        ))

        # Save summary
        lines = result.count('\n') + 1
        chars = len(result)
        extractions.append(ExtractionResult(
            key=f"file_summary:{path}",
            value=f"{lines} lines, {chars} chars",
            fact_type=FactType.VALUE,
            category="context",
            ttl_days=1,
        ))

        return extractions

    def _extract_command_info(
        self,
        args: dict,
        result: str
    ) -> List[ExtractionResult]:
        """
        v0.7.0: Extract info from execute_command.

        Handles:
        - POINTER patterns (pip, git, etc.)
        - Errors
        - Test results
        - ML metrics
        """
        extractions = []
        cmd = args.get("cmd", args.get("command", ""))

        # Check for POINTER patterns
        pointer_match = self._match_pointer(cmd, "execute_command")
        if pointer_match:
            key, command, category = pointer_match
            extractions.append(ExtractionResult(
                key=key,
                value=command,
                fact_type=FactType.POINTER,
                category=category,
            ))

        # Check for test results FIRST (before errors)
        test_match = self._extract_test_results(result)
        if test_match:
            extractions.append(ExtractionResult(
                key="test_results",
                value=test_match,
                fact_type=FactType.VALUE,
                category="project",
            ))
            return extractions  # Don't check errors if it's test output

        # Check for errors
        error_match = self._extract_error(result)
        if error_match:
            extractions.append(ExtractionResult(
                key="last_error",
                value=error_match[:200],
                fact_type=FactType.VALUE,
                category="error",
            ))

        # Check for ML metrics
        metric_match = self._extract_metrics(result)
        if metric_match:
            extractions.append(ExtractionResult(
                key="last_metrics",
                value=metric_match,
                fact_type=FactType.VALUE,
                category="project",
            ))

        return extractions

    def _legacy_classify(
        self,
        tool_name: str,
        args: dict,
        result: str
    ) -> Optional[ExtractionResult]:
        """Legacy single-extraction for backward compatibility."""
        cmd = args.get("cmd", args.get("command", ""))

        # Check for POINTER patterns
        pointer_match = self._match_pointer(cmd, tool_name)
        if pointer_match:
            key, command, category = pointer_match
            return ExtractionResult(
                key=key,
                value=command,
                fact_type=FactType.POINTER,
                category=category,
            )

        return None

    def _match_pointer(
        self,
        cmd: str,
        tool_name: str
    ) -> Optional[Tuple[str, str, str]]:
        """Match command against POINTER patterns."""
        # Check tool_name for list_files
        if tool_name == "list_files":
            return ("project_structure", "list_files .", "project")

        # Check command patterns
        for pattern, (key, command, category) in self.POINTER_PATTERNS.items():
            if re.search(pattern, cmd, re.IGNORECASE):
                return (key, command, category)

        return None

    def _extract_error(self, result: str) -> Optional[str]:
        """Extract error message if present."""
        result_lower = result.lower()

        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern, result_lower, re.IGNORECASE):
                # Try to extract meaningful error line
                lines = result.strip().split('\n')

                # Find the most informative error line
                for line in lines:
                    line_lower = line.lower()
                    if any(re.search(p, line_lower, re.IGNORECASE)
                           for p in self.ERROR_PATTERNS):
                        return line.strip()

                # Fallback: first line
                return lines[0].strip() if lines else result[:100]

        return None

    def _extract_metrics(self, result: str) -> Optional[str]:
        """Extract ML metrics if present."""
        metrics = []

        for pattern in self.ML_METRIC_PATTERNS:
            matches = re.findall(pattern, result, re.IGNORECASE)
            metrics.extend(matches)

        if metrics:
            return "; ".join(metrics[:5])  # Max 5 metrics

        return None

    def _extract_test_results(self, result: str) -> Optional[str]:
        """Extract test results summary."""
        # pytest style: "5 passed, 2 failed"
        match = re.search(r"(\d+)\s+passed.*?(\d+)\s+failed", result, re.IGNORECASE)
        if match:
            return f"{match.group(1)} passed, {match.group(2)} failed"

        match = re.search(r"(\d+)\s+passed", result, re.IGNORECASE)
        if match:
            return f"{match.group(1)} passed"

        # unittest style: "OK (5 tests)"
        match = re.search(r"OK\s*\((\d+)\s+tests?\)", result)
        if match:
            return f"{match.group(1)} tests passed"

        return None

    def _is_sensitive(self, text: str) -> bool:
        """Check if text contains sensitive data."""
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_volatile(self, result: str) -> bool:
        """Check if data is too volatile to save."""
        result_stripped = result.strip()

        for pattern in self.VOLATILE_PATTERNS:
            if re.match(pattern, result_stripped, re.IGNORECASE):
                return True

        # Very short results are usually not worth saving
        if len(result_stripped) < 10:
            return True

        return False


def extract_from_conversation(
    user_message: str,
    llm_response: str
) -> Optional[ExtractionResult]:
    """
    Extract facts from conversation (user message + LLM response).

    Used for extracting:
    - Project decisions ("let's use FastAPI")
    - User preferences ("I prefer tabs")
    - Stack choices ("React with TypeScript")
    """
    # Check for explicit stack/framework mentions
    stack_patterns = [
        (r"(?:use|using|chose|choosing|with)\s+(FastAPI|Django|Flask|Express|Next\.?js|React|Vue|Angular)",
         "stack", "project"),
        (r"(?:use|using)\s+(SQLite|PostgreSQL|MySQL|MongoDB|Redis)",
         "database", "project"),
        (r"(?:use|using)\s+(TypeScript|JavaScript|Python|Go|Rust)",
         "language", "project"),
    ]

    combined = f"{user_message} {llm_response}"

    for pattern, key, category in stack_patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return ExtractionResult(
                key=key,
                value=match.group(1),
                fact_type=FactType.VALUE,
                category=category,
            )

    return None
