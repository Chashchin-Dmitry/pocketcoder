"""
ContentPreview — Universal Gearbox for content preview generation.

v2.4.0: Implements the "gearbox" philosophy for SESSION_CONTEXT.

Gears:
  N (Neutral): None/empty content → return "[empty]"
  1 (Small):   ≤ preview_lines → return as-is (escaped)
  2 (Medium):  > preview_lines → truncate + "... (N lines total)"
  3 (Large):   > file_threshold + save_full=True → save to file + head/tail preview

Usage:
    preview = ContentPreview()

    # For write_file / read_file:
    result = preview.generate(content)

    # For execute_command (may need full output):
    result = preview.generate(output, save_full=True)

    # Access result:
    result.preview      # The preview string (always XML-safe)
    result.full_path    # Path to full content file (if saved)
    result.total_lines  # Total line count
    result.truncated    # Was content truncated?

Flow:
    Any content → generate() → PreviewResult

    Gear is selected automatically:
    - Empty? → N → "[empty]"
    - Few lines? → 1 → full content
    - Many lines? → 2 → truncate
    - Very large + save_full? → 3 → file + head/tail
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import time


@dataclass
class PreviewConfig:
    """
    Configuration for preview generation — the gear ratios.

    Attributes:
        preview_lines: Threshold for Gear 1→2 (default: 50)
        file_threshold: Threshold for Gear 2→3 (default: 100)
        head_lines: Lines at start for head/tail preview (default: 20)
        tail_lines: Lines at end for head/tail preview (default: 20)
        outputs_dir: Directory to save full output files (default: .pocketcoder/outputs)
    """
    preview_lines: int = 50
    file_threshold: int = 100
    head_lines: int = 20
    tail_lines: int = 20
    outputs_dir: str = ".pocketcoder/outputs"


@dataclass
class PreviewResult:
    """
    Result of preview generation.

    Attributes:
        preview: The preview content (XML-escaped, or "[empty]")
        full_path: Path to full content file (if saved, else "")
        total_lines: Total line count in original content
        truncated: Whether content was truncated
    """
    preview: str = ""
    full_path: str = ""
    total_lines: int = 0
    truncated: bool = False


class ContentPreview:
    """
    Universal preview generator — the 'gearbox'.

    Provides consistent preview generation for any text content:
    - write_file content
    - read_file result
    - execute_command output
    - Any future tool output

    All previews are XML-safe (escaped) and informative ("[empty]" for empty).
    """

    def __init__(self, config: Optional[PreviewConfig] = None):
        """
        Initialize ContentPreview with optional config.

        Args:
            config: PreviewConfig instance, or None for defaults
        """
        self.config = config or PreviewConfig()

    @staticmethod
    def escape_xml(s: str) -> str:
        """
        Escape XML special characters.

        Prevents XML injection and ensures valid XML output.

        Args:
            s: String to escape

        Returns:
            XML-safe string with &, <, >, " escaped
        """
        if not s:
            return s
        return (s
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
        )

    def generate(self, content: Optional[str], save_full: bool = False) -> PreviewResult:
        """
        Generate preview for any content.

        This is the main "gearbox" method that automatically selects
        the appropriate gear based on content size and save_full flag.

        Args:
            content: The content to preview (can be None or empty)
            save_full: If True and content is large, save full content to file

        Returns:
            PreviewResult with preview, optional path, and metadata

        Gears:
            N (Neutral): content is None/empty → "[empty]"
            1 (Small):   lines ≤ preview_lines → return as-is (escaped)
            2 (Medium):  lines > preview_lines, save_full=False → truncate
            3 (Large):   lines > file_threshold, save_full=True → file + head/tail
        """
        # =================================================================
        # Gear N: Neutral — empty content
        # =================================================================
        if not content:
            return PreviewResult(
                preview="[empty]",
                total_lines=0,
                truncated=False
            )

        lines = content.split('\n')
        total = len(lines)

        # =================================================================
        # Gear 1: Small — fits in preview, return as-is
        # =================================================================
        if total <= self.config.preview_lines:
            return PreviewResult(
                preview=self.escape_xml(content),
                total_lines=total,
                truncated=False
            )

        # =================================================================
        # Gear 3: Large + save — save to file, return head/tail
        # =================================================================
        if save_full and total > self.config.file_threshold:
            full_path = self._save_to_file(content)
            preview = self._head_tail(lines, total)
            return PreviewResult(
                preview=self.escape_xml(preview),
                full_path=full_path,
                total_lines=total,
                truncated=True
            )

        # =================================================================
        # Gear 2: Medium — just truncate
        # =================================================================
        preview = '\n'.join(lines[:self.config.preview_lines])
        preview += f"\n... ({total} lines total)"
        return PreviewResult(
            preview=self.escape_xml(preview),
            total_lines=total,
            truncated=True
        )

    def _head_tail(self, lines: List[str], total: int) -> str:
        """
        Generate head + tail preview for large content.

        Shows first N lines, then "...", then last N lines.

        Args:
            lines: List of content lines
            total: Total line count

        Returns:
            Preview string with head, separator, and tail
        """
        head = '\n'.join(lines[:self.config.head_lines])
        tail = '\n'.join(lines[-self.config.tail_lines:])
        return f"{head}\n... ({total} lines total) ...\n{tail}"

    def _save_to_file(self, content: str) -> str:
        """
        Save full content to file, return path.

        Creates output directory if needed. Uses timestamp for unique filename.

        Args:
            content: Full content to save

        Returns:
            Path to saved file, or "" if save failed
        """
        try:
            outputs_dir = Path(self.config.outputs_dir)
            outputs_dir.mkdir(parents=True, exist_ok=True)

            # Unique filename with millisecond timestamp
            filename = f"output_{int(time.time() * 1000)}.txt"
            filepath = outputs_dir / filename

            # Write with UTF-8, replace invalid chars
            filepath.write_text(content, encoding='utf-8', errors='replace')
            return str(filepath)
        except (OSError, IOError):
            # Fallback: no file saved
            return ""
