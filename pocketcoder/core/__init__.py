"""
Core module - main logic of PocketCoder.

Contains:
- Coder: Main orchestrator class
- Session: Chat history management
- Parser: SEARCH/REPLACE parsing
- Applier: File modification logic
- Models: Data classes (Edit, Message, FileContext)
"""

from pocketcoder.core.models import Edit, Message, FileContext, ParsedResponse, ChatResponse
from pocketcoder.core.parser import parse_edits, parse_response
from pocketcoder.core.applier import apply_edit, apply_edits_to_content
from pocketcoder.core.coder import Coder

__all__ = [
    "Edit",
    "Message",
    "FileContext",
    "ParsedResponse",
    "ChatResponse",
    "parse_edits",
    "parse_response",
    "apply_edit",
    "apply_edits_to_content",
    "Coder",
]
