"""
Hooks system for extending PocketCoder.

Available events:
- before_send: Before sending messages to LLM
- after_response: After receiving LLM response
- before_edit: Before applying file changes
- after_edit: After applying file changes
- before_command: Before executing shell command
- after_command: After executing shell command
- file_added: When file is added to chat
- file_removed: When file is removed from chat
- session_start: When session starts
- session_end: When session ends
"""

from pocketcoder.hooks.manager import HookManager

# Global hook manager instance
hooks = HookManager()

__all__ = ["HookManager", "hooks"]
