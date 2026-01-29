"""
Hook manager for PocketCoder plugin system.

Hooks allow extending PocketCoder without modifying core code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable
import importlib.util
from pathlib import Path


# Available hook events
HOOK_EVENTS = {
    "before_send": "Before sending messages to LLM (messages) -> messages",
    "after_response": "After receiving LLM response (response) -> response",
    "before_edit": "Before applying file changes (file, search, replace) -> (search, replace) | None",
    "after_edit": "After applying file changes (file, old_content, new_content)",
    "before_command": "Before executing shell command (cmd) -> cmd | None",
    "after_command": "After executing shell command (cmd, stdout, stderr, returncode)",
    "file_added": "When file is added to chat (file)",
    "file_removed": "When file is removed from chat (file)",
    "session_start": "When session starts ()",
    "session_end": "When session ends ()",
}


class HookManager:
    """
    Manages hook registration and execution.

    Singleton pattern - use hooks global instance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.hooks: dict[str, list[tuple[int, Callable]]] = defaultdict(list)
        self._executing: set[str] = set()  # For circular dependency detection
        self._initialized = True

    def register(self, event: str, priority: int = 100) -> Callable:
        """
        Decorator to register a hook function.

        Args:
            event: Event name (see HOOK_EVENTS)
            priority: Lower runs first (default 100)

        Usage:
            @hooks.register("after_edit")
            def my_hook(file, old, new):
                print(f"File {file} was edited")
        """

        def decorator(func: Callable) -> Callable:
            if event not in HOOK_EVENTS:
                print(f"Warning: Unknown hook event '{event}'")

            self.hooks[event].append((priority, func))
            # Keep sorted by priority
            self.hooks[event].sort(key=lambda x: x[0])
            return func

        return decorator

    def trigger(self, event: str, *args, **kwargs) -> Any:
        """
        Trigger all hooks for an event.

        Args:
            event: Event name
            *args: Arguments to pass to hooks
            **kwargs: Keyword arguments to pass to hooks

        Returns:
            Modified first argument (for before_* hooks) or None

        For before_* hooks:
            - Return None to cancel the action
            - Return modified value to change it
            - Return original value to continue unchanged
        """
        # Circular dependency protection
        if event in self._executing:
            print(f"Warning: Circular hook detected for '{event}'")
            return args[0] if args else None

        self._executing.add(event)

        try:
            result = args[0] if args else None

            for priority, func in self.hooks[event]:
                try:
                    ret = func(result, *args[1:], **kwargs)

                    # For before_* hooks, None means cancel
                    if ret is None and event.startswith("before_"):
                        return None

                    # Update result if function returned something
                    if ret is not None:
                        result = ret

                except Exception as e:
                    print(f"Warning: Hook {func.__name__} failed: {e}")

            return result

        finally:
            self._executing.discard(event)

    def load_hooks_from_file(self, path: Path) -> bool:
        """
        Load hooks from a Python file.

        The file can use @on decorator which is injected into the module namespace.

        Args:
            path: Path to Python file with hooks

        Returns:
            True if loaded successfully
        """
        try:
            spec = importlib.util.spec_from_file_location("user_hooks", path)
            if spec is None or spec.loader is None:
                return False

            module = importlib.util.module_from_spec(spec)

            # Inject hook manager and decorator into module
            module.hooks = self  # type: ignore
            module.on = self.register  # type: ignore

            spec.loader.exec_module(module)
            return True

        except Exception as e:
            print(f"Warning: Failed to load hooks from {path}: {e}")
            return False

    def clear(self):
        """Clear all registered hooks."""
        self.hooks.clear()

    def list_hooks(self) -> dict[str, list[str]]:
        """List all registered hooks by event."""
        return {
            event: [func.__name__ for _, func in hooks]
            for event, hooks in self.hooks.items()
            if hooks
        }


# Global hook manager instance
hooks = HookManager()
