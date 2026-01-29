"""
Base provider interface for LLM backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from pocketcoder.core.models import Message, ChatResponse


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ChatResponse | Iterator[str]:
        """
        Send messages to LLM and get response.

        Args:
            messages: List of chat messages
            model: Model name (uses default if None)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            stream: If True, yields chunks instead of full response

        Returns:
            ChatResponse or Iterator[str] if streaming
        """
        pass

    @abstractmethod
    def check_connection(self) -> tuple[bool, str]:
        """
        Check if provider is reachable.

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        List available models.

        Returns:
            List of model names
        """
        pass
