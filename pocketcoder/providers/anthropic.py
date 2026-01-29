"""
Anthropic Claude API provider.

For Claude 3.5 Sonnet, Claude 3 Opus, etc.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import requests

from pocketcoder.core.models import Message, ChatResponse
from pocketcoder.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude API."""

    API_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        timeout: int = 300,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.default_model = default_model
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }

    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ChatResponse | Iterator[str]:
        """Send chat request to Anthropic API."""
        url = f"{self.API_URL}/messages"
        model = model or self.default_model

        # Anthropic uses system separately
        system_msg = ""
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }

        if system_msg:
            payload["system"] = system_msg

        if temperature != 1.0:
            payload["temperature"] = temperature

        if stream:
            payload["stream"] = True
            return self._stream_response(url, payload)
        else:
            response = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            if data.get("content"):
                for block in data["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")

            return ChatResponse(
                content=content,
                finish_reason=data.get("stop_reason", "stop"),
                usage={
                    "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                },
            )

    def _stream_response(self, url: str, payload: dict) -> Iterator[str]:
        """Stream response chunks from Anthropic."""
        with requests.post(
            url,
            json=payload,
            headers=self._headers(),
            stream=True,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue

    def check_connection(self) -> tuple[bool, str]:
        """Check if Anthropic API is reachable."""
        # Anthropic doesn't have a simple health check endpoint
        # Try a minimal API call
        try:
            response = requests.post(
                f"{self.API_URL}/messages",
                json={
                    "model": self.default_model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code == 401:
                return False, "Invalid API key"
            # Any response means connection works
            return True, "OK"
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                return False, "Invalid API key"
            return False, f"HTTP error: {e}"
        except Exception as e:
            return False, f"Connection error: {e}"

    def list_models(self) -> list[str]:
        """List available Claude models."""
        return [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
