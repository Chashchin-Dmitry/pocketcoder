"""
Official OpenAI API provider.

For GPT-4, GPT-3.5-turbo, etc.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import requests

from pocketcoder.core.models import Message, ChatResponse
from pocketcoder.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for official OpenAI API."""

    API_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
        timeout: int = 300,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_model = default_model
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model uses reasoning (GPT-5, o1, o3)."""
        return model.startswith(("gpt-5", "o1", "o3"))

    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ChatResponse | Iterator[str]:
        """Send chat request to OpenAI API."""
        url = f"{self.API_URL}/chat/completions"
        model = model or self.default_model

        # GPT-5 and reasoning models use different parameters
        if self._is_reasoning_model(model):
            payload = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "max_completion_tokens": max(max_tokens, 8192),  # reasoning needs more
                "stream": stream,
            }
            # Note: reasoning models don't support temperature
        else:
            payload = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }

        if stream:
            return self._stream_response(url, payload)
        else:
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]

                # Handle empty content (reasoning models may return empty if tokens exhausted)
                content = choice["message"].get("content") or ""

                return ChatResponse(
                    content=content,
                    finish_reason=choice.get("finish_reason", "stop"),
                    usage=data.get("usage", {}),
                )
            except requests.HTTPError as e:
                # Parse error message from OpenAI
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", str(e))
                except Exception:
                    error_msg = str(e)
                raise RuntimeError(f"OpenAI API error: {error_msg}") from e
            except requests.Timeout:
                raise RuntimeError(f"OpenAI API timeout after {self.timeout}s")
            except Exception as e:
                raise RuntimeError(f"OpenAI API error: {e}") from e

    def _stream_response(self, url: str, payload: dict) -> Iterator[str]:
        """Stream response chunks from OpenAI."""
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
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue

    def check_connection(self) -> tuple[bool, str]:
        """Check if OpenAI API is reachable."""
        try:
            response = requests.get(
                f"{self.API_URL}/models",
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            return True, "OK"
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                return False, "Invalid API key"
            return False, f"HTTP error: {e}"
        except Exception as e:
            return False, f"Connection error: {e}"

    def list_models(self) -> list[str]:
        """List available models from OpenAI API."""
        try:
            response = requests.get(
                f"{self.API_URL}/models",
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            # Filter chat-capable models (gpt-*, o1-*, chatgpt-*)
            chat_prefixes = ("gpt-", "o1-", "chatgpt-", "o3-")
            models = [
                m["id"] for m in data.get("data", [])
                if m.get("id", "").startswith(chat_prefixes)
            ]

            # Sort: newest/best first (GPT-5 > GPT-4o > O-series > older)
            priority = ["gpt-5", "gpt-4o", "gpt-4o-mini", "o3", "o1", "gpt-4", "gpt-3.5"]

            def sort_key(name: str) -> tuple[int, str]:
                for i, prefix in enumerate(priority):
                    if name.startswith(prefix):
                        return (i, name)
                return (len(priority), name)

            models.sort(key=sort_key)
            return models if models else self._fallback_models()

        except Exception:
            # Fallback if API fails
            return self._fallback_models()

    def _fallback_models(self) -> list[str]:
        """Fallback model list if API is unavailable."""
        return [
            "gpt-5",
            "gpt-5-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o3-mini",
            "o1-preview",
            "o1-mini",
        ]
