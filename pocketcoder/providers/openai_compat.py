"""
OpenAI-compatible provider implementation.

Works with vLLM, LM Studio, llama.cpp, and other OpenAI-compatible APIs.
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from pocketcoder.core.models import Message, ChatResponse
from pocketcoder.providers.base import BaseProvider


class OpenAICompatProvider(BaseProvider):
    """Provider for OpenAI-compatible API servers."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        default_model: str = "",
        api_key: str | None = None,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ChatResponse | Iterator[str]:
        """Send chat request to OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"
        model = model or self.default_model

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
            response = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]

            return ChatResponse(
                content=choice["message"]["content"],
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage", {}),
            )

    def _stream_response(self, url: str, payload: dict) -> Iterator[str]:
        """Stream response chunks from OpenAI-compatible API."""
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
        """Check if API server is reachable."""
        try:
            # Try models endpoint
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=5,
            )
            response.raise_for_status()
            return True, "OK"
        except requests.ConnectionError:
            return False, f"Cannot connect to {self.base_url}"
        except requests.Timeout:
            return False, "Connection timeout"
        except requests.HTTPError as e:
            # Some servers don't have /models endpoint but still work
            if e.response.status_code == 404:
                return True, "OK (models endpoint not available)"
            return False, f"HTTP error: {e}"
        except Exception as e:
            return False, f"Unknown error: {e}"

    def list_models(self) -> list[str]:
        """List available models from API."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
