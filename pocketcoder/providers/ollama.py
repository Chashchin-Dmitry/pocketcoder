"""
Ollama provider implementation.

Uses Ollama's native API (not OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from pocketcoder.core.models import Message, ChatResponse
from pocketcoder.providers.base import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for Ollama LLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5-coder:7b",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> ChatResponse | Iterator[str]:
        """Send chat request to Ollama."""
        url = f"{self.base_url}/api/chat"
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "stream": stream,
        }

        if stream:
            return self._stream_response(url, payload)
        else:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return ChatResponse(
                content=data["message"]["content"],
                finish_reason="stop" if data.get("done") else "length",
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )

    def _stream_response(self, url: str, payload: dict) -> Iterator[str]:
        """Stream response chunks from Ollama."""
        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

    def check_connection(self) -> tuple[bool, str]:
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True, "OK"
        except requests.ConnectionError:
            return False, f"Cannot connect to {self.base_url}"
        except requests.Timeout:
            return False, "Connection timeout"
        except requests.HTTPError as e:
            return False, f"HTTP error: {e}"
        except Exception as e:
            return False, f"Unknown error: {e}"

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
